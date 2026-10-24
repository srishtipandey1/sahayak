"""Sentinel-2 scene screening, selection, and common-grid ingestion."""

from __future__ import annotations

import logging
import math
import tempfile
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.shutil import copy as raster_copy

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.ingest.s1 import (
    NODATA,
    SentinelHubClient,
    _format_utc,
)
from sahayak.provenance import RunManifest
from sahayak.quicklook import write_png_atomically

from .s2_catalog import build_s2_catalog_request, parse_s2_catalog_item

LOGGER = logging.getLogger(__name__)
S2_SOURCE = "Copernicus Data Space Sentinel-2 L2A via Sentinel Hub Process API"
S2_LICENSE = "Copernicus free, full and open data policy"
S2_ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026."
S2_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask")
S2_DESCRIPTIONS = (
    "B02 blue surface reflectance",
    "B03 green surface reflectance",
    "B04 red surface reflectance",
    "B08 near-infrared surface reflectance",
    "B11 shortwave infrared 1 surface reflectance",
    "B12 shortwave infrared 2 surface reflectance",
    "Scene classification layer (unitless class code)",
    "Data mask (1 valid data, 0 no data)",
)
S2_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"],
      units: ["REFLECTANCE", "REFLECTANCE", "REFLECTANCE", "REFLECTANCE",
              "REFLECTANCE", "REFLECTANCE", "DN", "DN"]
    }],
    output: { bands: 8, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B08,
          sample.B11, sample.B12, sample.SCL, sample.dataMask];
}
"""
SCL_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["SCL"], units: ["DN"] }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.SCL];
}
"""
SCL_CLASSES = {
    "clear": (4, 5, 6, 7),
    "dark_topographic_shadow": (2,),
    "cloud_shadow": (3,),
    "cloud_and_cirrus": (8, 9, 10),
    "snow_or_ice": (11,),
    "no_data_or_defective": (0, 1),
}


def deduplicate_s2_scenes(
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the lowest-cloud-cover item per acquisition date and report duplicates."""
    by_date: dict[str, list[dict[str, Any]]] = {}
    for scene in scenes:
        acquired = scene["acquired_at"]
        by_date.setdefault(acquired.date().isoformat(), []).append(scene)
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for acquisition_date, daily_scenes in sorted(by_date.items()):
        ordered = sorted(
            daily_scenes,
            key=lambda scene: (
                scene["eo_cloud_cover"]
                if scene["eo_cloud_cover"] is not None
                else math.inf,
                scene["id"],
            ),
        )
        kept.append(ordered[0])
        duplicates.extend(
            {
                "date": acquisition_date,
                "kept_id": ordered[0]["id"],
                "duplicate_id": duplicate["id"],
                "reason": "Same-date duplicate; item with lower eo:cloud_cover retained.",
            }
            for duplicate in ordered[1:]
        )
    return kept, duplicates


def choose_s2_scenes(
    scenes: list[dict[str, Any]],
    event_date: date,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Choose pre/post acquisitions using clear-fraction rules and stable fallbacks."""
    pre = [scene for scene in scenes if scene["acquired_at"].date() < event_date]
    post = [scene for scene in scenes if scene["acquired_at"].date() > event_date]
    if not pre or not post:
        raise ValueError(
            "Sentinel-2 selection requires evaluated scenes before and after event date."
        )
    eligible_pre = [scene for scene in pre if scene["clear_fraction"] >= 0.5]
    eligible_post = [scene for scene in post if scene["clear_fraction"] >= 0.8]
    if eligible_pre:
        chosen_pre = max(
            eligible_pre,
            key=lambda scene: (scene["acquired_at"], -_cloud_rank(scene)),
        )
    else:
        chosen_pre = max(
            pre,
            key=lambda scene: (
                scene["clear_fraction"],
                -_cloud_rank(scene),
                scene["acquired_at"],
            ),
        )
    chosen_post = min(
        eligible_post,
        key=lambda scene: (
            scene["acquired_at"],
            _cloud_rank(scene),
        ),
    ) if eligible_post else max(
        post,
        key=lambda scene: (
            scene["clear_fraction"],
            -_cloud_rank(scene),
            -scene["acquired_at"].timestamp(),
        ),
    )
    return chosen_pre, chosen_post, {
        "pre": (
            "Latest pre-event scene with clear fraction >= 0.5."
            if eligible_pre
            else "Fallback: highest clear fraction among evaluated pre-event scenes."
        ),
        "post": (
            "Earliest post-event scene with clear fraction >= 0.8."
            if eligible_post
            else "Fallback: highest clear fraction among evaluated post-event scenes."
        ),
    }


def scl_fractions(values: np.ndarray) -> dict[str, Any]:
    """Return pixel counts and fractions for clear and major SCL categories."""
    classes = np.asarray(values)
    finite = np.isfinite(classes)
    total = int(finite.sum())
    if total == 0:
        raise ValueError("Sentinel-2 SCL response contains no finite class values.")
    result: dict[str, Any] = {"pixel_count": total, "fractions": {}}
    for name, codes in SCL_CLASSES.items():
        count = int((finite & np.isin(classes, codes)).sum())
        result["fractions"][name] = {
            "count": count,
            "fraction": count / total if total else None,
        }
    known = tuple(code for values in SCL_CLASSES.values() for code in values)
    other_count = int((finite & ~np.isin(classes, known)).sum())
    result["fractions"]["other"] = {
        "count": other_count,
        "fraction": other_count / total if total else None,
    }
    result["clear_fraction"] = result["fractions"]["clear"]["fraction"]
    return result


def _cloud_rank(scene: dict[str, Any]) -> float:
    cover = scene.get("eo_cloud_cover")
    return float(cover) if cover is not None else math.inf


def _catalog_scenes(
    bbox: tuple[float, float, float, float],
    event_date: date,
    client: SentinelHubClient,
) -> list[dict[str, Any]]:
    """Fetch every Sentinel-2 Catalog page in the requested temporal window."""
    start = event_date - timedelta(days=45)
    end = event_date + timedelta(days=70)
    token: int | None = None
    scenes: list[dict[str, Any]] = []
    while True:
        page = client.catalog_search(
            build_s2_catalog_request(bbox, start, end, next_page=token)
        )
        features = page.get("features")
        if not isinstance(features, list):
            raise RuntimeError("Sentinel-2 Catalog response has no features list.")
        for feature in features:
            if isinstance(feature, dict):
                scenes.append(parse_s2_catalog_item(feature))
        context = page.get("context")
        next_token = context.get("next") if isinstance(context, dict) else None
        if not next_token:
            break
        if not isinstance(next_token, int) or isinstance(next_token, bool) or next_token == token:
            raise RuntimeError("Sentinel-2 Catalog returned an invalid pagination token.")
        token = next_token
    return scenes


def _process_request(
    *,
    scene: dict[str, Any],
    grid: GridSpec,
    bands: tuple[str, ...],
    evalscript: str,
    width: int,
    height: int,
) -> dict[str, object]:
    acquired_at = scene["acquired_at"]
    epsg_crs = f"http://www.opengis.net/def/crs/EPSG/0/{grid.epsg}"
    resampling = "NEAREST" if bands == ("SCL",) else "BILINEAR"
    return {
        "input": {
            "bounds": {
                "bbox": list(grid.bounds),
                "properties": {"crs": epsg_crs},
            },
            "data": [
                {
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": {
                            "from": _format_utc(acquired_at - timedelta(seconds=1)),
                            "to": _format_utc(acquired_at + timedelta(seconds=1)),
                        },
                        "mosaickingOrder": "mostRecent",
                    },
                    "processing": {
                        "upsampling": resampling,
                        "downsampling": "NEAREST",
                    },
                }
            ],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
        "requested_bands": list(bands),
    }


def _read_process_tiff(payload: bytes, band_count: int, width: int, height: int) -> np.ndarray:
    with rasterio.MemoryFile(payload) as memory_file:
        with memory_file.open() as dataset:
            if dataset.count != band_count or (dataset.width, dataset.height) != (width, height):
                raise RuntimeError(
                    "Sentinel-2 Process response has dimensions/bands "
                    f"{dataset.width}x{dataset.height}/{dataset.count}; expected "
                    f"{width}x{height}/{band_count}."
                )
            return dataset.read(out_dtype="float32")


def _write_s2_cog(path: Path, data: np.ndarray, grid: GridSpec) -> None:
    if data.shape != (len(S2_BANDS), grid.height, grid.width):
        raise ValueError(f"Sentinel-2 array shape {data.shape} does not match common grid.")
    output = np.where(np.isfinite(data), data, NODATA).astype(np.float32)
    profile = grid.rasterio_profile(dtype="float32", nodata=NODATA, count=len(S2_BANDS))
    profile.update(compress="deflate", predictor=3, tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_",
        suffix=".source.tif",
        dir=path.parent,
        delete=False,
    ) as temporary_file:
        source_path = Path(temporary_file.name)
    cog_path = source_path.with_name(f"{source_path.stem}.cog.tif")
    try:
        with rasterio.open(source_path, "w", **profile) as destination:
            destination.write(output)
            for index, description in enumerate(S2_DESCRIPTIONS, start=1):
                destination.set_band_description(index, description)
        raster_copy(
            source_path,
            cog_path,
            driver="COG",
            compress="DEFLATE",
            predictor="FLOATING_POINT",
            nodata=NODATA,
        )
        cog_path.replace(path)
    finally:
        source_path.unlink(missing_ok=True)
        cog_path.unlink(missing_ok=True)


def ingest_s2(
    *,
    bbox: tuple[float, float, float, float],
    event_date: date,
    grid: GridSpec,
    settings: Settings,
    cache: RequestCache,
    offline: bool,
    output_dir: Path,
    manifest: RunManifest,
    osm_path: Path,
    streams_path: Path | None,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    session: Any | None = None,
) -> dict[str, Any]:
    """Screen candidates by SCL, ingest selected reflectance images, and write overlays."""
    client = SentinelHubClient(settings, cache, offline=offline, session=session)
    scenes, duplicates = deduplicate_s2_scenes(_catalog_scenes(bbox, event_date, client))
    pre_candidates = [
        item
        for item in scenes
        if item["acquired_at"].date() < event_date
    ]
    post_candidates = [
        item
        for item in scenes
        if item["acquired_at"].date() > event_date
    ]
    pre_candidates = sorted(
        pre_candidates, key=lambda scene: (_cloud_rank(scene), scene["id"])
    )[:6]
    post_candidates = sorted(
        post_candidates, key=lambda scene: (_cloud_rank(scene), scene["id"])
    )[:8]
    evaluation_candidates = sorted(
        [*pre_candidates, *post_candidates],
        key=lambda scene: (scene["acquired_at"], _cloud_rank(scene)),
    )
    evaluated_ids = {scene["id"] for scene in evaluation_candidates}
    scl_width = max(1, math.ceil(grid.width * grid.resolution / 60.0))
    scl_height = max(1, math.ceil(grid.height * grid.resolution / 60.0))
    scene_results: list[dict[str, Any]] = [
        {
            **scene,
            "acquired_at": scene["acquired_at"].isoformat(),
            "phase": (
                "pre-event"
                if scene["acquired_at"].date() < event_date
                else "post-event"
                if scene["acquired_at"].date() > event_date
                else "event-date"
            ),
            "evaluated": scene["id"] in evaluated_ids,
            "optical_usable": None,
            "selection_reason": (
                "Awaiting SCL screening."
                if scene["id"] in evaluated_ids
                else "Not SCL-screened: outside the lowest-cloud-cover candidate limit "
                "or acquired on the event date."
            ),
        }
        for scene in sorted(scenes, key=lambda item: item["acquired_at"])
    ]
    result_by_id = {record["id"]: record for record in scene_results}
    selection: dict[str, object] = {
        "provider": "sentinel-hub-process-api",
        "catalog_collection": "sentinel-2-l2a",
        "catalog_search_window_days": {"before": 45, "after": 70},
        "catalog_item_count_after_deduplication": len(scenes),
        "same_date_duplicates": duplicates,
        "candidates_evaluated_limit": {"pre_event": 6, "post_event": 8},
        "scenes": scene_results,
        "attribution": S2_ATTRIBUTION,
    }
    if checkpoint is not None:
        checkpoint({"s2": selection})
    for scene in evaluation_candidates:
        request_body = _process_request(
            scene=scene,
            grid=grid,
            bands=("SCL",),
            evalscript=SCL_EVALSCRIPT,
            width=scl_width,
            height=scl_height,
        )
        request_body.pop("requested_bands")
        before_units = len(client.processing_units)
        scl = _read_process_tiff(
            client.process(request_body),
            band_count=1,
            width=scl_width,
            height=scl_height,
        )[0]
        fractions = scl_fractions(scl)
        is_pre = scene["acquired_at"].date() < event_date
        required_clear = 0.5 if is_pre else 0.8
        record = result_by_id[scene["id"]]
        record.update(
            {
                **fractions,
                "optical_usable": fractions["clear_fraction"] >= required_clear,
                "usability_threshold": required_clear,
                "scl_processing_units": _processing_units_since(client, before_units),
                "selection_reason": "SCL screened; candidate alternative.",
            }
        )
        if checkpoint is not None:
            checkpoint({"s2": selection})

    evaluated_scenes = [
        {**record, "acquired_at": datetime.fromisoformat(record["acquired_at"])}
        for record in scene_results
        if record["evaluated"]
    ]
    selected_pre, selected_post, choice_reasons = choose_s2_scenes(
        evaluated_scenes,
        event_date,
    )
    selected_pre_id = selected_pre["id"]
    selected_post_id = selected_post["id"]
    selection["selection_reasons"] = choice_reasons
    selection["selected"] = {
        "pre": _selected_record(selected_pre, "s2_pre.tif", choice_reasons["pre"]),
        "post": _selected_record(selected_post, "s2_post.tif", choice_reasons["post"]),
    }
    for record in scene_results:
        if record["id"] in {selected_pre_id, selected_post_id}:
            record["selection_reason"] = (
                choice_reasons["pre"] if record["id"] == selected_pre_id else choice_reasons["post"]
            )
    selection["alternatives"] = [
        {
            "id": record["id"],
            "acquired_at": record["acquired_at"],
            "phase": record["phase"],
            "clear_fraction": record.get("clear_fraction"),
            "optical_usable": record["optical_usable"],
            "selection_reason": record["selection_reason"],
        }
        for record in scene_results
        if record["evaluated"] and record["id"] not in {selected_pre_id, selected_post_id}
    ]
    if checkpoint is not None:
        checkpoint({"s2": selection})

    image_results: dict[str, dict[str, Any]] = {}
    for scene, filename in (
        (selected_pre, "s2_pre.tif"),
        (selected_post, "s2_post.tif"),
    ):
        request_body = _process_request(
            scene=scene,
            grid=grid,
            bands=S2_BANDS,
            evalscript=S2_EVALSCRIPT,
            width=grid.width,
            height=grid.height,
        )
        request_body.pop("requested_bands")
        before_units = len(client.processing_units)
        data = _read_process_tiff(
            client.process(request_body),
            band_count=len(S2_BANDS),
            width=grid.width,
            height=grid.height,
        )
        output_path = output_dir / filename
        _write_s2_cog(output_path, data, grid)
        manifest.inputs = [item for item in manifest.inputs if item.name != filename]
        manifest.add_input(
            name=filename,
            role="input",
            source=S2_SOURCE,
            license=S2_LICENSE,
            retrieved_utc=datetime.now(UTC),
            path=output_path,
            settings=settings,
        )
        image_results[filename] = {
            "scene_id": scene["id"],
            "acquired_at": scene["acquired_at"].isoformat(),
            "platform": scene.get("platform"),
            "tile_id": scene.get("tile_id"),
            "relative_orbit": scene.get("relative_orbit"),
            "scl_fractions": scene["fractions"],
            "processing_units": _processing_units_since(client, before_units),
            "path": output_path,
        }
        if checkpoint is not None:
            checkpoint(
                {
                    "s2": selection,
                    "completed_artifacts": [
                        *selection.get("completed_artifacts", []),
                        filename,
                    ],
                }
            )

    quicklook_dir = output_dir / "optical"
    quicklook_dir.mkdir(parents=True, exist_ok=True)
    failed_quicklooks = _write_s2_quicklooks(
        quicklook_dir,
        image_results["s2_pre.tif"]["path"],
        image_results["s2_post.tif"]["path"],
        grid,
        osm_path,
        streams_path,
    )
    processing_units = [
        float(value) for value in client.processing_units
    ]
    return {
        "selection": selection,
        "files": image_results,
        "catalog_request_count": client.catalog_request_count,
        "process_request_count": client.process_request_count,
        "processing_units": sum(processing_units) if processing_units else None,
        "failed_quicklooks": failed_quicklooks,
    }


def _processing_units_since(client: SentinelHubClient, index: int) -> float | None:
    values = client.processing_units[index:]
    return float(sum(values)) if values else None


def _selected_record(scene: dict[str, Any], filename: str, reason: str) -> dict[str, Any]:
    return {
        "id": scene["id"],
        "file": filename,
        "acquired_at": scene["acquired_at"].isoformat(),
        "tile_id": scene.get("tile_id"),
        "relative_orbit": scene.get("relative_orbit"),
        "eo_cloud_cover": scene.get("eo_cloud_cover"),
        "clear_fraction": scene["clear_fraction"],
        "optical_usable": scene["optical_usable"],
        "selection_reason": reason,
    }


def _write_s2_quicklooks(
    directory: Path,
    pre_path: Path,
    post_path: Path,
    grid: GridSpec,
    osm_path: Path,
    streams_path: Path,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    pre = _read_s2_output(pre_path)
    post = _read_s2_output(post_path)
    roads, settlements = _read_s2_annotations(osm_path, grid.crs)
    if streams_path is None:
        streams = gpd.GeoDataFrame(geometry=[], crs=grid.crs)
    else:
        streams = gpd.read_file(streams_path, layer="streams")
        if streams.crs is None:
            raise ValueError("Stream network has no CRS for Sentinel-2 quicklooks.")
        streams = streams.to_crs(grid.crs)
    bounds = grid.bounds
    failed: list[str] = []
    pre_valid = (pre[7] > 0) & (pre[7] != NODATA)
    fixed_stretch = []
    for index in (2, 1, 0):
        samples = pre[index][pre_valid & np.isfinite(pre[index]) & (pre[index] != NODATA)]
        if samples.size == 0:
            raise ValueError("Pre-event Sentinel-2 image contains no valid reflectance samples.")
        fixed_stretch.append(np.nanpercentile(samples, [2, 98]))
    for filename, data in (("s2_pre_truecolor.png", pre), ("s2_post_truecolor.png", post)):
        figure, axis = plt.subplots(figsize=(10, 8))
        rgb = _true_color(data, fixed_stretch)
        axis.imshow(rgb, extent=(bounds[0], bounds[2], bounds[1], bounds[3]))
        scl = data[6]
        masked = np.isin(scl, (2, 3, 8, 9, 10, 11))
        shade = np.zeros((*scl.shape, 4), dtype=np.float32)
        shade[..., :3] = 0.55
        shade[..., 3] = masked.astype(np.float32) * 0.55
        axis.imshow(shade, extent=(bounds[0], bounds[2], bounds[1], bounds[3]))
        _plot_s2_annotations(axis, roads, settlements, streams)
        axis.set_title(filename.replace("_", " ").removesuffix(".png"))
        axis.legend(
            handles=[
                Patch(facecolor="#888888", alpha=0.55, label="Cloud/shadow/snow SCL"),
                Patch(facecolor="yellow", label="Pre-event OSM roads"),
                Patch(facecolor="#00a6ff", label="Stream network"),
            ],
            loc="upper right",
            fontsize=8,
        )
        _s2_figure_footer(axis, figure, grid)
        if not write_png_atomically(
            directory / filename,
            lambda path, fig=figure: fig.savefig(path, dpi=150),
        ):
            failed.append(filename)
        plt.close(figure)

    settlement = _find_syapru(settlements)
    centre = settlement.geometry.representative_point()
    crop = _crop_window(grid, centre.x, centre.y, 1500.0)
    row_slice, column_slice, crop_bounds = crop
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, data, title in (
        (axes[0], pre, "Pre-event B04"),
        (axes[1], post, "Post-event B04"),
        (
            axes[2],
            np.where(
                (post[2] != NODATA) & (pre[2] != NODATA),
                post[2] - pre[2],
                NODATA,
            ),
            "Post minus pre B04",
        ),
    ):
        if data.ndim == 3:
            layer = data[2, row_slice, column_slice]
        else:
            layer = data[row_slice, column_slice]
        image = axis.imshow(
            np.ma.masked_where(layer == NODATA, layer),
            extent=(crop_bounds[0], crop_bounds[2], crop_bounds[1], crop_bounds[3]),
            cmap="RdBu_r" if "minus" in title else "gray",
        )
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.75)
    _s2_figure_footer(axes[0], figure, grid)
    zoom_path = directory / "s2_side_by_side_syapru.png"
    if not write_png_atomically(zoom_path, lambda path: figure.savefig(path, dpi=150)):
        failed.append(zoom_path.name)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    class_colors = [
        "#000000",
        "#ff00ff",
        "#5a5a5a",
        "#aa0000",
        "#00a000",
        "#aa7e00",
        "#0000ff",
        "#808080",
        "#f08080",
        "#ff0000",
        "#8000ff",
        "#00ffff",
    ]
    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(np.arange(-0.5, 12.5, 1), cmap.N)
    image = axis.imshow(
        post[6],
        cmap=cmap,
        norm=norm,
        extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
    )
    figure.colorbar(image, ax=axis, ticks=list(range(12)), label="SCL class")
    axis.set_title("Sentinel-2 post-event scene classification")
    _s2_figure_footer(axis, figure, grid)
    overview_path = directory / "s2_overview_valid.png"
    if not write_png_atomically(
        overview_path,
        lambda path: figure.savefig(path, dpi=150),
    ):
        failed.append(overview_path.name)
    plt.close(figure)
    return failed


def _read_s2_output(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read(out_dtype="float32")


def _true_color(data: np.ndarray, stretch: list[np.ndarray]) -> np.ndarray:
    rgb = np.stack((data[2], data[1], data[0]), axis=-1)
    output = np.zeros_like(rgb, dtype=np.float32)
    for index, limits in enumerate(stretch):
        low, high = float(limits[0]), float(limits[1])
        if high <= low:
            high = low + 1e-6
        output[..., index] = np.clip((rgb[..., index] - low) / (high - low), 0.0, 1.0)
    return output


def _read_s2_annotations(
    osm_path: Path,
    crs: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    roads = gpd.read_file(osm_path, layer="roads")
    settlements = gpd.read_file(osm_path, layer="settlements")
    if roads.crs is None or settlements.crs is None:
        raise ValueError("OSM roads or settlements layer has no CRS.")
    return roads.to_crs(crs), settlements.to_crs(crs)


def _plot_s2_annotations(
    axis: Any,
    roads: gpd.GeoDataFrame,
    settlements: gpd.GeoDataFrame,
    streams: gpd.GeoDataFrame,
) -> None:
    if not roads.empty:
        roads.plot(ax=axis, color="yellow", linewidth=0.45, zorder=4)
    if not streams.empty:
        streams.boundary.plot(ax=axis, color="#00a6ff", linewidth=0.45, zorder=5)
    if not settlements.empty:
        settlements.plot(ax=axis, color="black", markersize=8, zorder=6)


def _find_syapru(settlements: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "name" not in settlements.columns:
        raise ValueError("OSM settlements layer has no name column.")
    names = settlements["name"].astype("string").str.casefold()
    matches = settlements[names.isin(("स्याप्रु बेसी".casefold(), "syapru besi"))]
    if matches.empty:
        raise ValueError("OSM settlements layer has no Syapru Besi (स्याप्रु बेसी) feature.")
    return matches.iloc[[0]]


def _crop_window(
    grid: GridSpec,
    x: float,
    y: float,
    radius_m: float,
) -> tuple[slice, slice, tuple[float, float, float, float]]:
    column = int((x - grid.left) / grid.resolution)
    row = int((grid.top - y) / grid.resolution)
    radius = math.ceil(radius_m / grid.resolution)
    row_start = max(0, row - radius)
    row_end = min(grid.height, row + radius)
    column_start = max(0, column - radius)
    column_end = min(grid.width, column + radius)
    if row_start >= row_end or column_start >= column_end:
        raise ValueError("Syapru Besi crop does not intersect the common raster grid.")
    bounds = (
        grid.left + column_start * grid.resolution,
        grid.top - row_end * grid.resolution,
        grid.left + column_end * grid.resolution,
        grid.top - row_start * grid.resolution,
    )
    return slice(row_start, row_end), slice(column_start, column_end), bounds


def _s2_figure_footer(axis: Any, figure: Any, grid: GridSpec) -> None:
    axis.set_xlabel(grid.crs)
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    x_start = min(x_limits) + abs(x_limits[1] - x_limits[0]) * 0.05
    y_position = min(y_limits) + abs(y_limits[1] - y_limits[0]) * 0.06
    axis.plot(
        [x_start, x_start + 500.0],
        [y_position, y_position],
        color="white",
        linewidth=3,
        zorder=10,
    )
    axis.text(
        x_start + 250.0,
        y_position,
        "500 m",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=10,
    )
    axis.text(
        0.01,
        0.015,
        S2_ATTRIBUTION,
        transform=axis.transAxes,
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    figure.tight_layout()
