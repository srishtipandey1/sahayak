"""Optical change detection orchestration and run products."""

import json
import logging
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize, shapes
from shapely.geometry import shape

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id
from sahayak.optical.config import OpticalConfig, load_optical_config
from sahayak.optical.detect import (
    CLASS_EXISTING_CHANNEL_OR_BARE,
    CLASS_INVALID,
    CLASS_NAMES,
    CLASS_NEW_WATER,
    CLASS_SCOUR_DEBRIS,
    OpticalDetection,
    detect_optical,
    false_alarm_accounting,
)
from sahayak.optical.preprocess import BANDS, OpticalPair, prepare_pair
from sahayak.provenance import RunManifest
from sahayak.quicklook import write_png_atomically

LOGGER = logging.getLogger(__name__)
ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026."


def run_optical(
    *,
    run_id: str,
    max_hand_m: float | None = None,
    config_path: Path = Path("configs/optical.toml"),
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run optical change detection for pre, post, and placebo Sentinel-2 scenes."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    selection_path = run_dir / "selection.json"
    manifest_path = run_dir / "manifest.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Run {run_id!r} lacks selection.json or manifest.json.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    s2_selection = selection.get("s2", {}).get("selected", {})
    if not all(name in s2_selection for name in ("pre", "pre_2", "post")):
        raise ValueError("Run selection.json must contain selected S2 pre, pre_2, and post scenes.")

    resolved_config = config_path
    if not resolved_config.is_absolute():
        resolved_config = active_settings.env_file_path.parent / resolved_config
    config = load_optical_config(resolved_config.resolve())
    selected_hand = config.max_hand_m if max_hand_m is None else max_hand_m
    if not math.isfinite(selected_hand) or selected_hand <= 0:
        raise ValueError("--max-hand must be a positive finite number of metres.")
    config = replace(config, max_hand_m=selected_hand)

    paths = {key: run_dir / record["file"] for key, record in s2_selection.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Optical input is missing: {path.name}")
    hand_path = run_dir / "flood" / "hand.tif"
    slope_path = run_dir / "slope.tif"
    streams_path = run_dir / "flood" / "streams.gpkg"
    osm_path = run_dir / "osm_pre_event.gpkg"
    for path in (hand_path, slope_path, streams_path, osm_path):
        if not path.is_file():
            raise FileNotFoundError(f"Optical terrain or OSM input is missing: {path.name}")

    scenes = {key: _read_s2_scene(path) for key, path in paths.items()}
    profile = scenes["pre"]["profile"]
    height, width = scenes["pre"]["arrays"]["SCL"].shape
    if any(scene["arrays"]["SCL"].shape != (height, width) for scene in scenes.values()):
        raise ValueError("Sentinel-2 optical inputs are not aligned to a common grid.")
    hand = _read_aligned_single_band(hand_path, profile, height, width)
    slope = _read_aligned_single_band(slope_path, profile, height, width)
    with rasterio.open(hand_path) as source:
        pixel_size = abs(source.transform.a)
    streams_gdf = gpd.read_file(streams_path, layer="streams")
    if streams_gdf.crs is None:
        raise ValueError("Flood stream layer has no CRS.")
    streams_gdf = streams_gdf.to_crs(profile["crs"])
    stream_mask = rasterize(
        ((geometry, 1) for geometry in streams_gdf.geometry if geometry is not None),
        out_shape=(height, width),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    real_pair = _prepare(scenes["pre"], scenes["post"], hand, slope, config)
    null_pair = _prepare(scenes["pre_2"], scenes["pre"], hand, slope, config)
    null_detection = detect_optical(
        null_pair,
        hand,
        slope,
        stream_mask,
        config,
        pixel_size_m=pixel_size,
    )
    detection = detect_optical(
        real_pair,
        hand,
        slope,
        stream_mask,
        config,
        pixel_size_m=pixel_size,
        sigma_null=null_detection.sigma_null,
    )
    null_area = null_detection.class_areas_ha["scour_debris"]
    real_area = detection.class_areas_ha["scour_debris"]
    reliability, reliability_ratio = reliability_flag(null_area, real_area)

    output_dir = run_dir / "optical"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "optical_class.tif": output_dir / "optical_class.tif",
        "optical_confidence.tif": output_dir / "optical_confidence.tif",
        "delta_ndvi.tif": output_dir / "delta_ndvi.tif",
        "delta_brightness.tif": output_dir / "delta_brightness.tif",
        "optical_polygons.gpkg": output_dir / "optical_polygons.gpkg",
    }
    _write_raster(
        outputs["optical_class.tif"],
        detection.classes,
        profile,
        "uint8",
        CLASS_INVALID,
    )
    _write_raster(
        outputs["optical_confidence.tif"],
        detection.confidence,
        profile,
        "float32",
        -9999.0,
    )
    _write_raster(
        outputs["delta_ndvi.tif"],
        detection.delta_ndvi,
        profile,
        "float32",
        -9999.0,
    )
    _write_raster(
        outputs["delta_brightness.tif"],
        detection.delta_brightness,
        profile,
        "float32",
        -9999.0,
    )
    _write_polygons(outputs["optical_polygons.gpkg"], detection, profile)

    null_false_alarms = false_alarm_accounting(
        null_detection.classes,
        null_detection.valid,
        hand,
        pixel_size * pixel_size / 10_000.0,
    )
    sensitivity = _sensitivity(
        real_pair,
        null_detection.sigma_null,
        hand,
        slope,
        stream_mask,
        config,
        pixel_size,
    )
    radar_corroboration = _radar_corroboration(
        run_dir,
        detection,
        hand,
        slope,
        config,
    )
    facts = {
        "module": "sahayak.optical.run",
        "provenance": {
            "input_files": [path.name for path in (*paths.values(), hand_path, slope_path)],
            "scene_dates": {
                key: s2_selection[key]["acquired_at"] for key in ("pre", "pre_2", "post")
            },
            "thresholds_used": _thresholds(config),
            "normalization_method": (
                "Per-band median-ratio scale followed by median offset, fit to valid stable "
                "pixels (HAND > 100 m, slope <= 35 degrees) after excluding the 2% tails "
                "of each band's post-minus-pre change distribution."
            ),
        },
        "class_areas_ha": _measurements(detection.class_areas_ha, "ha"),
        "null_pair": {
            "class_areas_ha": _measurements(null_detection.class_areas_ha, "ha"),
            "false_alarm_accounting": null_false_alarms,
            "sigma_null": detection.sigma_null,
            "funnels": {"real": detection.funnels, "null": null_detection.funnels},
        },
        "normalization_coefficients": real_pair.coefficients,
        "valid_pixel_fraction": _measurement(float(detection.valid.mean()), "fraction"),
        "reliability": reliability,
        "reliability_ratio_null_to_real_scour_area": reliability_ratio,
        "sensitivity": sensitivity,
        "radar_corroboration": radar_corroboration,
        "attribution": ATTRIBUTION,
    }
    _merge_facts(run_dir / "facts.json", {"optical": facts})
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(facts, indent=2), encoding="utf-8")
    expected_quicklooks = (
        "optical_overlay.png",
        "optical_zoom_syapru.png",
        "optical_stable_pixels.png",
    )
    try:
        quicklook_failures = _write_optical_quicklooks(
            output_dir,
            scenes["pre"]["arrays"],
            scenes["post"]["arrays"],
            detection,
            real_pair,
            profile,
            osm_path,
            streams_gdf,
            config,
        )
    except Exception as error:
        LOGGER.warning(
            "Optical quicklook generation failed: %s: %s",
            type(error).__name__,
            error,
        )
        quicklook_failures = list(expected_quicklooks)
    manifest = RunManifest.read(manifest_path)
    quicklook_paths = [output_dir / name for name in expected_quicklooks]
    all_output_paths = [
        *outputs.values(),
        stats_path,
        *(path for path in quicklook_paths if path.is_file()),
    ]
    for output in all_output_paths:
        manifest.inputs = [record for record in manifest.inputs if record.name != output.name]
        manifest.add_input(
            name=output.name,
            role="output",
            source="Sahayak optical-change analysis",
            license="Sahayak generated product",
            retrieved_utc=datetime.now(UTC),
            path=output,
            settings=active_settings,
        )
    manifest.write(manifest_path)
    return {
        "facts": facts,
        "normalization": real_pair.coefficients,
        "sensitivity": sensitivity,
        "failed_quicklooks": quicklook_failures,
        "files": all_output_paths,
    }


def _prepare(
    pre_scene: dict[str, Any],
    post_scene: dict[str, Any],
    hand: np.ndarray,
    slope: np.ndarray,
    config: OpticalConfig,
) -> OpticalPair:
    pre = pre_scene["arrays"]
    post = post_scene["arrays"]
    return prepare_pair(
        {band: pre[band] for band in BANDS},
        {band: post[band] for band in BANDS},
        pre["SCL"],
        post["SCL"],
        pre["dataMask"],
        post["dataMask"],
        hand,
        slope,
        config,
    )


def _read_s2_scene(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as source:
        descriptions = source.descriptions
        arrays: dict[str, np.ndarray] = {}
        for index, description in enumerate(descriptions, 1):
            description_text = (description or "").strip()
            description_lower = description_text.casefold()
            if description_lower.startswith("scene classification"):
                name = "SCL"
            elif description_lower.startswith("data mask"):
                name = "dataMask"
            else:
                name = description_text.split()[0] if description_text else ""
            if name:
                arrays[name] = source.read(index, out_dtype="float32")
        missing = set((*BANDS, "SCL", "dataMask")) - set(arrays)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"{path.name} is missing described bands: {names}.")
        if source.crs is None:
            raise ValueError(f"{path.name} has no CRS.")
        profile = source.profile.copy()
    return {"arrays": arrays, "profile": profile}


def _read_aligned_single_band(
    path: Path,
    profile: dict[str, Any],
    height: int,
    width: int,
) -> np.ndarray:
    from rasterio.warp import Resampling, reproject

    with rasterio.open(path) as source:
        if source.count != 1 or source.crs is None:
            raise ValueError(f"{path.name} must be a single-band raster with a CRS.")
        destination = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return destination


def _write_raster(
    path: Path,
    data: np.ndarray,
    profile: dict[str, Any],
    dtype: str,
    nodata: float | int,
) -> None:
    output_profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": profile["crs"],
        "transform": profile["transform"],
        "nodata": nodata,
        "compress": "deflate",
    }
    array = np.asarray(data).copy()
    if dtype == "float32":
        array[~np.isfinite(array)] = nodata
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(array.astype(dtype), 1)


def _write_polygons(path: Path, detection: OpticalDetection, profile: dict[str, Any]) -> None:
    records: list[dict[str, Any]] = []
    for geometry, value in shapes(
        detection.classes,
        mask=np.isin(detection.classes, (CLASS_SCOUR_DEBRIS, CLASS_NEW_WATER)),
        transform=profile["transform"],
    ):
        code = int(value)
        geom = shape(geometry)
        mask = geometry_mask(
            [geom],
            out_shape=detection.classes.shape,
            transform=profile["transform"],
            invert=True,
        ) & (detection.classes == code)
        records.append(
            {
                "class": CLASS_NAMES[code],
                "area_ha": float(
                    mask.sum() * abs(profile["transform"].a * profile["transform"].e) / 10_000
                ),
                "mean_confidence": float(np.nanmean(detection.confidence[mask])),
                "geometry": geom,
            }
        )
    frame = gpd.GeoDataFrame(
        records,
        columns=["class", "area_ha", "mean_confidence", "geometry"],
        geometry="geometry",
        crs=profile["crs"],
    )
    frame.to_file(path, driver="GPKG", layer="optical_polygons")


def _sensitivity(
    pair: OpticalPair,
    sigma_null: dict[str, float],
    hand: np.ndarray,
    slope: np.ndarray,
    streams: np.ndarray,
    config: OpticalConfig,
    pixel_size: float,
) -> list[dict[str, Any]]:
    rows = []
    combinations = [
        (max_hand, ndvi_offset, delta_offset)
        for max_hand in config.sensitivity_max_hand_m
        for ndvi_offset in config.ndvi_pre_offsets
        for delta_offset in config.delta_ndvi_offsets
    ]
    for max_hand, ndvi_offset, delta_offset in combinations:
        adjusted = replace(config, max_hand_m=max_hand)
        result = detect_optical(
            pair,
            hand,
            slope,
            streams,
            adjusted,
            pixel_size_m=pixel_size,
            sigma_null=sigma_null,
            ndvi_pre_min=config.scour_ndvi_pre_min + ndvi_offset,
            delta_ndvi_max=config.scour_delta_ndvi_max + delta_offset,
        )
        rows.append(
            {
                "max_hand_m": max_hand,
                "ndvi_pre_min": config.scour_ndvi_pre_min + ndvi_offset,
                "delta_ndvi_max": config.scour_delta_ndvi_max + delta_offset,
                "class_areas_ha": result.class_areas_ha,
            }
        )
    return rows


def _radar_corroboration(
    run_dir: Path,
    detection: OpticalDetection,
    hand: np.ndarray,
    slope: np.ndarray,
    config: OpticalConfig,
) -> dict[str, Any]:
    diagnostics_path = run_dir / "flood" / "diagnostics.json"
    vv_path = run_dir / "flood" / "change_vv_db.tif"
    vh_path = run_dir / "flood" / "change_vh_db.tif"
    if not diagnostics_path.is_file() or not vv_path.is_file() or not vh_path.is_file():
        return {"available": False, "reason": "Radar change or null-noise diagnostics unavailable."}
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    sigma = diagnostics.get("null_pair", {}).get("sigma_null_db", {})
    overall_sigma = sigma.get("overall", {})
    vv_sigma = _extract_overall_sigma(overall_sigma.get("vv"))
    vh_sigma = _extract_overall_sigma(overall_sigma.get("vh"))
    if vv_sigma is None or vh_sigma is None:
        return {"available": False, "reason": "Radar null-pair sigma values unavailable."}
    with rasterio.open(vv_path) as source:
        vv = source.read(1)
        radar_transform = source.transform
        radar_crs = source.crs
    with rasterio.open(vh_path) as source:
        vh = source.read(1)
    with rasterio.open(run_dir / "s2_pre.tif") as source:
        optical_transform = source.transform
        optical_crs = source.crs
    if (
        vv.shape != detection.classes.shape
        or vh.shape != detection.classes.shape
        or radar_transform != optical_transform
        or radar_crs != optical_crs
    ):
        return {
            "available": False,
            "reason": "Radar change rasters are not aligned to optical grid.",
        }
    radar_valid = np.isfinite(vv) & np.isfinite(vh) & (vv != -9999.0) & (vh != -9999.0)
    corridor = (
        detection.valid
        & radar_valid
        & (hand <= config.max_hand_m)
        & (slope <= config.max_slope_deg)
    )
    scour = (detection.classes == CLASS_SCOUR_DEBRIS) & radar_valid
    corroborates = ((np.abs(vv) > 2.0 * vv_sigma) | (np.abs(vh) > 2.0 * vh_sigma)) & radar_valid
    rng = np.random.default_rng(config.random_seed)
    sample_indices = _matched_corridor_sample(scour, corridor, hand, slope, rng)
    sample_size = int(sample_indices.size)
    return {
        "available": True,
        "null_sigma_vv_db": vv_sigma,
        "null_sigma_vh_db": vh_sigma,
        "scour_fraction_above_two_sigma": (
            float(corroborates[scour].mean()) if scour.any() else None
        ),
        "matched_corridor_sample_fraction_above_two_sigma": (
            float(corroborates.flat[sample_indices].mean()) if sample_size else None
        ),
        "random_seed": config.random_seed,
    }


def _matched_corridor_sample(
    scour: np.ndarray,
    corridor: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample corridor cells by the same HAND/slope strata as scour pixels."""
    hand_bins = np.array([0.0, 10.0, 20.0, 30.0, 40.0, np.inf])
    slope_bins = np.array([0.0, 10.0, 20.0, 30.0, 35.0, np.inf])
    sampled_indices: list[np.ndarray] = []
    for hand_low, hand_high in zip(hand_bins[:-1], hand_bins[1:], strict=True):
        hand_stratum = (hand >= hand_low) & (hand < hand_high)
        for slope_low, slope_high in zip(slope_bins[:-1], slope_bins[1:], strict=True):
            stratum = corridor & hand_stratum & (slope >= slope_low) & (slope < slope_high)
            wanted = int((scour & stratum).sum())
            candidates = np.flatnonzero(stratum)
            if wanted and candidates.size:
                sampled_indices.append(
                    rng.choice(candidates, wanted, replace=candidates.size < wanted)
                )
    if not sampled_indices:
        return np.array([], dtype=np.intp)
    return np.concatenate(sampled_indices)


def _extract_overall_sigma(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def reliability_flag(
    null_scour_area_ha: float,
    real_scour_area_ha: float,
) -> tuple[str, float | None]:
    """Apply the requested null-to-real scour reliability rule."""
    if real_scour_area_ha < 0 or null_scour_area_ha < 0:
        raise ValueError("Scour areas cannot be negative.")
    if real_scour_area_ha == 0:
        return "low", None
    ratio = null_scour_area_ha / real_scour_area_ha
    return ("low" if ratio > 0.25 else "adequate", ratio)


def _write_optical_quicklooks(
    directory: Path,
    pre: dict[str, np.ndarray],
    post: dict[str, np.ndarray],
    detection: OpticalDetection,
    pair: OpticalPair,
    profile: dict[str, Any],
    osm_path: Path,
    streams: gpd.GeoDataFrame,
    config: OpticalConfig,
) -> list[str]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    roads = gpd.read_file(osm_path, layer="roads")
    settlements = gpd.read_file(osm_path, layer="settlements")
    for layer in (roads, settlements):
        if layer.crs is None:
            raise ValueError("OSM annotation layer has no CRS.")
    roads = roads.to_crs(profile["crs"])
    settlements = settlements.to_crs(profile["crs"])
    height, width = detection.classes.shape
    transform = profile["transform"]
    extent = (
        transform.c,
        transform.c + width * transform.a,
        transform.f + height * transform.e,
        transform.f,
    )
    failures: list[str] = []
    truecolor = _truecolor(post, pre)
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(truecolor, extent=extent)
    colors = np.zeros((*detection.classes.shape, 4), dtype=np.float32)
    colors[~detection.valid] = (0.85, 0.85, 0.85, 0.7)
    class_color = {
        CLASS_SCOUR_DEBRIS: (1.0, 0.0, 0.0, 0.55),
        CLASS_NEW_WATER: (0.0, 0.2, 1.0, 0.55),
        CLASS_EXISTING_CHANNEL_OR_BARE: (0.3, 0.3, 0.3, 0.5),
    }
    for class_code, rgba in class_color.items():
        colors[detection.classes == class_code] = rgba
    axis.imshow(colors, extent=extent)
    _plot_annotations(axis, roads, streams, settlements)
    axis.legend(
        handles=[
            Patch(facecolor="#ff0000", alpha=0.55, label="Scour/debris"),
            Patch(facecolor="#0033ff", alpha=0.55, label="New water"),
            Patch(facecolor="#555555", alpha=0.5, label="Existing channel/bare"),
            Patch(facecolor="#eeeeee", alpha=0.7, label="Invalid/cloud/shadow"),
            Patch(color="#e5c100", label="Pre-event OSM roads"),
            Patch(color="#00a6ff", label="Stream network"),
            Patch(color="black", label="Settlements"),
        ],
        loc="upper right",
        fontsize=8,
    )
    _figure_footer(axis, profile)
    if not write_png_atomically(
        directory / "optical_overlay.png",
        lambda path: figure.savefig(path, dpi=150),
    ):
        failures.append("optical_overlay.png")
    plt.close(figure)

    settlement_x, settlement_y = _find_settlement(settlements)
    radius = int(1500 / abs(transform.a))
    row = int((transform.f - settlement_y) / abs(transform.e))
    column = int((settlement_x - transform.c) / transform.a)
    r0, r1 = max(0, row - radius), min(height, row + radius)
    c0, c1 = max(0, column - radius), min(width, column + radius)
    crop_extent = (
        transform.c + c0 * transform.a,
        transform.c + c1 * transform.a,
        transform.f + r1 * transform.e,
        transform.f + r0 * transform.e,
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(_truecolor(pre, pre)[r0:r1, c0:c1], extent=crop_extent)
    axes[0].set_title("Pre-event")
    axes[1].imshow(_truecolor(post, pre)[r0:r1, c0:c1], extent=crop_extent)
    axes[1].set_title("Post-event")
    class_map = np.ma.masked_where(
        detection.classes[r0:r1, c0:c1] == CLASS_INVALID,
        detection.classes[r0:r1, c0:c1],
    )
    axes[2].imshow(
        class_map,
        extent=crop_extent,
        cmap=ListedColormap(["#eeeeee", "#ff0000", "#0033ff", "#555555"]),
        vmin=0,
        vmax=3,
    )
    axes[2].set_title("Optical classes")
    _figure_footer(axes[0], profile)
    if not write_png_atomically(
        directory / "optical_zoom_syapru.png",
        lambda path: figure.savefig(path, dpi=150),
    ):
        failures.append("optical_zoom_syapru.png")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(pair.stable, extent=extent, cmap="gray")
    axis.set_title("Stable pixels used for radiometric normalization")
    _figure_footer(axis, profile)
    if not write_png_atomically(
        directory / "optical_stable_pixels.png",
        lambda path: figure.savefig(path, dpi=150),
    ):
        failures.append("optical_stable_pixels.png")
    plt.close(figure)
    return failures


def _truecolor(pre: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> np.ndarray:
    rgb = np.stack([pre["B04"], pre["B03"], pre["B02"]], axis=-1)
    ref_rgb = np.stack([reference["B04"], reference["B03"], reference["B02"]], axis=-1)
    clear = (
        np.isin(reference["SCL"], (4, 5, 6, 7))
        & (reference["dataMask"] > 0)
        & np.all(np.isfinite(ref_rgb), axis=-1)
        & np.all((ref_rgb >= 0.0) & (ref_rgb <= 1.5), axis=-1)
    )
    if not clear.any():
        raise ValueError("No clear pre-event reflectance is available for quicklook stretch.")
    limits = np.nanpercentile(ref_rgb[clear], [2, 98], axis=0)
    return np.clip((rgb - limits[0]) / np.maximum(limits[1] - limits[0], 1e-6), 0, 1)


def _plot_annotations(
    axis: Any,
    roads: gpd.GeoDataFrame,
    streams: gpd.GeoDataFrame,
    settlements: gpd.GeoDataFrame,
) -> None:
    if not streams.empty:
        streams.boundary.plot(ax=axis, color="#00a6ff", linewidth=0.45, zorder=4)
    if not roads.empty:
        roads.plot(ax=axis, color="#e5c100", linewidth=0.4, zorder=5)
    if not settlements.empty:
        settlements.plot(ax=axis, color="black", markersize=8, zorder=6)


def _find_settlement(settlements: gpd.GeoDataFrame) -> tuple[float, float]:
    names = settlements["name"] if "name" in settlements else []
    for position, name in enumerate(names):
        if isinstance(name, str) and (
            "स्याप्रु बेसी" in name or "syapru" in name.casefold() or "syabru" in name.casefold()
        ):
            centroid = settlements.iloc[position].geometry.centroid
            return float(centroid.x), float(centroid.y)
    raise ValueError("OSM settlements contain no Syapru Besi match.")


def _figure_footer(axis: Any, profile: dict[str, Any]) -> None:
    axis.set_xlabel(str(profile["crs"]))
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    bar_x = min(x_limits) + abs(x_limits[1] - x_limits[0]) * 0.05
    bar_y = min(y_limits) + abs(y_limits[1] - y_limits[0]) * 0.06
    axis.plot(
        [bar_x, bar_x + 500.0],
        [bar_y, bar_y],
        color="white",
        linewidth=3,
        zorder=10,
    )
    axis.text(
        bar_x + 250.0,
        bar_y,
        "500 m",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axis.text(
        0.01,
        0.015,
        ATTRIBUTION,
        transform=axis.transAxes,
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )


def _thresholds(config: OpticalConfig) -> dict[str, float | int | list[float]]:
    return {
        "scour_ndvi_pre_min": config.scour_ndvi_pre_min,
        "scour_ndvi_post_max": config.scour_ndvi_post_max,
        "scour_delta_ndvi_max": config.scour_delta_ndvi_max,
        "new_water_mndwi_post_min": config.new_water_mndwi_post_min,
        "new_water_mndwi_pre_max": config.new_water_mndwi_pre_max,
        "new_water_ndvi_post_max": config.new_water_ndvi_post_max,
        "max_hand_m": config.max_hand_m,
        "max_slope_deg": config.max_slope_deg,
        "minimum_component_area_ha": config.minimum_component_area_ha,
        "stream_proximity_m": config.stream_proximity_m,
    }


def _measurement(value: Any, unit: str) -> dict[str, Any]:
    return {"value": value, "unit": unit, "module": "sahayak.optical.run", "provenance": "optical"}


def _measurements(values: dict[str, float], unit: str) -> dict[str, dict[str, Any]]:
    return {name: _measurement(value, unit) for name, value in values.items()}


def _merge_facts(path: Path, updates: dict[str, Any]) -> None:
    existing: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Facts file {path} must contain a JSON object.")
        existing = loaded
    existing.update(updates)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
