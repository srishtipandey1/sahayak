"""Static dashboard export and bounded derived sample bundle generation."""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image
from pyproj import Transformer

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id

SAMPLE_MAX_BYTES = 15 * 1024 * 1024
GEOJSON_MAX_BYTES = 1024 * 1024
SAMPLE_FEATURE_LIMIT = 50_000
SAMPLE_ATTRIBUTIONS = (
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
    "the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors (ODbL). Any published modified OSM data stays under ODbL.",
    "Kuro Siwo: Bountos et al., NeurIPS 2024 (Datasets and Benchmarks).",
    "Sen1Floods11, if used: Bonafilia et al., CVPR Workshops 2020.",
    "Validation only: European Union, Copernicus Emergency Management Service data (EMSR927).",
    "Educational prototype, not an operational tool.",
)


def _json_safe(value: Any) -> Any:
    """Convert numpy and pandas scalars into strict JSON values."""
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_geojson(path: Path, frame: gpd.GeoDataFrame, *, max_features: int) -> None:
    """Simplify, whitelist, and write a size-bounded GeoJSON layer."""
    if len(frame) > max_features:
        raise ValueError(f"Layer {path.stem} has too many features: {len(frame)}.")
    projected = frame.to_crs("EPSG:4326").copy()
    tolerance = 0.00005
    projected.geometry = projected.geometry.simplify(tolerance, preserve_topology=True)
    omit = {"geometry", "tags_json", "population", "osm_timestamp"}
    features: list[dict[str, Any]] = []
    for _, row in projected.iterrows():
        properties = {
            key: _json_safe(value)
            for key, value in row.items()
            if key not in omit and key != "geometry"
        }
        properties = {
            key: value
            for key, value in properties.items()
            if value is None or isinstance(value, (str, bool))
        }
        features.append(
            {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": properties,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": features},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    if path.stat().st_size >= GEOJSON_MAX_BYTES:
        raise ValueError(f"Dashboard GeoJSON layer exceeds 1 MB: {path.name}.")


def _export_layers(run_dir: Path, dashboard_dir: Path, *, max_features: int) -> list[str]:
    """Export analytical and scenario-specific mapped layers as simplified GeoJSON."""
    layers_dir = dashboard_dir / "layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    output_names: list[str] = []

    def export(name: str, frame: gpd.GeoDataFrame) -> None:
        if frame.crs is None:
            raise ValueError(f"Dashboard layer {name} is missing its CRS.")
        if frame.empty:
            return
        _write_geojson(layers_dir / f"{name}.geojson", frame, max_features=max_features)
        output_names.append(name)

    hazard_path = run_dir / "hazard.gpkg"
    hazard_frame = gpd.read_file(hazard_path, layer="hazards")
    for scenario in ("strict", "default", "lenient"):
        subset = hazard_frame.loc[hazard_frame["scenario"] == scenario].copy()
        export(f"hazard_{scenario}", subset)

    exposure_path = run_dir / "exposure.gpkg"
    for layer, name, condition in (
        ("building_exposure", "buildings_hit", "inside"),
        ("road_exposure", "roads_hit", "is_hit"),
        ("bridge_exposure", "bridges_hit", "is_hit_20m"),
    ):
        frame = gpd.read_file(exposure_path, layer=layer)
        selected = frame.loc[frame[condition].fillna(False).astype(bool)].copy()
        export(name, selected)

    access_path = run_dir / "settlement_access.gpkg"
    settlements = gpd.read_file(access_path, layer="settlements")
    for scenario in ("strict", "default", "lenient"):
        fields = [
            column
            for column in settlements.columns
            if column in {"osm_id", "name", "place", f"{scenario}_class", "geometry"}
            or column.startswith(f"{scenario}_hospital_before_")
        ]
        export(
            f"settlements_{scenario}",
            settlements.loc[:, fields].copy(),
        )
    blocked_path = run_dir / "blocked_edges.gpkg"
    export("blocked_edges", gpd.read_file(blocked_path, layer="blocked_edges"))
    streams_path = run_dir / "flood" / "streams.gpkg"
    export("streams", gpd.read_file(streams_path, layer="streams"))
    flowpath_path = run_dir / "flowpath.geojson"
    if flowpath_path.is_file():
        flowpath = gpd.read_file(flowpath_path)
        export("flowpath", flowpath)
    return output_names


def _make_georeferenced_quicklook(run_dir: Path, dashboard_dir: Path) -> dict[str, Any]:
    """Downsample post-event Sentinel RGB and store image bounds in WGS84."""
    source_path = run_dir / "s2_post.tif"
    if not source_path.is_file():
        raise FileNotFoundError("Dashboard requires s2_post.tif for its post-event image.")
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError("Post-event Sentinel image has no CRS.")
        descriptions: dict[str, int] = {}
        for index, description in enumerate(source.descriptions, start=1):
            if description:
                band_name = description.strip().split(maxsplit=1)[0]
                descriptions.setdefault(band_name, index)
        bands = [descriptions.get(name) for name in ("B04", "B03", "B02")]
        if any(index is None for index in bands):
            raise ValueError("Post-event Sentinel image lacks B04/B03/B02 RGB bands.")
        scale = max(1, math.ceil(max(source.width, source.height) / 1200))
        width = math.ceil(source.width / scale)
        height = math.ceil(source.height / scale)
        rgb = np.stack(
            [
                source.read(
                    index,
                    out_shape=(height, width),
                    resampling=rasterio.enums.Resampling.average,
                )
                for index in bands
            ],
            axis=-1,
        ).astype(np.float32)
        mask = np.all(np.isfinite(rgb), axis=-1)
        if source.nodata is not None:
            mask &= np.all(rgb != source.nodata, axis=-1)
        for band_index in range(3):
            valid = rgb[..., band_index][mask]
            if valid.size:
                low, high = np.percentile(valid, [2, 98])
                if high > low:
                    rgb[..., band_index] = np.clip(
                        (rgb[..., band_index] - low) / (high - low), 0, 1
                    )
        image = np.clip(rgb * 255, 0, 255).astype(np.uint8)
        image[~mask] = 0
        destination = dashboard_dir / "post_event_truecolor.png"
        Image.fromarray(image, mode="RGB").save(destination, optimize=True)
        bounds = source.bounds
        transformer = Transformer.from_crs(source.crs, "EPSG:4326", always_xy=True)
        west, south = transformer.transform(bounds.left, bounds.bottom)
        east, north = transformer.transform(bounds.right, bounds.top)
    return {
        "url": "post_event_truecolor.png",
        "bounds": [[south, west], [north, east]],
        "width": width,
        "height": height,
        "source": "s2_post.tif",
    }


def build_dashboard(run_id: str, settings: Settings | None = None) -> dict[str, Any]:
    """Build a static dashboard with local Leaflet assets and derived GeoJSON layers."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    required = ("facts.json", "headline.json", "manifest.json", "hazard.gpkg", "exposure.gpkg")
    for name in required:
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Dashboard input is missing: {name}")
    dashboard_dir = run_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    package_root = Path(__file__).resolve().parents[3] / "dashboard"
    for filename in ("index.html", "app.js", "style.css"):
        source = package_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"Dashboard source asset is missing: {filename}")
        shutil.copy2(source, dashboard_dir / filename)
    leaflet_dir = package_root / "node_modules" / "leaflet" / "dist"
    for name in ("leaflet.js", "leaflet.css"):
        source = leaflet_dir / name
        if not source.is_file():
            raise FileNotFoundError(
                "Local Leaflet assets missing; run npm install in the dashboard directory."
            )
        vendor = dashboard_dir / "vendor"
        vendor.mkdir(exist_ok=True)
        shutil.copy2(source, vendor / name)
    layers = _export_layers(run_dir, dashboard_dir, max_features=SAMPLE_FEATURE_LIMIT)
    image = _make_georeferenced_quicklook(run_dir, dashboard_dir)
    facts_path = run_dir / "facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    facts["dashboard"] = {
        "module": "sahayak.dashboard",
        "run_id": run_id,
        "post_event_image": image,
        "layers": layers,
        "attributions": list(SAMPLE_ATTRIBUTIONS),
    }
    facts_path.write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    metadata_path = dashboard_dir / "dashboard.json"
    metadata_path.write_text(
        json.dumps({"layers": layers}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"directory": dashboard_dir, "layers": layers, "image": image}


def make_sample(
    *,
    run_id: str,
    output: Path,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Build a small shareable bundle of derived outputs without source rasters or secrets."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    dashboard = run_dir / "dashboard"
    if not dashboard.is_dir():
        raise FileNotFoundError("Build the run dashboard before creating its sample bundle.")
    if output.exists():
        raise FileExistsError(f"Sample output already exists: {output}")
    source_files = [
        run_dir / "facts.json",
        run_dir / "headline.json",
        run_dir / "report" / "situation-en.pdf",
        run_dir / "report" / "situation-hi.pdf",
        run_dir / "report" / "situation-ne.pdf",
    ]
    for source in source_files:
        if not source.is_file():
            raise FileNotFoundError(f"Sample input is missing: {source.name}")
    output.mkdir(parents=True)
    for source in source_files:
        relative_path = (
            Path("report") / source.name if source.parent.name == "report" else Path(source.name)
        )
        destination = output / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".json":
            data = json.loads(source.read_text(encoding="utf-8"))
            destination.write_text(
                json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        else:
            shutil.copy2(source, destination)
    static_dir = output / "dashboard"
    shutil.copytree(dashboard, static_dir)
    prohibited_extensions = {".tif", ".tiff", ".gpkg", ".pt", ".zip"}
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.casefold() in prohibited_extensions:
            raise ValueError(
                f"Sample bundle contains a prohibited source or model file: {path.name}"
            )
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if size >= SAMPLE_MAX_BYTES:
        raise ValueError(f"Sample bundle exceeds 15 MB: {size} bytes.")
    readme = output / "README.txt"
    readme.write_text(
        "Sahayak derived sample bundle\n\n"
        f"Source run: {run_id}\n"
        "Contents: curated facts/headline, simplified GeoJSON dashboard layers, "
        "downscaled post-event image, and situation-report PDFs.\n"
        "Source satellite scenes, DEM rasters, input GeoPackages, model weights, and credentials "
        "are intentionally excluded.\n\n"
        + "\n".join(SAMPLE_ATTRIBUTIONS)
        + "\n",
        encoding="utf-8",
    )
    size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    if size >= SAMPLE_MAX_BYTES:
        raise ValueError(f"Sample bundle exceeds 15 MB: {size} bytes.")
    secret_pattern = re.compile(
        r"(cdse_client_secret|cdse_password|llm_api_key|ohsome_api_key|"
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
        re.IGNORECASE,
    )
    for path in output.rglob("*"):
        if path.is_file() and path.suffix.casefold() in {".json", ".js", ".html", ".txt"}:
            if secret_pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                raise ValueError(f"Sample bundle contains a credential-like field in {path.name}.")
    return {
        "path": output,
        "size_bytes": size,
        "file_count": sum(path.is_file() for path in output.rglob("*")),
    }
