"""Terrain-only downstream flowpath screening from a DEM and D8 flow direction."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
from scipy.spatial import cKDTree
from shapely import transform as transform_geometry
from shapely.geometry import LineString, mapping

from sahayak.access.hazard import merge_facts
from sahayak.config import Settings
from sahayak.flood.terrain import HAND_NODATA
from sahayak.ingest.run import validate_run_id

D8_STEPS = {
    64: (-1, 0),
    128: (-1, 1),
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
}
NEAR_FEATURE_DISTANCE_M = 100.0
MAX_STREAM_SNAP_DISTANCE_M = 300.0


def trace_downstream(
    flow_direction: np.ndarray,
    valid_mask: np.ndarray,
    start: tuple[int, int],
    *,
    pixel_size_x: float,
    pixel_size_y: float,
    max_distance_m: float,
) -> list[tuple[int, int]]:
    """Follow D8 directions until grid edge, invalid data, distance limit, or a cycle."""
    if flow_direction.shape != valid_mask.shape:
        raise ValueError("Flow-direction and validity arrays must have matching shapes.")
    if not math.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("Maximum flowpath distance must be positive and finite.")
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("Flowpath pixel dimensions must be positive.")
    row, column = start
    height, width = flow_direction.shape
    if not (0 <= row < height and 0 <= column < width and valid_mask[row, column]):
        raise ValueError("Flowpath start cell is outside the valid DEM.")

    cells = [(row, column)]
    visited = {(row, column)}
    length_m = 0.0
    while True:
        step = D8_STEPS.get(int(flow_direction[row, column]))
        if step is None:
            break
        next_row, next_column = row + step[0], column + step[1]
        if not (0 <= next_row < height and 0 <= next_column < width):
            break
        if not valid_mask[next_row, next_column] or (next_row, next_column) in visited:
            break
        step_length = math.hypot(step[0] * pixel_size_y, step[1] * pixel_size_x)
        if length_m + step_length > max_distance_m:
            break
        length_m += step_length
        row, column = next_row, next_column
        cells.append((row, column))
        visited.add((row, column))
    return cells


def _nearest_stream_cell(
    stream_mask: np.ndarray,
    transform: Affine,
    x: float,
    y: float,
) -> tuple[int, int, float]:
    """Snap a projected point to the nearest stream cell within the supported distance."""
    stream_rows, stream_columns = np.nonzero(stream_mask)
    if not len(stream_rows):
        raise ValueError("Stream network does not overlap valid DEM cells.")
    stream_xy = np.column_stack(
        (
            transform.c + (stream_columns + 0.5) * transform.a,
            transform.f + (stream_rows + 0.5) * transform.e,
        )
    )
    distance_m, nearest_index = cKDTree(stream_xy).query((x, y), k=1)
    if distance_m > MAX_STREAM_SNAP_DISTANCE_M:
        raise ValueError(
            "No stream cell lies within "
            f"{MAX_STREAM_SNAP_DISTANCE_M:g} m of the requested start point "
            f"(nearest distance {distance_m:.1f} m)."
        )
    return (
        int(stream_rows[nearest_index]),
        int(stream_columns[nearest_index]),
        float(distance_m),
    )


def _write_geojson(path: Path, features: list[dict[str, Any]]) -> None:
    """Write an RFC 7946 FeatureCollection atomically."""
    payload = {"type": "FeatureCollection", "features": features}
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        suffix=".geojson",
        delete=False,
    ) as temporary:
        temporary.write(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)


def _feature(
    geometry: Any,
    properties: dict[str, Any],
    transformer: Transformer,
) -> dict[str, Any]:
    """Create a WGS84 GeoJSON feature with JSON-safe scalar properties."""
    def project_coordinates(coordinates: np.ndarray) -> np.ndarray:
        if len(coordinates) == 1:
            longitude, latitude = transformer.transform(
                float(coordinates[0, 0]),
                float(coordinates[0, 1]),
            )
            return np.asarray([[longitude, latitude]])
        longitudes, latitudes = transformer.transform(
            coordinates[:, 0],
            coordinates[:, 1],
        )
        return np.column_stack((longitudes, latitudes))

    transformed = transform_geometry(
        geometry,
        project_coordinates,
    )
    safe_properties = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in properties.items()
    }
    return {"type": "Feature", "geometry": mapping(transformed), "properties": safe_properties}


def _nearby_features(
    *,
    flowline: LineString,
    run_dir: Path,
    output_crs: Any,
    transformer: Transformer,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Collect settlements and bridges near the path and roads it crosses."""
    network_path = run_dir / "network_osm_pre_event.gpkg"
    features: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for layer in ("settlements", "bridges", "roads"):
        frame = gpd.read_file(network_path, layer=layer).to_crs(output_crs)
        if layer == "roads":
            selected = frame.loc[frame.geometry.intersects(flowline)]
            output_layer = "road_crossing"
        else:
            selected = frame.loc[
                frame.geometry.distance(flowline) <= NEAR_FEATURE_DISTANCE_M
            ]
            output_layer = "settlement" if layer == "settlements" else "bridge"
        counts[output_layer] = len(selected)
        for _, row in selected.iterrows():
            properties = {
                key: value
                for key, value in row.items()
                if key != "geometry" and not _missing(value)
            }
            properties["feature_type"] = output_layer
            features.append(_feature(row.geometry, properties, transformer))
    return features, counts


def _missing(value: Any) -> bool:
    """Recognize null and non-finite scalar attributes."""
    if value is None:
        return True
    try:
        return bool(np.asarray(value).ndim == 0 and np.isnan(value))
    except (TypeError, ValueError):
        return False


def run_flowpath(
    *,
    run_id: str,
    longitude: float,
    latitude: float,
    distance_km: float = 30.0,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compute a D8 terrain trace and nearby pre-event OSM features."""
    validate_run_id(run_id)
    if not all(math.isfinite(value) for value in (longitude, latitude, distance_km)):
        raise ValueError("Longitude, latitude, and distance must be finite.")
    if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
        raise ValueError("Longitude or latitude is outside geographic bounds.")
    if distance_km <= 0:
        raise ValueError("Distance limit must be positive.")

    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    dem_path = run_dir / "dem.tif"
    streams_path = run_dir / "flood" / "streams.gpkg"
    network_path = run_dir / "network_osm_pre_event.gpkg"
    for required in (dem_path, streams_path, network_path):
        if not required.is_file():
            raise FileNotFoundError(f"Flowpath input is missing: {required.name}")

    from pysheds.grid import Grid

    with rasterio.open(dem_path) as source:
        if source.crs is None or not source.crs.is_projected:
            raise ValueError("Flowpath requires a projected DEM.")
        source_profile = source.profile.copy()
        dem = source.read(1).astype(np.float32)
        source_nodata = source.nodata
        if source_nodata is not None:
            dem[dem == source_nodata] = HAND_NODATA
        source_transform = source.transform
        source_crs = source.crs
        source_height, source_width = dem.shape
        pixel_size_x = abs(source_transform.a)
        pixel_size_y = abs(source_transform.e)
    if not np.isclose(pixel_size_x, pixel_size_y):
        raise ValueError("Flowpath requires square DEM pixels.")
    valid_source = np.isfinite(dem) & (dem != HAND_NODATA)
    if not valid_source.any():
        raise ValueError("Flowpath DEM has no valid elevation cells.")

    streams = gpd.read_file(streams_path, layer="streams").to_crs(source_crs)
    stream_shapes = [
        (geometry, 1)
        for geometry in streams.geometry
        if geometry is not None and not geometry.is_empty
    ]
    work_resolution = 30.0
    work_width = int(math.ceil(source_width * pixel_size_x / work_resolution))
    work_height = int(math.ceil(source_height * pixel_size_y / work_resolution))
    work_transform = source_transform * Affine.scale(
        work_resolution / pixel_size_x,
        work_resolution / pixel_size_y,
    )
    dem_work = np.full((work_height, work_width), HAND_NODATA, dtype=np.float32)
    reproject(
        source=dem,
        destination=dem_work,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=HAND_NODATA,
        dst_transform=work_transform,
        dst_crs=source_crs,
        dst_nodata=HAND_NODATA,
        resampling=Resampling.average,
        init_dest_nodata=True,
    )
    valid_work = np.isfinite(dem_work) & (dem_work != HAND_NODATA)
    stream_mask = rasterize(
        stream_shapes,
        out_shape=(work_height, work_width),
        transform=work_transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    stream_mask &= valid_work
    if not stream_mask.any():
        raise ValueError("Stream network does not overlap valid DEM cells.")

    with tempfile.TemporaryDirectory(prefix="sahayak-flowpath-") as temp_dir:
        working_dem = Path(temp_dir) / "dem.tif"
        profile = source_profile.copy()
        profile.update(
            width=work_width,
            height=work_height,
            transform=work_transform,
            dtype="float32",
            count=1,
            nodata=HAND_NODATA,
            compress="deflate",
        )
        with rasterio.open(working_dem, "w", **profile) as destination:
            destination.write(dem_work, 1)
        grid = Grid.from_raster(str(working_dem))
        raw_dem = grid.read_raster(str(working_dem))
        conditioned = grid.resolve_flats(
            grid.fill_depressions(grid.fill_pits(raw_dem))
        )
        direction = np.asarray(grid.flowdir(conditioned, routing="d8"))

    source_to_grid = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
    x, y = source_to_grid.transform(longitude, latitude)
    start_row, start_column, distance_m = _nearest_stream_cell(
        stream_mask,
        work_transform,
        x,
        y,
    )
    cells = trace_downstream(
        direction,
        valid_work,
        (start_row, start_column),
        pixel_size_x=work_resolution,
        pixel_size_y=work_resolution,
        max_distance_m=distance_km * 1000,
    )
    coordinates = [
        (
            work_transform.c + (cell_column + 0.5) * work_transform.a,
            work_transform.f + (cell_row + 0.5) * work_transform.e,
        )
        for cell_row, cell_column in cells
    ]
    if len(coordinates) < 2:
        raise ValueError(
            "D8 tracing ended at the snapped stream cell; no downstream segment found."
        )
    flowline = LineString(coordinates)
    nearby, nearby_counts = _nearby_features(
        flowline=flowline,
        run_dir=run_dir,
        output_crs=source_crs,
        transformer=Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True),
    )
    properties = {
        "feature_type": "flowpath",
        "start_longitude": longitude,
        "start_latitude": latitude,
        "snapped_stream_distance_m": float(distance_m),
        "trace_length_km": float(flowline.length / 1000),
        "distance_limit_km": distance_km,
        "cell_count": len(cells),
    }
    features = [
        _feature(
            flowline,
            properties,
            Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True),
        )
    ]
    features.extend(nearby)
    output_path = run_dir / "flowpath.geojson"
    _write_geojson(output_path, features)
    facts = {
        "module": "sahayak.flowpath",
        "algorithm": (
            "Copernicus DEM GLO-30 averaged to 30 m; pit/depression filling, "
            "flat resolution, D8 routing"
        ),
        "interpretation": (
            "Terrain-only screening; does not model hydraulics, discharge, or flood depth."
        ),
        "start": {
            "longitude": longitude,
            "latitude": latitude,
            "snapped_stream_distance_m": float(distance_m),
        },
        "trace_length_km": float(flowline.length / 1000),
        "distance_limit_km": distance_km,
        "cell_count": len(cells),
        "nearby_counts": nearby_counts,
        "nearby_feature_search_distance_m": NEAR_FEATURE_DISTANCE_M,
        "output": output_path.name,
    }
    merge_facts(run_dir / "facts.json", "flowpath", facts)
    return {"path": output_path, "facts": facts, "feature_count": len(features)}
