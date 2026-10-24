"""HAND derivation and stream-mask export using a terrain-resolution working grid."""

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import reproject
from shapely.geometry import shape

from sahayak.config import Settings
from sahayak.flood.config import FloodConfig
from sahayak.provenance import require_input

HAND_NODATA = -9999.0


class FloodTerrainError(RuntimeError):
    """Raised when terrain processing cannot produce valid HAND values."""


def build_hand(
    dem_path: Path,
    hand_path: Path,
    streams_path: Path,
    config: FloodConfig,
    settings: Settings,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Derive HAND at configured terrain resolution and resample it to the map grid."""
    source_path = require_input(dem_path, settings)
    try:
        from pysheds.grid import Grid
        from pysheds.sview import Raster

        with rasterio.open(source_path) as source:
            output_profile = source.profile.copy()
            dem_10m = source.read(1).astype(np.float32)
            source_nodata = source.nodata
            if source_nodata is not None:
                dem_10m[dem_10m == source_nodata] = HAND_NODATA
            if source.crs is None or not source.crs.is_projected:
                raise FloodTerrainError("HAND requires a projected DEM with metric coordinates.")
            source_crs = source.crs
            source_transform = source.transform
            source_height, source_width = dem_10m.shape
            pixel_size_x = abs(source_transform.a)
            pixel_size_y = abs(source_transform.e)

        if not np.isclose(pixel_size_x, pixel_size_y, rtol=1e-6):
            raise FloodTerrainError("HAND requires square source-grid pixels.")
        valid_10m = np.isfinite(dem_10m) & (dem_10m != HAND_NODATA)
        if not valid_10m.any():
            raise FloodTerrainError(f"DEM {source_path.name} contains no valid elevation cells.")

        terrain_resolution = config.terrain_resolution_m
        work_width = int(np.ceil(source_width * pixel_size_x / terrain_resolution))
        work_height = int(np.ceil(source_height * pixel_size_y / terrain_resolution))
        work_transform = source_transform * Affine.scale(
            terrain_resolution / pixel_size_x,
            terrain_resolution / pixel_size_y,
        )
        dem_work = np.full((work_height, work_width), HAND_NODATA, dtype=np.float32)
        reproject(
            source=dem_10m,
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
        if not valid_work.any():
            raise FloodTerrainError("DEM resampling produced no valid terrain cells.")

        with TemporaryDirectory(prefix="sahayak-hand-") as temporary_dir:
            work_dem_path = Path(temporary_dir) / "dem_working.tif"
            work_profile = output_profile.copy()
            work_profile.update(
                width=work_width,
                height=work_height,
                transform=work_transform,
                dtype="float32",
                count=1,
                nodata=HAND_NODATA,
                compress="deflate",
            )
            with rasterio.open(work_dem_path, "w", **work_profile) as destination:
                destination.write(dem_work, 1)

            grid = Grid.from_raster(str(work_dem_path))
            raw_dem = grid.read_raster(str(work_dem_path))
            pit_filled = grid.fill_pits(raw_dem)
            depression_filled = grid.fill_depressions(pit_filled)
            conditioned_dem = grid.resolve_flats(depression_filled)
            flow_direction = grid.flowdir(conditioned_dem, routing="d8")
            accumulation = grid.accumulation(flow_direction, routing="d8")
            pixel_area_m2 = terrain_resolution**2
            minimum_cells = int(
                np.ceil(config.stream_area_threshold_km2 * 1_000_000 / pixel_area_m2)
            )
            stream_work = valid_work & (np.asarray(accumulation) >= minimum_cells)
            main_river_cells = int(np.ceil(5_000_000 / pixel_area_m2))
            main_river_work = valid_work & (np.asarray(accumulation) >= main_river_cells)
            if not stream_work.any():
                raise FloodTerrainError(
                    "PySheds found no stream cells at the configured upstream-area threshold "
                    f"of {config.stream_area_threshold_km2:g} km2."
                )

            stream_raster = Raster(
                stream_work.astype(np.float32),
                viewfinder=conditioned_dem.viewfinder,
            )
            hand_work_raster = grid.compute_hand(
                flow_direction,
                conditioned_dem,
                stream_raster,
                nodata_out=HAND_NODATA,
                routing="d8",
            )
            hand_work = np.asarray(hand_work_raster, dtype=np.float32)
            hand_work_valid = (
                valid_work
                & np.isfinite(hand_work)
                & (hand_work != HAND_NODATA)
            )
            if not hand_work_valid.any():
                raise FloodTerrainError("PySheds returned no valid HAND values.")
            hand_work, hand_qc = _clip_negative_hand(hand_work, hand_work_valid)

            hand_10m_raw = np.full(
                (source_height, source_width),
                HAND_NODATA,
                dtype=np.float32,
            )
            reproject(
                source=np.where(hand_work_valid, hand_work, HAND_NODATA),
                destination=hand_10m_raw,
                src_transform=work_transform,
                src_crs=source_crs,
                src_nodata=HAND_NODATA,
                dst_transform=source_transform,
                dst_crs=source_crs,
                dst_nodata=HAND_NODATA,
                resampling=Resampling.bilinear,
                init_dest_nodata=True,
            )
            stream_10m = np.zeros((source_height, source_width), dtype=np.uint8)
            reproject(
                source=stream_work.astype(np.uint8),
                destination=stream_10m,
                src_transform=work_transform,
                src_crs=source_crs,
                src_nodata=0,
                dst_transform=source_transform,
                dst_crs=source_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
            )
            main_river_10m = np.zeros((source_height, source_width), dtype=np.uint8)
            reproject(
                source=main_river_work.astype(np.uint8),
                destination=main_river_10m,
                src_transform=work_transform,
                src_crs=source_crs,
                src_nodata=0,
                dst_transform=source_transform,
                dst_crs=source_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
                init_dest_nodata=True,
            )

            stream_length_km = _stream_length_km(
                stream_work,
                np.asarray(flow_direction),
                terrain_resolution,
                terrain_resolution,
            )
            stream_count_work = int(stream_work.sum())

        hand_valid_10m = (
            valid_10m
            & np.isfinite(hand_10m_raw)
            & (hand_10m_raw != HAND_NODATA)
        )
        hand, resampled_hand_qc = _clip_negative_hand(hand_10m_raw, hand_valid_10m)
        stream_mask_10m = stream_10m.astype(bool) & valid_10m
        hand[stream_mask_10m & hand_valid_10m] = 0.0
        combined_hand_qc = {
            "working_grid": hand_qc,
            "resampled_10m_grid": resampled_hand_qc,
        }

        output_profile.update(dtype="float32", count=1, nodata=HAND_NODATA, compress="deflate")
        hand_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(hand_path, "w", **output_profile) as destination:
            destination.write(np.where(hand_valid_10m, hand, HAND_NODATA), 1)
            destination.set_band_description(1, "Height above nearest downstream drainage (m)")
        main_river_path = hand_path.parent / "main_river.tif"
        stream_profile = output_profile.copy()
        stream_profile.update(dtype="uint8", nodata=255)
        with rasterio.open(main_river_path, "w", **stream_profile) as destination:
            destination.write(
                np.where(valid_10m, main_river_10m, 255).astype(np.uint8),
                1,
            )
            destination.set_band_description(
                1,
                "Stream cells with upstream area at least 5 km2",
            )
        _write_streams(
            streams_path,
            stream_mask_10m,
            output_profile,
            config.stream_area_threshold_km2,
        )
        metadata: dict[str, Any] = {
            "stream_area_threshold_km2": config.stream_area_threshold_km2,
            "stream_area_threshold_cells": minimum_cells,
            "stream_cell_count": int(stream_mask_10m.sum()),
            "main_river_cell_count": int(main_river_10m.sum()),
            "stream_cell_count_working_grid": stream_count_work,
            "stream_length_km": stream_length_km,
            "hand_valid_pixel_fraction": float(hand_valid_10m.mean()),
            "pixel_area_m2": pixel_size_x * pixel_size_y,
            "pixel_size_m": [pixel_size_x, pixel_size_y],
            "terrain_resolution_m": terrain_resolution,
            "hand_qc": combined_hand_qc,
            "algorithm": (
                "DEM averaged to configured terrain resolution; PySheds pit fill, "
                "depression fill, flat resolution, D8 flow direction and accumulation; "
                "HAND computed with the same flat-resolved DEM used for flow direction; "
                "HAND bilinearly and streams by nearest-neighbour resampled to map grid"
            ),
        }
        return hand, stream_mask_10m, metadata
    except FloodTerrainError:
        raise
    except Exception as error:
        raise FloodTerrainError(f"PySheds {type(error).__name__}: {error}") from error


def _clip_negative_hand(
    hand: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """Validate negative HAND prevalence and clip only within configured QC limits."""
    valid = valid_mask & np.isfinite(hand) & (hand != HAND_NODATA)
    if not valid.any():
        raise FloodTerrainError("HAND contains no valid cells.")
    negative = valid & (hand < 0)
    negative_count = int(negative.sum())
    valid_count = int(valid.sum())
    negative_fraction = negative_count / valid_count
    minimum = float(hand[negative].min()) if negative_count else None
    if negative_fraction > 0.01 or (minimum is not None and minimum < -20.0):
        raise FloodTerrainError(
            "HAND negative-value QC failed: "
            f"{negative_count}/{valid_count} valid cells ({negative_fraction:.6%}) are negative; "
            f"minimum={minimum} m."
        )
    result = hand.copy()
    result[negative] = 0.0
    result[~valid] = np.nan
    return result, {
        "negative_count": negative_count,
        "valid_count": valid_count,
        "negative_fraction": negative_fraction,
        "minimum_negative_m": minimum,
    }


def _stream_length_km(
    stream_mask: np.ndarray,
    flow_direction: np.ndarray,
    pixel_size_x: float,
    pixel_size_y: float,
) -> float:
    """Sum downstream D8 link lengths whose endpoints are both stream cells."""
    direction_steps = {
        64: (-1, 0),
        128: (-1, 1),
        1: (0, 1),
        2: (1, 1),
        4: (1, 0),
        8: (1, -1),
        16: (0, -1),
        32: (-1, -1),
    }
    height, width = stream_mask.shape
    length_m = 0.0
    for code, (row_step, column_step) in direction_steps.items():
        source_rows, source_columns = np.nonzero(stream_mask & (flow_direction == code))
        destination_rows = source_rows + row_step
        destination_columns = source_columns + column_step
        in_bounds = (
            (destination_rows >= 0)
            & (destination_rows < height)
            & (destination_columns >= 0)
            & (destination_columns < width)
        )
        if in_bounds.any():
            downstream_is_stream = stream_mask[
                destination_rows[in_bounds],
                destination_columns[in_bounds],
            ]
            step_length = float(
                np.hypot(row_step * pixel_size_y, column_step * pixel_size_x)
            )
            length_m += float(downstream_is_stream.sum()) * step_length
    return length_m / 1000.0


def _write_streams(
    path: Path,
    stream_mask: np.ndarray,
    profile: dict[str, Any],
    threshold_km2: float,
) -> None:
    """Write 10 m grid stream cells as GeoPackage polygons."""
    features: Iterator[tuple[dict[str, object], int]] = shapes(
        stream_mask.astype(np.uint8),
        mask=stream_mask,
        transform=profile["transform"],
    )
    records = [
        {"geometry": shape(geometry), "threshold_km2": threshold_km2}
        for geometry, value in features
        if value == 1
    ]
    if not records:
        raise FloodTerrainError("PySheds stream mask could not be polygonized.")
    streams = gpd.GeoDataFrame(records, geometry="geometry", crs=profile["crs"])
    path.parent.mkdir(parents=True, exist_ok=True)
    streams.to_file(path, driver="GPKG", index=False)
