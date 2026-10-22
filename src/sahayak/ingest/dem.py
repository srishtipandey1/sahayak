"""Copernicus DEM retrieval, common-grid reprojection, and slope derivation."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from math import ceil, floor
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.errors import RasterioIOError
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds
from scipy.ndimage import maximum_filter

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.provenance import RunManifest

DEM_BUCKET_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_SOURCE = "Copernicus DEM GLO-30 via AWS Open Data"
DEM_LICENSE = "Copernicus DEM Public licence"
DEM_NODATA = -9999.0
DEM_ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
    "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
    "the European Union and ESA; all rights reserved."
)


def copernicus_dem_tile_name(latitude: float, longitude: float) -> str:
    """Build the verified one-degree Copernicus DEM COG key for a coordinate."""
    if not (-90 <= latitude < 90 and -180 <= longitude < 180):
        raise ValueError(
            "DEM tile coordinates must satisfy -90 <= latitude < 90 and "
            "-180 <= longitude < 180."
        )
    lat_index = floor(latitude)
    lon_index = floor(longitude)
    lat_hemisphere = "N" if lat_index >= 0 else "S"
    lon_hemisphere = "E" if lon_index >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{lat_hemisphere}{abs(lat_index):02d}_00_"
        f"{lon_hemisphere}{abs(lon_index):03d}_00_DEM"
    )


def copernicus_dem_tile_url(latitude: float, longitude: float) -> str:
    """Return the public HTTPS URL for a Copernicus DEM COG tile."""
    tile_name = copernicus_dem_tile_name(latitude, longitude)
    return f"{DEM_BUCKET_URL}/{tile_name}/{tile_name}.tif"


def tile_overlaps_for_bounds(
    bounds: tuple[float, float, float, float],
) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Return COG tile names and their exact intersections with WGS84 bounds."""
    west, south, east, north = bounds
    west = max(-180.0, west)
    south = max(-90.0, south)
    east = min(180.0, east)
    north = min(90.0, north)
    if not (west < east and south < north):
        raise ValueError("DEM bounds must have positive area within the world extent.")

    first_lon = floor(west)
    last_lon = ceil(east) - 1
    first_lat = floor(south)
    last_lat = ceil(north) - 1
    tiles: list[tuple[str, tuple[float, float, float, float]]] = []
    for lat_index in range(first_lat, last_lat + 1):
        for lon_index in range(first_lon, last_lon + 1):
            tile_bounds = (
                max(west, float(lon_index)),
                max(south, float(lat_index)),
                min(east, float(lon_index + 1)),
                min(north, float(lat_index + 1)),
            )
            tiles.append(
                (
                    copernicus_dem_tile_name(lat_index + 0.5, lon_index + 0.5),
                    tile_bounds,
                )
            )
    return tiles


def _window_for_bounds(
    bounds: tuple[float, float, float, float],
    transform: Affine,
    width: int,
    height: int,
) -> Window:
    """Calculate a clipped pixel window for the requested geographic bounds."""
    requested = from_bounds(*bounds, transform=transform)
    col_start = max(0, floor(requested.col_off))
    row_start = max(0, floor(requested.row_off))
    col_stop = min(width, ceil(requested.col_off + requested.width))
    row_stop = min(height, ceil(requested.row_off + requested.height))
    return Window(col_start, row_start, max(0, col_stop - col_start), max(0, row_stop - row_start))


def _read_remote_window(
    url: str,
    bounds: tuple[float, float, float, float],
) -> bytes:
    """Read only the requested COG window through rasterio's HTTP virtual filesystem."""
    try:
        with rasterio.open(f"/vsicurl/{url}") as source:
            window = _window_for_bounds(bounds, source.transform, source.width, source.height)
            if window.width == 0 or window.height == 0:
                raise ValueError(f"Requested bounds do not overlap DEM tile {url}.")
            data = source.read(1, window=window, masked=True).filled(DEM_NODATA).astype(np.float32)
            transform = source.window_transform(window)
    except (RasterioIOError, OSError) as error:
        message = f"Could not read Copernicus DEM COG window from {url}: {error}"
        raise RuntimeError(message) from error

    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        data=data,
        transform=np.asarray(tuple(transform)[:6], dtype=np.float64),
        nodata=np.asarray([DEM_NODATA], dtype=np.float32),
    )
    return buffer.getvalue()


def _decode_cached_window(payload: bytes) -> tuple[np.ndarray, Affine, float]:
    """Decode cached DEM pixels, transform, and nodata value."""
    with np.load(BytesIO(payload)) as archive:
        data = archive["data"].astype(np.float32)
        affine = Affine(*archive["transform"].tolist())
        nodata = float(archive["nodata"][0])
    return data, affine, nodata


def _write_grid_raster(path: Path, data: np.ndarray, grid: GridSpec, nodata: float) -> None:
    """Write a single-band raster on the shared grid."""
    profile = grid.rasterio_profile(dtype="float32", nodata=nodata)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(data.astype(np.float32), 1)
        destination.update_tags(grid_resolution_m=str(grid.resolution))


def ingest_dem(
    *,
    grid: GridSpec,
    settings: Settings,
    cache: RequestCache,
    offline: bool,
    output_dir: Path,
    manifest: RunManifest,
) -> dict[str, object]:
    """Fetch requested COG windows, write a common-grid DEM and derive slope."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dem = np.full((grid.height, grid.width), DEM_NODATA, dtype=np.float32)

    for tile_name, bounds in tile_overlaps_for_bounds(grid.geographic_bounds):
        url = f"{DEM_BUCKET_URL}/{tile_name}/{tile_name}.tif"
        request_key = {"kind": "copernicus-dem-window-v1", "url": url, "bounds": bounds}
        payload = cache.get_or_fetch(
            request_key,
            lambda tile_url=url, tile_bounds=bounds: _read_remote_window(tile_url, tile_bounds),
            suffix=".npz",
            offline=offline,
        )
        window_path = cache.path_for(request_key, ".npz")
        retrieved = cache.cached_at(request_key, ".npz")
        manifest.add_input(
            name=f"{tile_name}-requested-window.npz",
            role="input",
            source=DEM_SOURCE,
            license=DEM_LICENSE,
            retrieved_utc=retrieved,
            path=window_path,
            settings=settings,
        )
        source_data, source_transform, source_nodata = _decode_cached_window(payload)
        reproject(
            source=source_data,
            destination=dem,
            src_transform=source_transform,
            src_crs="EPSG:4326",
            src_nodata=source_nodata,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=DEM_NODATA,
            resampling=Resampling.bilinear,
            init_dest_nodata=False,
        )

    valid = np.isfinite(dem) & (dem != DEM_NODATA)
    if not valid.any():
        raise RuntimeError(
            "Copernicus DEM tiles returned no valid elevation pixels for the requested grid."
        )

    dem_path = output_dir / "dem.tif"
    _write_grid_raster(dem_path, dem, grid, DEM_NODATA)
    manifest.add_input(
        name="dem.tif",
        role="input",
        source=DEM_SOURCE,
        license=DEM_LICENSE,
        retrieved_utc=datetime.now(UTC),
        path=dem_path,
        settings=settings,
    )

    elevation_for_gradient = np.where(valid, dem, np.nan)
    gradient_y, gradient_x = np.gradient(elevation_for_gradient, grid.resolution)
    slope = np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))
    invalid_neighbors = maximum_filter((~valid).astype(np.uint8), size=3, mode="nearest") > 0
    slope_valid = np.isfinite(slope) & ~invalid_neighbors
    slope = np.where(slope_valid, slope, DEM_NODATA).astype(np.float32)

    slope_path = output_dir / "slope.tif"
    _write_grid_raster(slope_path, slope, grid, DEM_NODATA)
    manifest.add_input(
        name="slope.tif",
        role="input",
        source=DEM_SOURCE,
        license=DEM_LICENSE,
        retrieved_utc=datetime.now(UTC),
        path=slope_path,
        settings=settings,
    )

    elevations = dem[valid]
    return {
        "dem_path": dem_path,
        "slope_path": slope_path,
        "shape": dem.shape,
        "minimum_elevation_m": float(elevations.min()),
        "maximum_elevation_m": float(elevations.max()),
    }
