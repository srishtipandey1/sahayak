"""Tests for common UTM grid selection and bounds snapping."""

from sahayak.ingest.grid import GridSpec, grid_bounds_to_bbox, transform_bbox_to_grid_bounds


def test_northern_bbox_selects_utm_north() -> None:
    grid = GridSpec.from_bbox((85.30, 28.10, 85.42, 28.20))

    assert grid.epsg == 32645
    assert grid.crs == "EPSG:32645"
    assert grid.width > 0
    assert grid.height > 0


def test_southern_bbox_selects_utm_south() -> None:
    grid = GridSpec.from_bbox((-70.6, -33.6, -70.4, -33.4))

    assert grid.epsg == 32719


def test_grid_bounds_snap_to_ten_metre_pixels() -> None:
    grid, bounds = transform_bbox_to_grid_bounds((85.30, 28.10, 85.42, 28.20))

    assert bounds == grid.bounds
    assert all(coordinate % 10 == 0 for coordinate in bounds)
    assert grid.width == round((grid.right - grid.left) / 10)
    assert grid.height == round((grid.top - grid.bottom) / 10)
    assert grid.grid_bounds_to_bbox()[0] < grid.bbox[0]


def test_raster_profile_and_reverse_bounds() -> None:
    grid = GridSpec.from_bbox((85.30, 28.10, 85.42, 28.20))
    profile = grid.rasterio_profile(dtype="float32", nodata=-9999)
    reverse_bbox = grid_bounds_to_bbox(grid.bounds, grid.epsg)

    assert profile["width"] == grid.width
    assert profile["height"] == grid.height
    assert profile["crs"] == grid.crs
    assert profile["nodata"] == -9999
    assert reverse_bbox[0] < reverse_bbox[2]
    assert reverse_bbox[1] < reverse_bbox[3]
