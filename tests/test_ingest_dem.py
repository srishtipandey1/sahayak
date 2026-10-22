"""Tests for Copernicus DEM tile key generation and tile intersections."""

import pytest

from sahayak.ingest.dem import (
    copernicus_dem_tile_name,
    copernicus_dem_tile_url,
    tile_overlaps_for_bounds,
)


@pytest.mark.parametrize(
    ("latitude", "longitude", "expected"),
    [
        (28.1, 85.3, "Copernicus_DSM_COG_10_N28_00_E085_00_DEM"),
        (50.5, -123.4, "Copernicus_DSM_COG_10_N50_00_W124_00_DEM"),
        (-33.5, -46.5, "Copernicus_DSM_COG_10_S34_00_W047_00_DEM"),
        (-34.5, 149.5, "Copernicus_DSM_COG_10_S35_00_E149_00_DEM"),
    ],
)
def test_tile_name_generation(latitude: float, longitude: float, expected: str) -> None:
    assert copernicus_dem_tile_name(latitude, longitude) == expected


def test_tile_url_has_verified_bucket_and_key_layout() -> None:
    tile_name = "Copernicus_DSM_COG_10_N28_00_E085_00_DEM"

    assert copernicus_dem_tile_url(28.1, 85.3) == (
        f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"
    )


def test_bounds_mosaic_crosses_tile_edge() -> None:
    tiles = tile_overlaps_for_bounds((84.9, 27.9, 85.1, 28.1))

    assert len(tiles) == 4
    assert {name for name, _bounds in tiles} == {
        "Copernicus_DSM_COG_10_N27_00_E084_00_DEM",
        "Copernicus_DSM_COG_10_N27_00_E085_00_DEM",
        "Copernicus_DSM_COG_10_N28_00_E084_00_DEM",
        "Copernicus_DSM_COG_10_N28_00_E085_00_DEM",
    }


@pytest.mark.parametrize("latitude,longitude", [(90, 0), (0, 180), (-91, 0)])
def test_tile_name_rejects_out_of_range_coordinates(latitude: float, longitude: float) -> None:
    with pytest.raises(ValueError, match="tile coordinates"):
        copernicus_dem_tile_name(latitude, longitude)
