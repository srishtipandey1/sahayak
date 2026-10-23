"""Synthetic tests for flood preprocessing, terrain, and classification."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from sahayak.config import Settings
from sahayak.flood.config import load_flood_config
from sahayak.flood.detect import (
    CLASS_DISTURBED,
    CLASS_INVALID,
    CLASS_NEW_WATER,
    CLASS_NONE,
    CLASS_PERMANENT_WATER,
    detect_flood,
)
from sahayak.flood.preprocess import RadarScene, lee_filter, preprocess_scene
from sahayak.flood.run import _fraction_or_none, _merge_facts
from sahayak.flood.terrain import (
    FloodTerrainError,
    _clip_negative_hand,
    build_hand,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FLOOD_CONFIG = load_flood_config(PROJECT_ROOT / "configs" / "flood.toml")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        sahayak_root=tmp_path,
        _env_file=None,
    )


def _scene(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    valid: np.ndarray | None = None,
) -> RadarScene:
    return RadarScene(
        vv_db.astype(np.float32),
        vh_db.astype(np.float32),
        np.ones(vv_db.shape, dtype=bool) if valid is None else valid,
        ("VV", "VH", "Local incidence angle", "Shadow mask", "Data mask"),
    )


def _write_sentinel_raster(path: Path) -> None:
    shape = (24, 24)
    vv = np.full(shape, 0.1, dtype=np.float32)
    vh = np.full(shape, 0.01, dtype=np.float32)
    angle = np.full(shape, 40.0, dtype=np.float32)
    shadow = np.zeros(shape, dtype=np.float32)
    data_mask = np.ones(shape, dtype=np.float32)
    vv[5, 5] = 100_000.0
    shadow[6, 6] = 1
    data_mask[7, 7] = 0
    angle[8, 8] = 80
    bands = [vh, data_mask, vv, shadow, angle]
    descriptions = [
        "VH gamma0 terrain backscatter (linear power)",
        "Data mask",
        "VV gamma0 terrain backscatter (linear power)",
        "Shadow mask",
        "Local incidence angle (degrees)",
    ]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=len(bands),
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0, shape[0] * 10, 10, 10),
        nodata=-9999,
    ) as destination:
        for index, (band, description) in enumerate(zip(bands, descriptions, strict=True), start=1):
            destination.write(band, index)
            destination.set_band_description(index, description)


def test_preprocess_uses_descriptions_and_masks_artifacts(
    tmp_path: Path,
    settings: Settings,
) -> None:
    path = tmp_path / "s1.tif"
    _write_sentinel_raster(path)

    scene = preprocess_scene(path, FLOOD_CONFIG, settings)

    assert scene.valid_mask[2, 2]
    assert scene.vv_db[2, 2] == pytest.approx(-10.0, abs=0.5)
    assert not scene.valid_mask[5, 5]
    assert not scene.valid_mask[6, 6]
    assert not scene.valid_mask[7, 7]
    assert not scene.valid_mask[8, 8]
    assert np.isnan(scene.vv_db[5, 5])


def test_lee_filter_reduces_noise_preserves_edge_and_excludes_invalid() -> None:
    rng = np.random.default_rng(37)
    image = np.ones((60, 60), dtype=np.float32)
    image[:, 30:] = 4.0
    image *= rng.lognormal(mean=0, sigma=0.35, size=image.shape).astype(np.float32)
    valid = np.ones(image.shape, dtype=bool)
    image[30, 29] = 1e8
    valid[30, 29] = False

    filtered = lee_filter(image, valid, 5)
    image[30, 29] = 0.5
    reference = lee_filter(image, valid, 5)

    assert np.isnan(filtered[30, 29])
    assert np.nanvar(filtered[5:25, 5:25]) < np.var(image[5:25, 5:25])
    left_mean = np.nanmedian(filtered[:, 10:25])
    right_mean = np.nanmedian(filtered[:, 35:50])
    assert (right_mean / left_mean) == pytest.approx(4.0, rel=0.15)
    assert filtered[30, 28] == pytest.approx(reference[30, 28])


def test_multitemporal_flood_and_disturbed_surface_rules() -> None:
    rows, columns = 80, 80
    baseline_vv = np.full((rows, columns), -10.0, dtype=np.float32)
    baseline_vh = np.full((rows, columns), -10.0, dtype=np.float32)
    baseline_vv[:, 38:40] = -22
    baseline_vh[:, 38:40] = -22
    post_vv = baseline_vv.copy()
    post_vh = baseline_vh.copy()
    post_vv[20:60, 34:38] = -22
    post_vh[20:60, 34:38] = -25
    post_vv[10:20, 10:20] = -5
    post_vv[50:60, 10:20] = -5
    hand = np.full((rows, columns), 10.0, dtype=np.float32)
    hand[50:60, 10:20] = 60.0
    slope = np.full((rows, columns), 10.0, dtype=np.float32)
    stream = np.zeros((rows, columns), dtype=bool)
    stream[:, 39] = True
    pre = [_scene(baseline_vv, baseline_vh) for _ in range(3)]
    post = _scene(post_vv, post_vh)

    result = detect_flood(
        pre,
        post,
        hand,
        slope,
        stream,
        FLOOD_CONFIG,
        max_hand_m=40,
        pixel_size_m=10,
    )

    assert result.vv_water_threshold_db == pytest.approx(-21.9668, abs=0.01)
    assert np.all(result.classes[25:55, 34:38] == CLASS_NEW_WATER)
    assert np.all(result.classes[:, 38:40] == CLASS_PERMANENT_WATER)
    assert np.all(result.classes[10:20, 10:20] == CLASS_DISTURBED)
    assert np.all(result.classes[50:60, 10:20] == CLASS_NONE)
    expected_new_water = np.zeros(result.classes.shape, dtype=bool)
    expected_new_water[20:60, 34:38] = True
    assert not np.any((result.classes == CLASS_NEW_WATER) & ~expected_new_water)


def test_shadow_and_invalid_pixels_are_never_classified() -> None:
    rows, columns = 60, 60
    vv = np.full((rows, columns), -10.0, dtype=np.float32)
    vh = np.full((rows, columns), -10.0, dtype=np.float32)
    post_vv = vv.copy()
    post_vh = vh.copy()
    post_vv[20:35, 20:35] = -22
    post_vh[20:35, 20:35] = -25
    valid = np.ones((rows, columns), dtype=bool)
    valid[25, 25] = False
    pre = [_scene(vv, vh) for _ in range(3)]
    post = _scene(post_vv, post_vh, valid)
    hand = np.full((rows, columns), 5.0)
    slope = np.full((rows, columns), 5.0)
    streams = np.zeros((rows, columns), dtype=bool)
    streams[25, :] = True

    result = detect_flood(
        pre,
        post,
        hand,
        slope,
        streams,
        FLOOD_CONFIG,
        max_hand_m=40,
        pixel_size_m=10,
    )

    assert result.classes[25, 25] == CLASS_INVALID
    assert result.classes[24, 25] != CLASS_INVALID


def test_minimum_area_and_stream_proximity_remove_small_objects() -> None:
    rows, columns = 80, 80
    vv = np.full((rows, columns), -10.0, dtype=np.float32)
    vh = np.full((rows, columns), -10.0, dtype=np.float32)
    post_vv = vv.copy()
    post_vh = vh.copy()
    post_vv[20:24, 20:24] = -22
    post_vh[20:24, 20:24] = -25
    post_vv[40:55, 40:55] = -22
    post_vh[40:55, 40:55] = -25
    hand = np.full((rows, columns), 5.0)
    slope = np.full((rows, columns), 5.0)
    streams = np.zeros((rows, columns), dtype=bool)
    streams[:, 39] = True

    result = detect_flood(
        [_scene(vv, vh) for _ in range(3)],
        _scene(post_vv, post_vh),
        hand,
        slope,
        streams,
        FLOOD_CONFIG,
        max_hand_m=40,
        pixel_size_m=10,
    )

    assert not np.any(result.classes[20:24, 20:24] == CLASS_NEW_WATER)
    assert np.any(result.classes[40:55, 40:55] == CLASS_NEW_WATER)
    no_near_stream = np.zeros_like(streams)
    isolated = detect_flood(
        [_scene(vv, vh) for _ in range(3)],
        _scene(post_vv, post_vh),
        hand,
        slope,
        no_near_stream,
        FLOOD_CONFIG,
        max_hand_m=40,
        pixel_size_m=10,
    )
    assert not np.any(isolated.classes == CLASS_NEW_WATER)


def test_hand_on_synthetic_v_shaped_valley(tmp_path: Path, settings: Settings) -> None:
    height, width = 80, 80
    columns = np.arange(width, dtype=np.float32)[None, :]
    rows = np.arange(height, dtype=np.float32)[:, None]
    dem = np.abs(columns - width // 2) * 2.0 + (height - rows) * 0.05
    dem_path = tmp_path / "dem.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0, height * 10, 10, 10),
        nodata=-9999,
    ) as destination:
        destination.write(dem.astype(np.float32), 1)
    config = replace(FLOOD_CONFIG, stream_area_threshold_km2=0.02)

    hand, streams, _metadata = build_hand(
        dem_path,
        tmp_path / "hand.tif",
        tmp_path / "streams.gpkg",
        config,
        settings,
    )

    assert np.any(streams)
    assert hand.shape == (height, width)
    assert np.nanmax(np.abs(hand[streams])) < 1e-4
    assert np.nanmedian(hand[:, 40]) < np.nanmedian(hand[:, 20])
    valley_row, valley_column = np.unravel_index(
        np.nanargmax(np.where(~streams, hand, np.nan)),
        hand.shape,
    )
    assert hand[valley_row, valley_column] > 0


def test_conditioned_dem_removes_synthetic_small_negative_hand(
    tmp_path: Path,
    settings: Settings,
) -> None:
    from pysheds.grid import Grid
    from pysheds.sview import Raster

    height = width = 60
    rows = np.arange(height, dtype=np.float32)[:, None]
    columns = np.arange(width, dtype=np.float32)[None, :]
    dem = 200.0 - 0.5 * rows + 0.3 * np.abs(columns - width // 2)
    dem[20, 15] = 175.0
    dem_path = tmp_path / "pit_dem.tif"
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0, height * 10, 10, 10),
        nodata=-9999,
    ) as destination:
        destination.write(dem.astype(np.float32), 1)

    grid = Grid.from_raster(str(dem_path))
    raw_dem = grid.read_raster(str(dem_path))
    conditioned_dem = grid.resolve_flats(
        grid.fill_depressions(grid.fill_pits(raw_dem))
    )
    flow_direction = grid.flowdir(conditioned_dem, routing="d8")
    accumulation = grid.accumulation(flow_direction, routing="d8")
    stream_mask = np.asarray(accumulation) >= 100
    stream_raster = Raster(
        stream_mask.astype(np.float32),
        viewfinder=conditioned_dem.viewfinder,
    )
    raw_hand = np.asarray(
        grid.compute_hand(
            flow_direction,
            raw_dem,
            stream_raster,
            nodata_out=-9999,
            routing="d8",
        )
    )
    conditioned_hand = np.asarray(
        grid.compute_hand(
            flow_direction,
            conditioned_dem,
            stream_raster,
            nodata_out=-9999,
            routing="d8",
        )
    )
    assert raw_hand[20, 15] == pytest.approx(-7.5)
    assert not np.any(conditioned_hand < 0)

    config = replace(
        FLOOD_CONFIG,
        terrain_resolution_m=10.0,
        stream_area_threshold_km2=0.01,
    )
    hand, _streams, metadata = build_hand(
        dem_path,
        tmp_path / "hand.tif",
        tmp_path / "streams.gpkg",
        config,
        settings,
    )
    assert np.nanmin(hand) >= 0
    assert metadata["hand_qc"]["working_grid"]["negative_count"] == 0


def test_hand_negative_qc_clips_at_one_percent_and_records_metrics() -> None:
    hand = np.arange(100, dtype=np.float32).reshape(10, 10)
    hand[0, 0] = -0.5
    valid = np.ones(hand.shape, dtype=bool)

    clipped, qc = _clip_negative_hand(hand, valid)

    assert clipped[0, 0] == 0
    assert qc == {
        "negative_count": 1,
        "valid_count": 100,
        "negative_fraction": 0.01,
        "minimum_negative_m": -0.5,
    }


@pytest.mark.parametrize(
    ("negative_count", "negative_value"),
    [(2, -1.0), (1, -20.01)],
)
def test_hand_negative_qc_rejects_excess_count_or_depth(
    negative_count: int,
    negative_value: float,
) -> None:
    hand = np.ones((10, 10), dtype=np.float32)
    hand.flat[:negative_count] = negative_value

    with pytest.raises(FloodTerrainError, match="HAND negative-value QC failed"):
        _clip_negative_hand(hand, np.ones(hand.shape, dtype=bool))


def test_facts_merge_preserves_other_sections(tmp_path: Path) -> None:
    path = tmp_path / "facts.json"
    path.write_text('{"access":{"settlements":3}}', encoding="utf-8")

    _merge_facts(path, {"flood": {"area_ha": 1.25}})

    import json

    facts = json.loads(path.read_text(encoding="utf-8"))
    assert facts["access"] == {"settlements": 3}
    assert facts["flood"] == {"area_ha": 1.25}


def test_fraction_is_unavailable_when_denominator_is_zero() -> None:
    assert _fraction_or_none(0, 0) is None
    assert _fraction_or_none(2, 4) == 0.5
