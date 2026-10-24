"""Network-free tests for optical change detection."""

from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from sahayak.optical.config import load_optical_config
from sahayak.optical.detect import (
    CLASS_EXISTING_CHANNEL_OR_BARE,
    CLASS_INVALID,
    CLASS_NEW_WATER,
    CLASS_SCOUR_DEBRIS,
    detect_optical,
    false_alarm_accounting,
)
from sahayak.optical.preprocess import (
    prepare_pair,
    robust_sigma,
)
from sahayak.optical.run import _read_s2_scene, _write_polygons, reliability_flag

CONFIG = load_optical_config(Path(__file__).parents[1] / "configs" / "optical.toml")


def _scenes() -> tuple[np.ndarray, ...]:
    shape = (40, 40)
    pre = {
        band: np.full(shape, value, dtype=np.float32)
        for band, value in {
            "B02": 0.1,
            "B03": 0.2,
            "B04": 0.2,
            "B08": 0.6,
            "B11": 0.3,
            "B12": 0.2,
        }.items()
    }
    post = {band: values.copy() for band, values in pre.items()}
    hand = np.full(shape, 10.0, dtype=np.float32)
    hand[:10, :] = 200.0
    slope = np.full(shape, 5.0, dtype=np.float32)
    pre_scl = np.full(shape, 4.0, dtype=np.float32)
    post_scl = np.full(shape, 4.0, dtype=np.float32)
    pre_mask = np.ones(shape, dtype=np.float32)
    post_mask = np.ones(shape, dtype=np.float32)

    # Vegetated strip becomes bright, bare surface.
    post["B04"][15:25, 5:15] = 0.5
    post["B08"][15:25, 5:15] = 0.05
    # A new water body has negative pre-event MNDWI and positive post-event MNDWI.
    post["B02"][15:25, 20:30] = 0.0
    post["B03"][15:25, 20:30] = 0.29
    post["B11"][15:25, 20:30] = 0.1
    post["B08"][15:25, 20:30] = 0.15
    # Existing bare channel.
    pre["B08"][28:38, 5:15] = 0.22
    post["B08"][28:38, 5:15] = 0.22
    # Cloud pixel must be invalid.
    post_scl[27, 27] = 8
    return pre, post, hand, slope, pre_scl, post_scl, pre_mask, post_mask


def _detect(
    pre: dict[str, np.ndarray] | None = None,
    post: dict[str, np.ndarray] | None = None,
    hand: np.ndarray | None = None,
    slope: np.ndarray | None = None,
    pre_scl: np.ndarray | None = None,
    post_scl: np.ndarray | None = None,
    pre_mask: np.ndarray | None = None,
    post_mask: np.ndarray | None = None,
    config=None,
):
    initial = _scenes()
    pre = initial[0] if pre is None else pre
    post = initial[1] if post is None else post
    hand = initial[2] if hand is None else hand
    slope = initial[3] if slope is None else slope
    pre_scl = initial[4] if pre_scl is None else pre_scl
    post_scl = initial[5] if post_scl is None else post_scl
    pre_mask = initial[6] if pre_mask is None else pre_mask
    post_mask = initial[7] if post_mask is None else post_mask
    selected_config = (
        replace(
            CONFIG,
            invalid_dilation_pixels=0,
            opening_size_pixels=1,
            minimum_component_area_ha=0.1,
        )
        if config is None
        else config
    )
    pair = prepare_pair(
        pre, post, pre_scl, post_scl, pre_mask, post_mask, hand, slope, selected_config
    )
    streams = np.zeros(hand.shape, dtype=bool)
    streams[:, 10] = True
    return (
        detect_optical(
            pair,
            hand,
            slope,
            streams,
            selected_config,
            pixel_size_m=10,
        ),
        pair,
        selected_config,
    )


def test_optical_classifies_scour_water_bare_and_cloud() -> None:
    detection, _, _ = _detect()

    assert np.all(detection.classes[17:23, 7:13] == CLASS_SCOUR_DEBRIS)
    assert np.all(detection.classes[17:23, 22:28] == CLASS_NEW_WATER)
    assert np.all(detection.classes[30:36, 7:13] == CLASS_EXISTING_CHANNEL_OR_BARE)
    assert detection.classes[27, 27] == CLASS_INVALID


def test_optical_stable_scene_normalization_removes_global_change() -> None:
    pre, post, hand, slope, pre_scl, post_scl, pre_mask, post_mask = _scenes()
    post = {band: value.copy() for band, value in pre.items()}
    post["B08"] *= 0.8
    detection, pair, _ = _detect(pre, post, hand, slope, pre_scl, post_scl, pre_mask, post_mask)

    assert np.allclose(pair.coefficients["B08"]["scale"], 1.25, atol=1e-5)
    assert not np.any(detection.classes == CLASS_SCOUR_DEBRIS)
    assert not np.any(detection.classes == CLASS_NEW_WATER)


def test_mad_estimator_uses_robust_median_absolute_deviation() -> None:
    values = np.array([0.0, 0.0, 0.0, 1.0, 1000.0])

    assert robust_sigma(values, np.ones(5, dtype=bool)) == 0.0


def test_null_pair_false_alarm_counts_and_corridor_rate() -> None:
    classes = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    valid = np.ones((2, 2), dtype=bool)
    hand = np.array([[10, 10], [50, 50]], dtype=np.float32)

    result = false_alarm_accounting(classes, valid, hand, 0.01)

    assert result["class_counts"]["scour_debris"] == 1
    assert result["false_alarms_per_1000_valid_pixels"]["new_water"] == 250.0
    assert result["corridor_false_alarms_per_1000_valid_pixels"]["scour_debris"] == 500.0


def test_reliability_flag_uses_null_to_real_area_ratio() -> None:
    assert reliability_flag(0.25, 1.0) == ("adequate", 0.25)
    assert reliability_flag(0.26, 1.0) == ("low", 0.26)
    assert reliability_flag(0.0, 0.0) == ("low", None)


def test_component_minimum_area_and_stream_proximity_filters() -> None:
    pre, post, hand, slope, pre_scl, post_scl, pre_mask, post_mask = _scenes()
    far_stream, _, config = _detect(pre, post, hand, slope, pre_scl, post_scl, pre_mask, post_mask)
    no_stream = np.zeros(hand.shape, dtype=bool)
    pair = prepare_pair(pre, post, pre_scl, post_scl, pre_mask, post_mask, hand, slope, config)
    distant = detect_optical(pair, hand, slope, no_stream, config, pixel_size_m=10)
    assert not np.any(distant.classes == CLASS_SCOUR_DEBRIS)

    large_minimum = replace(config, minimum_component_area_ha=2.0)
    pair_large = prepare_pair(
        pre, post, pre_scl, post_scl, pre_mask, post_mask, hand, slope, large_minimum
    )
    removed = detect_optical(
        pair_large, hand, slope, np.ones(hand.shape, dtype=bool), large_minimum, pixel_size_m=10
    )
    assert not np.any(removed.classes == CLASS_SCOUR_DEBRIS)
    assert far_stream.classes.shape == hand.shape


def test_scene_reader_uses_band_descriptions_and_polygon_writer(tmp_path) -> None:
    data = np.ones((8, 4, 4), dtype=np.float32)
    descriptions = (
        "B02 blue surface reflectance",
        "B03 green surface reflectance",
        "B04 red surface reflectance",
        "B08 near-infrared surface reflectance",
        "B11 shortwave infrared 1 surface reflectance",
        "B12 shortwave infrared 2 surface reflectance",
        "Scene classification layer (unitless class code)",
        "Data mask (1 valid data, 0 no data)",
    )
    path = tmp_path / "scene.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=8,
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0, 40, 10, 10),
    ) as output:
        output.write(data)
        for index, description in enumerate(descriptions, 1):
            output.set_band_description(index, description)

    scene = _read_s2_scene(path)
    assert set(scene["arrays"]) == {"B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"}
    detection, _, _ = _detect()
    polygon_path = tmp_path / "polygons.gpkg"
    profile = {
        "crs": "EPSG:32645",
        "transform": from_origin(0, 400, 10, 10),
    }
    _write_polygons(polygon_path, detection, profile)
    polygons = gpd.read_file(polygon_path, layer="optical_polygons")
    assert set(polygons["class"]) == {"scour_debris", "new_water"}
