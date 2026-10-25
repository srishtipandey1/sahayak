"""Network-free tests for optical change detection."""

import json
from dataclasses import replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin

from sahayak.optical.config import load_optical_config
from sahayak.optical.detect import (
    CLASS_BRIGHT_RECOVERED,
    CLASS_EXISTING_CHANNEL_OR_BARE,
    CLASS_INVALID,
    CLASS_NEW_WATER,
    CLASS_SCOUR_DEBRIS,
    detect_optical,
    false_alarm_accounting,
)
from sahayak.optical.diagnose import (
    _component_statistics,
    _merge_diagnosis_facts,
    _scl_histogram,
)
from sahayak.optical.preprocess import (
    prepare_pair,
    robust_sigma,
)
from sahayak.optical.run import (
    _control_quality_passed,
    _footprint_areas,
    _read_s2_scene,
    _write_polygons,
    reliability_flag,
)

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


def test_optical_diagnosis_reports_scl_fractions_and_component_shape() -> None:
    histogram = _scl_histogram(
        np.array([[4, 11], [11, 11]], dtype=np.float32),
        np.ones((2, 2), dtype=bool),
    )
    components = _component_statistics(
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 1]], dtype=bool),
        pixel_area_ha=0.01,
    )

    assert histogram["class_counts"]["11"] == 3
    assert histogram["class_fraction"]["11"] == 0.75
    assert components["component_count"] == 2
    assert components["components"][0]["area_ha"] == 0.03
    assert components["components"][0]["elongation"] >= 1.0


def test_optical_diagnosis_merges_into_facts_without_losing_other_sections(tmp_path) -> None:
    path = tmp_path / "facts.json"
    path.write_text(
        '{"existing": {"count": 3}, "optical": {"old_measure": 2}}',
        encoding="utf-8",
    )

    _merge_diagnosis_facts(path, {"histogram": {"11": 4}})
    facts = json.loads(path.read_text(encoding="utf-8"))

    assert facts["existing"]["count"] == 3
    assert facts["optical"]["old_measure"] == 2
    assert facts["optical"]["invalid_diagnosis"]["histogram"]["11"] == 4


def test_null_pair_false_alarm_counts_and_corridor_rate() -> None:
    classes = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    valid = np.ones((2, 2), dtype=bool)
    hand = np.array([[10, 10], [50, 50]], dtype=np.float32)

    result = false_alarm_accounting(classes, valid, hand, 0.01)

    assert result["class_counts"]["scour_debris"] == 1
    assert result["false_alarms_per_1000_valid_pixels"]["new_water"] == 250.0
    assert result["corridor_false_alarms_per_1000_valid_pixels"]["scour_debris"] == 500.0


def test_null_pair_counts_bright_recovery_candidates_outside_clear_valid_mask() -> None:
    classes = np.array([[CLASS_BRIGHT_RECOVERED, 0]], dtype=np.uint8)
    valid = np.array([[False, True]])
    hand = np.array([[10, 10]], dtype=np.float32)

    result = false_alarm_accounting(classes, valid, hand, 0.01)

    assert result["class_counts"]["bright_recovered"] == 1
    assert result["false_alarms_per_1000_valid_pixels"]["bright_recovered"] == 500.0


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


def _bright_recovery_detection(
    *,
    b02: float = 0.1,
    elevation_m: float = 1000.0,
    scl_code: int = 9,
    stream_column: int = 10,
    component_slice: tuple[slice, slice] = (slice(15, 20), slice(8, 13)),
):
    pre, post, hand, slope, pre_scl, post_scl, pre_mask, post_mask = _scenes()
    post = {band: values.copy() for band, values in pre.items()}
    post_scl = post_scl.copy()
    hand = hand.copy()
    shape = hand.shape
    rows, columns = component_slice
    post_scl[rows, columns] = scl_code
    post["B02"][rows, columns] = b02
    post["B04"][rows, columns] = 0.5
    post["B08"][rows, columns] = 0.05
    config = replace(
        CONFIG,
        invalid_dilation_pixels=0,
        opening_size_pixels=1,
        minimum_component_area_ha=0.1,
        bright_recovered_minimum_component_area_ha=0.25,
    )
    elevation = np.full(shape, elevation_m, dtype=np.float32)
    streams = np.zeros(shape, dtype=bool)
    streams[:, stream_column] = True
    pair = prepare_pair(
        pre,
        post,
        pre_scl,
        post_scl,
        pre_mask,
        post_mask,
        hand,
        slope,
        config,
    )
    result = detect_optical(
        pair,
        hand,
        slope,
        streams,
        config,
        pixel_size_m=10,
        pre_scl=pre_scl,
        post_scl=post_scl,
        pre_data_mask=pre_mask,
        post_data_mask=post_mask,
        post_raw_bands=post,
        elevation=elevation,
    )
    return result, config


def test_bright_scl_9_corridor_is_recovered_without_invalid_dilation() -> None:
    result, _ = _bright_recovery_detection()

    assert np.all(result.classes[15:20, 8:13] == CLASS_BRIGHT_RECOVERED)
    assert result.bright_recovery["candidate_count_before_b02"] == 25
    assert result.bright_recovery["candidate_count_after_b02"] == 25
    assert np.nanmax(result.confidence[15:20, 8:13]) <= 0.5


def test_bright_recovery_rejects_distant_snow_and_very_bright_cloud() -> None:
    distant, _ = _bright_recovery_detection(stream_column=35)
    near_stream_but_outside_30m, _ = _bright_recovery_detection(stream_column=5)
    snow, _ = _bright_recovery_detection(elevation_m=3500, scl_code=11)
    bright_cloud, _ = _bright_recovery_detection(b02=0.31)

    assert not np.any(distant.classes == CLASS_BRIGHT_RECOVERED)
    assert not np.any(near_stream_but_outside_30m.classes == CLASS_BRIGHT_RECOVERED)
    assert not np.any(snow.classes == CLASS_BRIGHT_RECOVERED)
    assert not np.any(bright_cloud.classes == CLASS_BRIGHT_RECOVERED)
    assert bright_cloud.bright_recovery["candidate_removed_by_b02_fraction"] == 1.0


def test_bright_recovery_drops_components_below_minimum_area() -> None:
    result, _ = _bright_recovery_detection(
        component_slice=(slice(15, 19), slice(8, 12))
    )

    assert not np.any(result.classes == CLASS_BRIGHT_RECOVERED)


def test_control_quality_gate_requires_both_clear_fractions() -> None:
    assert _control_quality_passed(0.7, 0.7)
    assert not _control_quality_passed(0.69, 0.9)
    assert not _control_quality_passed(0.9, None)


def test_footprint_areas_measure_class_and_union_masks() -> None:
    detection, _, _ = _detect()
    classes = np.zeros((3, 3), dtype=np.uint8)
    classes[0, 0] = CLASS_SCOUR_DEBRIS
    classes[0, 1] = CLASS_BRIGHT_RECOVERED
    classes[0, 2] = CLASS_NEW_WATER
    confidence = np.ones((3, 3), dtype=np.float32)
    detection = replace(detection, classes=classes, confidence=confidence)

    areas = _footprint_areas(detection, pixel_size=10)

    assert areas["class_1"]["value"] == 0.01
    assert areas["class_4"]["value"] == 0.01
    assert areas["class_1_4_union"]["value"] == 0.02
    assert areas["default"]["value"] == 0.03


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
