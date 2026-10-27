"""Synthetic tests for tiled Sen1Floods11 model inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rasterio import Affine

torch = pytest.importorskip("torch")

from sahayak.flood.config import load_flood_config  # noqa: E402
from sahayak.flood.preprocess import radar_valid_mask  # noqa: E402
from sahayak.ml.infer import (  # noqa: E402
    _read_model_metadata,
    apply_terrain_mask,
    build_model_features,
    false_alarm_accounting,
    overlap_summary,
    pixel_area_m2_from_transform,
    predict_tiled,
    remove_small_components,
    terrain_valid_mask,
    threshold_mask,
)


def test_tiled_prediction_matches_random_pointwise_model_across_blends() -> None:
    torch.manual_seed(19)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    features = np.random.default_rng(42).normal(size=(3, 13, 17)).astype(np.float32)

    probability = predict_tiled(
        model,
        features,
        tile_size=8,
        overlap=4,
    )
    with torch.inference_mode():
        expected = torch.sigmoid(model(torch.from_numpy(features[None])))[0, 0].numpy()

    assert probability.shape == (13, 17)
    np.testing.assert_allclose(probability, expected, rtol=1e-6, atol=1e-6)


def test_tiled_prediction_pads_smaller_than_one_tile() -> None:
    torch.manual_seed(7)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    features = np.ones((3, 3, 5), dtype=np.float32)

    probability = predict_tiled(model, features, tile_size=8, overlap=2)

    assert probability.shape == (3, 5)
    assert np.isfinite(probability).all()


def test_training_clip_normalization_and_feature_channel_order() -> None:
    vv_db = np.asarray([[-35.0, -10.0], [-4.0, 0.0]], dtype=np.float32)
    vh_db = np.asarray([[-25.0, -20.0], [-15.0, -5.0]], dtype=np.float32)
    valid = np.asarray([[True, True], [False, True]])

    features, vv_clipped, vh_clipped = build_model_features(
        vv_db,
        vh_db,
        valid,
        means_db=(-10.0, -20.0),
        standard_deviations_db=(5.0, 10.0),
        clip_db=(-30.0, 5.0),
    )

    assert vv_clipped[0, 0] == -30.0
    assert vh_clipped[0, 0] == -25.0
    assert features.shape == (3, 2, 2)
    assert features[0, 0, 1] == pytest.approx(0.0)
    assert features[1, 0, 1] == pytest.approx(0.0)
    assert features[2, 0, 1] == pytest.approx(0.0)
    assert np.all(features[:, 1, 0] == 0.0)


def test_shared_radar_validity_applies_existing_quality_rules() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_flood_config(root / "configs" / "flood.toml")
    vv = np.full((2, 3), 0.1, dtype=np.float32)
    vh = np.full((2, 3), 0.01, dtype=np.float32)
    incidence = np.full((2, 3), 40.0, dtype=np.float32)
    shadow = np.zeros((2, 3), dtype=np.float32)
    data_mask = np.ones((2, 3), dtype=np.float32)
    shadow[0, 1] = 1.0
    data_mask[0, 2] = 0.0
    incidence[1, 0] = 10.0
    vv[1, 1] = 0.0
    vv[1, 2] = -9999.0

    valid = radar_valid_mask(
        vv,
        vh,
        incidence,
        shadow,
        data_mask,
        config,
        nodata=-9999.0,
    )

    assert valid.tolist() == [[True, False, False], [False, False, False]]


def test_threshold_and_terrain_masks_keep_only_eligible_pixels() -> None:
    probability = np.asarray([[0.7, 0.6], [0.8, 0.9]], dtype=np.float32)
    valid = np.asarray([[True, True], [True, False]])
    hand = np.asarray([[20.0, 41.0], [30.0, 10.0]], dtype=np.float32)
    slope = np.asarray([[10.0, 10.0], [31.0, 2.0]], dtype=np.float32)

    binary = threshold_mask(probability, valid, 0.6)
    terrain_limited = apply_terrain_mask(binary, hand, slope)

    assert binary.tolist() == [[True, True], [True, False]]
    assert terrain_limited.tolist() == [[True, False], [False, False]]


def test_placebo_rate_denominator_uses_radar_and_terrain_eligible_pixels() -> None:
    post_radar_valid = np.asarray([[True, True, True, True, False]])
    placebo_radar_valid = np.asarray([[True, True, True, False, True]])
    hand = np.asarray([[10.0, 41.0, 20.0, 10.0, 10.0]])
    slope = np.asarray([[10.0, 10.0, 31.0, 10.0, 10.0]])
    terrain_valid = terrain_valid_mask(hand, slope)
    post_valid = post_radar_valid & terrain_valid
    placebo_valid = placebo_radar_valid & terrain_valid
    placebo_prediction = np.asarray([[True, True, True, True, True]]) & placebo_valid

    assert post_valid.tolist() == [[True, False, False, True, False]]
    assert placebo_valid.tolist() == [[True, False, False, False, True]]
    assert false_alarm_accounting(
        placebo_prediction,
        placebo_valid,
        pixel_area_m2=100.0,
    ) == {
        "detection_pixels": 2,
        "valid_pixels": 2,
        "area_ha": 0.02,
        "false_alarms_per_1000_valid_pixels": 1000.0,
    }


def test_small_components_are_dropped_at_physical_area_threshold() -> None:
    prediction = np.zeros((12, 12), dtype=bool)
    prediction[0:2, 0:2] = True
    prediction[4:9, 0:10] = True

    cleaned = remove_small_components(
        prediction,
        pixel_area_m2=100.0,
        minimum_area_ha=0.5,
    )

    assert not cleaned[0:2, 0:2].any()
    assert cleaned[4:9, 0:10].all()
    assert cleaned.sum() == 50


def test_component_cutoff_uses_affine_determinant_pixel_area() -> None:
    pixel_area_m2 = pixel_area_m2_from_transform(Affine(10, 2, 0, 1, -10, 0))
    prediction = np.zeros((13, 12), dtype=bool)
    prediction[0:7, 0:7] = True
    prediction[8:13, 0:10] = True

    cleaned = remove_small_components(prediction, pixel_area_m2=pixel_area_m2)

    assert pixel_area_m2 == pytest.approx(102.0)
    assert not cleaned[0:7, 0:7].any()
    assert cleaned[8:13, 0:10].all()
    assert int(cleaned.sum()) == 50


def test_placebo_false_alarm_rate_uses_valid_pixel_denominator() -> None:
    valid = np.asarray([[True] * 5 + [False] * 5], dtype=bool)
    detections = np.asarray([[True, False, True, False, False, True, True, True, True, True]])

    result = false_alarm_accounting(detections, valid, pixel_area_m2=100.0)

    assert result == {
        "detection_pixels": 2,
        "valid_pixels": 5,
        "area_ha": 0.02,
        "false_alarms_per_1000_valid_pixels": 400.0,
    }


def test_overlap_reports_both_directional_percentages_on_shared_valid_pixels() -> None:
    prediction = np.asarray([[True, True, False], [False, True, False]])
    reference = np.asarray([[True, False, True], [False, True, False]])
    valid = np.asarray([[True, True, False], [False, True, False]])

    result = overlap_summary(
        prediction,
        reference,
        valid,
        pixel_area_m2=100.0,
    )

    assert result["prediction_pixels"] == 3
    assert result["reference_pixels"] == 2
    assert result["intersection_pixels"] == 2
    assert result["prediction_inside_reference_percent"] == pytest.approx(200 / 3)
    assert result["reference_inside_prediction_percent"] == 100.0


def test_inference_uses_validation_threshold_and_recorded_normalization(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    metrics_path = tmp_path / "metrics.json"
    config_path.write_text(
        json.dumps(
            {
                "dataset": "sen1floods11",
                "synthetic": False,
                "final_metrics": True,
                "normalization": {
                    "VV_mean_db": -10.0,
                    "VH_mean_db": -20.0,
                    "VV_std_db": 4.0,
                    "VH_std_db": 5.0,
                    "clip_db": [-30.0, 5.0],
                },
                "hyperparameters": {
                    "input_channels": ["VV", "VH", "VV_minus_VH"],
                },
                "training_distributions": None,
            }
        ),
        encoding="utf-8",
    )
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": "sen1floods11",
                "synthetic": False,
                "final_metrics": True,
                "validation_threshold": 0.526,
            }
        ),
        encoding="utf-8",
    )

    metadata = _read_model_metadata(config_path, metrics_path)

    assert metadata["threshold"] == 0.526
    assert metadata["means_db"] == (-10.0, -20.0)
    assert metadata["clip_db"] == (-30.0, 5.0)
    assert metadata["training_distributions"] is None
