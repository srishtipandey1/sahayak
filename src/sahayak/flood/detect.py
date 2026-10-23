"""Multi-temporal Sentinel-1 flood and disturbed-surface classification."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, distance_transform_edt, label
from skimage.filters import threshold_otsu

from sahayak.flood.config import FloodConfig
from sahayak.flood.preprocess import RadarScene

CLASS_NONE = 0
CLASS_NEW_WATER = 1
CLASS_DISTURBED = 2
CLASS_PERMANENT_WATER = 3
CLASS_INVALID = 255


@dataclass(frozen=True)
class FloodDetection:
    """Classification rasters and the measured VV water threshold."""

    classes: np.ndarray
    confidence: np.ndarray
    change_vv_db: np.ndarray
    change_vh_db: np.ndarray
    valid_mask: np.ndarray
    permanent_water_mask: np.ndarray
    vv_water_threshold_db: float


def detect_flood(
    pre_scenes: list[RadarScene],
    post_scene: RadarScene,
    hand: np.ndarray,
    slope: np.ndarray,
    stream_mask: np.ndarray,
    config: FloodConfig,
    *,
    max_hand_m: float,
    pixel_size_m: float,
) -> FloodDetection:
    """Classify one HAND sensitivity using valid pre-event dB samples and post-event change."""
    if not pre_scenes:
        raise ValueError("At least one pre-event Sentinel-1 scene is required.")
    shape = post_scene.vv_db.shape
    if any(
        array.shape != shape
        for array in (post_scene.vh_db, post_scene.valid_mask, hand, slope, stream_mask)
    ) or any(
        scene.vv_db.shape != shape
        or scene.vh_db.shape != shape
        or scene.valid_mask.shape != shape
        for scene in pre_scenes
    ):
        raise ValueError("All radar and terrain arrays must share the same grid shape.")
    if max_hand_m <= 0 or pixel_size_m <= 0:
        raise ValueError("Maximum HAND and pixel size must be positive.")

    baseline_vv, baseline_vh, std_vv, std_vh, baseline_valid = _baseline(pre_scenes, config)
    valid = (
        post_scene.valid_mask
        & baseline_valid
        & np.isfinite(hand)
        & np.isfinite(slope)
    )
    if not np.any(valid):
        raise ValueError(
            "No valid pixels are shared by the post-event scene, baseline, and terrain."
        )

    otsu_population = valid & (slope < config.disturbed_max_slope_deg)
    if not np.any(otsu_population):
        raise ValueError("No valid low-slope pixels are available for the VV Otsu threshold.")
    otsu_value = float(threshold_otsu(post_scene.vv_db[otsu_population]))
    water_threshold = float(
        np.clip(
            otsu_value,
            config.vv_water_threshold_min_db,
            config.vv_water_threshold_max_db,
        )
    )

    change_vv = np.full(shape, np.nan, dtype=np.float32)
    change_vh = np.full(shape, np.nan, dtype=np.float32)
    change_vv[valid] = post_scene.vv_db[valid] - baseline_vv[valid]
    change_vh[valid] = post_scene.vh_db[valid] - baseline_vh[valid]
    z_vv = np.full(shape, np.nan, dtype=np.float32)
    z_vh = np.full(shape, np.nan, dtype=np.float32)
    z_vv[valid] = change_vv[valid] / std_vv[valid]
    z_vh[valid] = change_vh[valid] / std_vh[valid]

    permanent = valid & (baseline_vv <= water_threshold)
    water_candidate = (
        valid
        & ~permanent
        & (post_scene.vv_db <= water_threshold)
        & (post_scene.vh_db <= config.vh_water_threshold_db)
        & (baseline_vv > water_threshold + config.baseline_water_margin_db)
        & (change_vv <= config.new_water_delta_db)
        & (hand <= max_hand_m)
        & (slope <= config.new_water_max_slope_deg)
    )
    disturbed_candidate = (
        valid
        & ~permanent
        & ~water_candidate
        & (
            ((change_vv >= config.disturbed_change_db) & (z_vv >= config.disturbed_z_minimum))
            | ((change_vh >= config.disturbed_change_db) & (z_vh >= config.disturbed_z_minimum))
        )
        & (hand <= max_hand_m)
        & (slope <= config.disturbed_max_slope_deg)
    )
    proximity_sources = stream_mask | permanent
    distance_to_source = distance_transform_edt(
        ~proximity_sources,
        sampling=(pixel_size_m, pixel_size_m),
    )
    water = _clean_candidates(
        water_candidate,
        distance_to_source,
        config,
        pixel_size_m,
    ) & valid
    disturbed = _clean_candidates(
        disturbed_candidate,
        distance_to_source,
        config,
        pixel_size_m,
    ) & valid
    disturbed &= ~water

    classes = np.full(shape, CLASS_INVALID, dtype=np.uint8)
    classes[valid] = CLASS_NONE
    classes[permanent] = CLASS_PERMANENT_WATER
    classes[disturbed] = CLASS_DISTURBED
    classes[water] = CLASS_NEW_WATER
    confidence = _confidence(
        valid,
        z_vv,
        z_vh,
        change_vv,
        change_vh,
        slope,
        pixel_size_m,
        config,
    )
    confidence[classes == CLASS_INVALID] = np.nan
    confidence[classes == CLASS_NONE] = 0.0
    return FloodDetection(
        classes,
        confidence,
        change_vv,
        change_vh,
        valid,
        permanent,
        water_threshold,
    )


def _baseline(
    scenes: list[RadarScene],
    config: FloodConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vv_stack = np.stack([scene.vv_db for scene in scenes]).astype(np.float32)
    vh_stack = np.stack([scene.vh_db for scene in scenes]).astype(np.float32)
    vv_valid = np.stack([scene.valid_mask & np.isfinite(scene.vv_db) for scene in scenes])
    vh_valid = np.stack([scene.valid_mask & np.isfinite(scene.vh_db) for scene in scenes])
    count_vv = vv_valid.sum(axis=0)
    count_vh = vh_valid.sum(axis=0)
    valid = (count_vv >= 2) & (count_vh >= 2)
    mean_vv = _masked_mean(vv_stack, vv_valid, count_vv)
    mean_vh = _masked_mean(vh_stack, vh_valid, count_vh)
    std_vv = _masked_std(vv_stack, vv_valid, count_vv, mean_vv)
    std_vh = _masked_std(vh_stack, vh_valid, count_vh, mean_vh)
    std_vv = np.maximum(std_vv, config.minimum_baseline_std_db)
    std_vh = np.maximum(std_vh, config.minimum_baseline_std_db)
    mean_vv[~valid] = np.nan
    mean_vh[~valid] = np.nan
    std_vv[~valid] = np.nan
    std_vh[~valid] = np.nan
    return mean_vv, mean_vh, std_vv, std_vh, valid


def _masked_mean(values: np.ndarray, mask: np.ndarray, count: np.ndarray) -> np.ndarray:
    total = np.where(mask, values, 0.0).sum(axis=0)
    return np.divide(
        total,
        count,
        out=np.full(count.shape, np.nan, dtype=np.float32),
        where=count > 0,
    )


def _masked_std(
    values: np.ndarray,
    mask: np.ndarray,
    count: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    residuals = np.where(mask, values - mean[None, :, :], 0.0)
    sum_squares = np.square(residuals).sum(axis=0)
    denominator = np.maximum(count - 1, 1)
    return np.sqrt(
        np.divide(
            sum_squares,
            denominator,
            out=np.full(count.shape, np.nan, dtype=np.float32),
            where=count >= 2,
        )
    )


def _clean_candidates(
    candidates: np.ndarray,
    distance_to_source: np.ndarray,
    config: FloodConfig,
    pixel_size_m: float,
) -> np.ndarray:
    structure = np.ones(
        (config.structuring_element_size, config.structuring_element_size),
        dtype=bool,
    )
    cleaned = binary_closing(
        binary_opening(candidates, structure=structure),
        structure=structure,
    )
    component_ids, component_count = label(cleaned, structure=np.ones((3, 3), dtype=bool))
    if component_count == 0:
        return np.zeros(candidates.shape, dtype=bool)
    pixel_area_m2 = pixel_size_m**2
    minimum_pixels = int(np.ceil(config.minimum_component_area_ha * 10_000 / pixel_area_m2))
    counts = np.bincount(component_ids.ravel(), minlength=component_count + 1)
    near_source = distance_to_source <= config.proximity_distance_m
    component_near_source = np.bincount(
        component_ids[near_source].ravel(),
        minlength=component_count + 1,
    ) > 0
    keep = (counts >= minimum_pixels) & component_near_source
    keep[0] = False
    return keep[component_ids]


def _confidence(
    valid: np.ndarray,
    z_vv: np.ndarray,
    z_vh: np.ndarray,
    change_vv: np.ndarray,
    change_vh: np.ndarray,
    slope: np.ndarray,
    pixel_size_m: float,
    config: FloodConfig,
) -> np.ndarray:
    padded = np.pad(valid, 1, constant_values=False)
    distance_from_invalid = distance_transform_edt(
        padded,
        sampling=(pixel_size_m, pixel_size_m),
    )[1:-1, 1:-1]
    z_evidence = np.clip(
        np.maximum(np.abs(z_vv), np.abs(z_vh)) / config.z_saturation,
        0.0,
        1.0,
    )
    agreement = ((change_vv * change_vh) >= 0).astype(np.float32)
    distance_quality = np.clip(
        distance_from_invalid / config.invalid_distance_saturation_m,
        0.0,
        1.0,
    )
    slope_quality = np.clip(1.0 - slope / config.disturbed_max_slope_deg, 0.0, 1.0)
    result = (
        config.z_weight * z_evidence
        + config.agreement_weight * agreement
        + config.invalid_distance_weight * distance_quality
        + config.slope_weight * slope_quality
    )
    result[~valid] = np.nan
    return result.astype(np.float32, copy=False)
