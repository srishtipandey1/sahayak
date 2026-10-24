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
CLASS_DARK_LOW_CONFIDENCE = 4
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
    raw_vv_water_threshold_db: float
    permanent_water_candidate_mask: np.ndarray
    dark_low_confidence_mask: np.ndarray


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

    dark_candidate = valid & (baseline_vv < water_threshold)
    permanent_candidate = dark_candidate & (hand <= 15.0) & (slope <= 10.0)
    permanent = _remove_small_components(
        permanent_candidate,
        config.minimum_component_area_ha,
        pixel_size_m,
    )
    dark_low_confidence = dark_candidate & ~permanent
    water_candidate = (
        valid
        & ~dark_candidate
        & (post_scene.vv_db < water_threshold)
        & (post_scene.vh_db < config.vh_water_threshold_db)
        & (baseline_vv > water_threshold + config.baseline_water_margin_db)
        & (change_vv <= config.new_water_delta_db)
        & (hand <= max_hand_m)
        & (slope <= config.new_water_max_slope_deg)
    )
    disturbed_candidate = (
        valid
        & ~dark_candidate
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
    water_stages = _candidate_cleanup_stages(
        water_candidate,
        distance_to_source,
        config,
        pixel_size_m,
    )
    water = water_stages["proximity"] & valid
    disturbed_candidate &= ~water
    disturbed_stages = _candidate_cleanup_stages(
        disturbed_candidate,
        distance_to_source,
        config,
        pixel_size_m,
    )
    disturbed = disturbed_stages["proximity"] & valid
    disturbed &= ~water

    classes = np.full(shape, CLASS_INVALID, dtype=np.uint8)
    classes[valid] = CLASS_NONE
    classes[permanent] = CLASS_PERMANENT_WATER
    classes[dark_low_confidence] = CLASS_DARK_LOW_CONFIDENCE
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
        otsu_value,
        dark_candidate,
        dark_low_confidence,
    )


def classification_funnels(
    pre_scenes: list[RadarScene],
    post_scene: RadarScene,
    hand: np.ndarray,
    slope: np.ndarray,
    stream_mask: np.ndarray,
    config: FloodConfig,
    *,
    max_hand_m: float,
    pixel_size_m: float,
) -> dict[str, object]:
    """Return counts for the same sequential candidate rules used by classification."""
    baseline_vv, baseline_vh, std_vv, std_vh, baseline_valid = _baseline(pre_scenes, config)
    valid = (
        post_scene.valid_mask
        & baseline_valid
        & np.isfinite(hand)
        & np.isfinite(slope)
    )
    if not valid.any():
        raise ValueError("No valid pixels are available for candidate-funnel diagnostics.")
    otsu_population = valid & (slope < config.disturbed_max_slope_deg)
    if not otsu_population.any():
        raise ValueError("No valid low-slope pixels are available for the VV Otsu threshold.")
    raw_threshold = float(threshold_otsu(post_scene.vv_db[otsu_population]))
    threshold = float(
        np.clip(raw_threshold, config.vv_water_threshold_min_db, config.vv_water_threshold_max_db)
    )
    delta_vv = post_scene.vv_db - baseline_vv
    delta_vh = post_scene.vh_db - baseline_vh
    z_vv = delta_vv / std_vv
    z_vh = delta_vh / std_vh

    water_masks: dict[str, np.ndarray] = {"valid": valid}
    water_masks["post_vv"] = water_masks["valid"] & (post_scene.vv_db < threshold)
    water_masks["post_vh"] = water_masks["post_vv"] & (
        post_scene.vh_db < config.vh_water_threshold_db
    )
    water_masks["baseline_vv"] = water_masks["post_vh"] & (
        baseline_vv > threshold + config.baseline_water_margin_db
    )
    water_masks["delta_vv"] = water_masks["baseline_vv"] & (
        delta_vv <= config.new_water_delta_db
    )
    water_masks["hand"] = water_masks["delta_vv"] & (hand <= max_hand_m)
    water_masks["slope"] = water_masks["hand"] & (slope <= config.new_water_max_slope_deg)

    dark_candidate = valid & (baseline_vv < threshold)
    permanent_candidate = dark_candidate & (hand <= 15.0) & (slope <= 10.0)
    permanent = _remove_small_components(
        permanent_candidate,
        config.minimum_component_area_ha,
        pixel_size_m,
    )
    proximity_sources = stream_mask | permanent
    distance_to_source = distance_transform_edt(
        ~proximity_sources,
        sampling=(pixel_size_m, pixel_size_m),
    )
    water_stages = _candidate_cleanup_stages(
        water_masks["slope"],
        distance_to_source,
        config,
        pixel_size_m,
    )
    water_masks.update(
        {
            "opening_closing": water_stages["opening_closing"],
            "minimum_area": water_stages["minimum_area"],
            "proximity": water_stages["proximity"],
        }
    )

    disturbed_masks: dict[str, np.ndarray] = {"valid": valid}
    disturbed_masks["backscatter_rise"] = valid & (
        (delta_vv >= config.disturbed_change_db)
        | (delta_vh >= config.disturbed_change_db)
    )
    disturbed_masks["z_score"] = disturbed_masks["backscatter_rise"] & (
        ((delta_vv >= config.disturbed_change_db) & (z_vv >= config.disturbed_z_minimum))
        | ((delta_vh >= config.disturbed_change_db) & (z_vh >= config.disturbed_z_minimum))
    )
    disturbed_masks["hand"] = disturbed_masks["z_score"] & (hand <= max_hand_m)
    disturbed_masks["slope"] = disturbed_masks["hand"] & (
        slope <= config.disturbed_max_slope_deg
    )
    disturbed_masks["not_new_water"] = disturbed_masks["slope"] & ~water_stages["proximity"]
    disturbed_stages = _candidate_cleanup_stages(
        disturbed_masks["not_new_water"],
        distance_to_source,
        config,
        pixel_size_m,
    )
    disturbed_masks.update(
        {
            "opening_closing": disturbed_stages["opening_closing"],
            "minimum_area": disturbed_stages["minimum_area"],
            "proximity": disturbed_stages["proximity"],
        }
    )

    permanent_eligible = permanent_candidate
    permanent_masks = {
        "valid": valid,
        "baseline_vv_below_threshold": dark_candidate,
        "hand": dark_candidate & (hand <= 15.0),
        "slope": permanent_eligible,
        "minimum_area": permanent,
    }
    def count_funnel(masks: dict[str, np.ndarray]) -> dict[str, int]:
        return {name: int(mask.sum()) for name, mask in masks.items()}
    permanent_values = {
        "before_restriction": _array_percentiles(hand[dark_candidate]),
        "after_restriction": _array_percentiles(hand[permanent]),
        "slope_before_restriction": _array_percentiles(slope[dark_candidate]),
        "slope_after_restriction": _array_percentiles(slope[permanent]),
        "candidate_pixel_count_before_restriction": int(dark_candidate.sum()),
        "pixel_count_after_restriction": int(permanent.sum()),
    }
    return {
        "raw_otsu_threshold_db": raw_threshold,
        "clamped_otsu_threshold_db": threshold,
        "valid_pixel_count": int(valid.sum()),
        "valid_pixel_fraction": float(valid.mean()),
        "water": count_funnel(water_masks),
        "disturbed_surface": count_funnel(disturbed_masks),
        "permanent_water": count_funnel(permanent_masks),
        "permanent_hand_and_slope_percentiles": permanent_values,
        "distributions": {
            "delta_vv_db": _array_percentiles(delta_vv[valid]),
            "delta_vh_db": _array_percentiles(delta_vh[valid]),
            "z_vv": _array_percentiles(z_vv[valid]),
            "z_vh": _array_percentiles(z_vh[valid]),
            "post_vv_db": _array_percentiles(post_scene.vv_db[valid]),
            "post_vh_db": _array_percentiles(post_scene.vh_db[valid]),
            "baseline_mean_vv_db": _array_percentiles(baseline_vv[valid]),
            "baseline_mean_vh_db": _array_percentiles(baseline_vh[valid]),
        },
        "arrays": {
            "valid": valid,
            "hand": hand,
            "slope": slope,
            "stream": stream_mask,
            "delta_vv_db": delta_vv,
            "delta_vh_db": delta_vh,
            "z_vv": z_vv,
            "z_vh": z_vh,
            "baseline_vv_db": baseline_vv,
            "baseline_vh_db": baseline_vh,
            "permanent_candidate": dark_candidate,
            "permanent_water": permanent,
        },
    }


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
    return _candidate_cleanup_stages(
        candidates,
        distance_to_source,
        config,
        pixel_size_m,
    )["proximity"]


def _candidate_cleanup_stages(
    candidates: np.ndarray,
    distance_to_source: np.ndarray,
    config: FloodConfig,
    pixel_size_m: float,
) -> dict[str, np.ndarray]:
    """Apply cleanup in separately observable opening, size, and proximity stages."""
    structure = np.ones(
        (config.structuring_element_size, config.structuring_element_size),
        dtype=bool,
    )
    cleaned = binary_closing(
        binary_opening(candidates, structure=structure),
        structure=structure,
    )
    component_ids, component_count = label(cleaned, structure=np.ones((3, 3), dtype=bool))
    minimum_area = _remove_small_components(
        cleaned,
        config.minimum_component_area_ha,
        pixel_size_m,
    )
    component_ids, component_count = label(
        minimum_area,
        structure=np.ones((3, 3), dtype=bool),
    )
    if component_count:
        near_source = distance_to_source <= config.proximity_distance_m
        component_near_source = np.bincount(
            component_ids[near_source].ravel(),
            minlength=component_count + 1,
        ) > 0
        component_near_source[0] = False
        proximity = component_near_source[component_ids]
    else:
        proximity = np.zeros(candidates.shape, dtype=bool)
    return {
        "opening_closing": cleaned,
        "minimum_area": minimum_area,
        "proximity": proximity,
    }


def _remove_small_components(
    candidates: np.ndarray,
    minimum_area_ha: float,
    pixel_size_m: float,
) -> np.ndarray:
    """Keep connected components at least the specified physical area."""
    component_ids, component_count = label(candidates, structure=np.ones((3, 3), dtype=bool))
    if component_count == 0:
        return np.zeros(candidates.shape, dtype=bool)
    minimum_pixels = int(np.ceil(minimum_area_ha * 10_000 / pixel_size_m**2))
    counts = np.bincount(component_ids.ravel(), minlength=component_count + 1)
    keep = counts >= minimum_pixels
    keep[0] = False
    return keep[component_ids]


def _array_percentiles(values: np.ndarray) -> dict[str, float | None]:
    """Summarize finite values at the fixed diagnostic percentiles."""
    finite_values = np.asarray(values)[np.isfinite(values)]
    if not finite_values.size:
        return {str(percentile): None for percentile in _DIAGNOSTIC_PERCENTILES}
    values_by_percentile = np.percentile(finite_values, _DIAGNOSTIC_PERCENTILES)
    return {
        str(percentile): float(value)
        for percentile, value in zip(
            _DIAGNOSTIC_PERCENTILES,
            values_by_percentile,
            strict=True,
        )
    }


_DIAGNOSTIC_PERCENTILES = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)


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
