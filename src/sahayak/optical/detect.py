"""Optical change classification, component cleanup, and placebo statistics."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_opening, distance_transform_edt, label

from sahayak.optical.config import OpticalConfig
from sahayak.optical.preprocess import OpticalPair, robust_sigma, spectral_indices

CLASS_NONE = 0
CLASS_SCOUR_DEBRIS = 1
CLASS_NEW_WATER = 2
CLASS_EXISTING_CHANNEL_OR_BARE = 3
CLASS_BRIGHT_RECOVERED = 4
CLASS_INVALID = 255
CLASS_NAMES = {
    CLASS_NONE: "none",
    CLASS_SCOUR_DEBRIS: "scour_debris",
    CLASS_NEW_WATER: "new_water",
    CLASS_EXISTING_CHANNEL_OR_BARE: "existing_channel_or_bare",
    CLASS_BRIGHT_RECOVERED: "bright_recovered",
}


@dataclass(frozen=True)
class OpticalDetection:
    """Classification raster, confidence, indices, funnels, and measured areas."""

    classes: np.ndarray
    confidence: np.ndarray
    delta_ndvi: np.ndarray
    delta_brightness: np.ndarray
    valid: np.ndarray
    stable: np.ndarray
    sigma_null: dict[str, float]
    funnels: dict[str, dict[str, int]]
    class_areas_ha: dict[str, float]
    bright_recovery: dict[str, float | int]


def detect_optical(
    pair: OpticalPair,
    hand: np.ndarray,
    slope: np.ndarray,
    stream_mask: np.ndarray,
    config: OpticalConfig,
    *,
    pixel_size_m: float,
    sigma_null: dict[str, float] | None = None,
    ndvi_pre_min: float | None = None,
    delta_ndvi_max: float | None = None,
    pre_scl: np.ndarray | None = None,
    post_scl: np.ndarray | None = None,
    pre_data_mask: np.ndarray | None = None,
    post_data_mask: np.ndarray | None = None,
    post_raw_bands: dict[str, np.ndarray] | None = None,
    elevation: np.ndarray | None = None,
    delta_b04_min: float | None = None,
    b02_max: float | None = None,
) -> OpticalDetection:
    """Run the fixed optical rules, using null-pair MAD for confidence scaling."""
    pre_index = spectral_indices(pair.pre)
    post_index = spectral_indices(pair.post)
    valid = pair.valid & np.isfinite(hand) & np.isfinite(slope)
    if not (hand.shape == slope.shape == stream_mask.shape == valid.shape):
        raise ValueError("Optical terrain and stream rasters must match the scene grid.")
    for index_values in (*pre_index.values(), *post_index.values()):
        valid &= np.isfinite(index_values)
    d_ndvi = post_index["ndvi"] - pre_index["ndvi"]
    d_mndwi = post_index["mndwi"] - pre_index["mndwi"]
    d_brightness = post_index["brightness"] - pre_index["brightness"]
    measured_sigma = {
        "dNDVI": robust_sigma(d_ndvi, valid),
        "dMNDWI": robust_sigma(d_mndwi, valid),
        "dBrightness": robust_sigma(d_brightness, valid),
    }
    sigma = measured_sigma if sigma_null is None else sigma_null
    ndvi_sigma = max(sigma["dNDVI"], 1e-6)
    ndvi_min = config.scour_ndvi_pre_min if ndvi_pre_min is None else ndvi_pre_min
    delta_max = config.scour_delta_ndvi_max if delta_ndvi_max is None else delta_ndvi_max
    corridor = hand <= config.max_hand_m
    slope_ok = slope <= config.max_slope_deg
    corridor_valid = valid & corridor
    scour_eligible = corridor_valid & slope_ok
    existing = corridor_valid & (
        (pre_index["ndvi"] < config.existing_bare_ndvi_pre_max)
        | (pre_index["mndwi"] >= config.existing_water_mndwi_pre_min)
    )
    scour_raw = (
        scour_eligible
        & (pre_index["ndvi"] >= ndvi_min)
        & (post_index["ndvi"] <= config.scour_ndvi_post_max)
        & (d_ndvi <= delta_max)
        & (post_index["brightness"] >= pre_index["brightness"])
    )
    water_raw = (
        corridor_valid
        & (post_index["mndwi"] >= config.new_water_mndwi_post_min)
        & (pre_index["mndwi"] <= config.new_water_mndwi_pre_max)
        & (post_index["ndvi"] < config.new_water_ndvi_post_max)
    )
    distance_to_stream = (
        distance_transform_edt(~stream_mask, sampling=(pixel_size_m, pixel_size_m))
        if stream_mask.any()
        else np.full(stream_mask.shape, np.inf, dtype=np.float32)
    )
    close_to_stream = distance_to_stream <= config.stream_proximity_m
    pixel_area_ha = pixel_size_m * pixel_size_m / 10_000.0
    stages: dict[str, dict[str, int]] = {}
    masks: dict[int, np.ndarray] = {}
    for class_code, raw_mask in (
        (CLASS_SCOUR_DEBRIS, scour_raw),
        (CLASS_NEW_WATER, water_raw),
        (CLASS_EXISTING_CHANNEL_OR_BARE, existing),
    ):
        stages[CLASS_NAMES[class_code]] = _funnel(
            class_code,
            valid,
            hand,
            slope,
            pre_index,
            post_index,
            d_ndvi,
            config,
            ndvi_min,
            delta_max,
            close_to_stream,
            pixel_area_ha,
        )
        masks[class_code] = _clean(
            raw_mask,
            pixel_area_ha,
            close_to_stream,
            config,
            apply_proximity=True,
        )

    recovery_details: dict[str, float | int] = {
        "candidate_count_before_b02": 0,
        "candidate_count_after_b02": 0,
        "candidate_removed_by_b02_fraction": 0.0,
        "component_count": 0,
    }
    if any(
        value is not None
        for value in (
            pre_scl,
            post_scl,
            pre_data_mask,
            post_data_mask,
            post_raw_bands,
            elevation,
        )
    ):
        if any(
            value is None
            for value in (
                pre_scl,
                post_scl,
                pre_data_mask,
                post_data_mask,
                post_raw_bands,
                elevation,
            )
        ):
            raise ValueError(
                "Bright recovery requires both SCL/data masks, raw post bands, and elevation."
            )
        assert pre_scl is not None
        assert post_scl is not None
        assert pre_data_mask is not None
        assert post_data_mask is not None
        assert post_raw_bands is not None
        assert elevation is not None
        missing_bands = set(("B02", "B03", "B04", "B08", "B11", "B12")) - set(post_raw_bands)
        if missing_bands:
            raise ValueError(
                f"Bright recovery is missing raw post bands: {', '.join(sorted(missing_bands))}."
            )
        recovery_arrays = (pre_scl, post_scl, pre_data_mask, post_data_mask, elevation)
        if any(np.asarray(array).shape != valid.shape for array in recovery_arrays) or any(
            np.asarray(array).shape != valid.shape for array in post_raw_bands.values()
        ):
            raise ValueError("Bright-recovery inputs must match the optical scene grid.")
        b04_delta = pair.post["B04"] - pair.pre["B04"]
        pre_clear = (
            np.isin(pre_scl, config.clear_scl_classes)
            & (pre_data_mask > 0)
            & np.isfinite(pre_index["ndvi"])
        )
        for band in ("B02", "B03", "B04", "B08", "B11", "B12"):
            pre_clear &= (
                np.isfinite(pair.pre[band])
                & (pair.pre[band] >= config.reflectance_min)
                & (pair.pre[band] <= config.reflectance_max)
            )
        post_available = (post_data_mask > 0)
        for band in ("B02", "B03", "B04", "B08", "B11", "B12"):
            post_available &= (
                np.isfinite(post_raw_bands[band])
                & (post_raw_bands[band] >= config.reflectance_min)
                & (post_raw_bands[band] <= config.reflectance_max)
            )
        raw_recovery = (
            pre_clear
            & post_available
            & np.isin(post_scl, config.bright_recovered_scl_classes)
            & np.isfinite(elevation)
            & (elevation <= config.bright_recovered_max_elevation_m)
            & corridor
            & (distance_to_stream <= config.stream_proximity_m)
            & (b04_delta >= (
                config.bright_recovered_delta_b04_min
                if delta_b04_min is None
                else delta_b04_min
            ))
            & (post_index["ndvi"] <= config.bright_recovered_ndvi_post_max)
        )
        before_b02 = int(raw_recovery.sum())
        selected_b02_max = (
            config.bright_recovered_b02_max if b02_max is None else b02_max
        )
        raw_recovery &= post_raw_bands["B02"] <= selected_b02_max
        after_b02 = int(raw_recovery.sum())
        recovery_details.update(
            {
                "candidate_count_before_b02": before_b02,
                "candidate_count_after_b02": after_b02,
                "candidate_removed_by_b02_fraction": (
                    float((before_b02 - after_b02) / before_b02) if before_b02 else 0.0
                ),
            }
        )
        recovery_near_damage = distance_transform_edt(
            ~(
                (masks[CLASS_SCOUR_DEBRIS] if CLASS_SCOUR_DEBRIS in masks else scour_raw)
                | stream_mask
            ),
            sampling=(pixel_size_m, pixel_size_m),
        ) <= config.bright_recovered_proximity_m
        recovery_mask, component_count = _clean_bright_recovery(
            raw_recovery,
            hand,
            recovery_near_damage,
            pixel_area_ha,
            config,
        )
        masks[CLASS_BRIGHT_RECOVERED] = recovery_mask
        recovery_details["component_count"] = component_count

    classes = np.full(valid.shape, CLASS_INVALID, dtype=np.uint8)
    classes[valid] = CLASS_NONE
    classes[masks[CLASS_EXISTING_CHANNEL_OR_BARE]] = CLASS_EXISTING_CHANNEL_OR_BARE
    classes[masks[CLASS_NEW_WATER]] = CLASS_NEW_WATER
    classes[masks[CLASS_SCOUR_DEBRIS]] = CLASS_SCOUR_DEBRIS
    classes[masks.get(CLASS_BRIGHT_RECOVERED, np.zeros(valid.shape, dtype=bool))] = (
        CLASS_BRIGHT_RECOVERED
    )

    distance_to_invalid = distance_transform_edt(
        valid,
        sampling=(pixel_size_m, pixel_size_m),
    )
    z = d_ndvi / ndvi_sigma + config.z_normalization_weight * np.maximum(
        1.0 - slope / config.max_slope_deg,
        0.0,
    )
    confidence = (
        0.4 * np.minimum(np.abs(d_ndvi) / config.delta_ndvi_saturation, 1.0)
        + 0.25 * np.minimum(distance_to_invalid / config.invalid_distance_saturation_m, 1.0)
        + 0.2 * np.minimum(np.abs(z) / config.z_saturation, 1.0)
    )
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    confidence[classes == CLASS_BRIGHT_RECOVERED] = np.minimum(
        confidence[classes == CLASS_BRIGHT_RECOVERED],
        0.5,
    )
    confidence[~valid & (classes != CLASS_BRIGHT_RECOVERED)] = np.nan
    areas = {
        name: float((classes == code).sum() * pixel_area_ha) for code, name in CLASS_NAMES.items()
    }
    return OpticalDetection(
        classes,
        confidence,
        d_ndvi.astype(np.float32),
        d_brightness.astype(np.float32),
        valid,
        pair.stable,
        sigma,
        stages,
        areas,
        recovery_details,
    )


def _clean_bright_recovery(
    mask: np.ndarray,
    hand: np.ndarray,
    near_damage: np.ndarray,
    pixel_area_ha: float,
    config: OpticalConfig,
) -> tuple[np.ndarray, int]:
    labels, count = label(mask)
    keep = np.zeros(mask.shape, dtype=bool)
    kept = 0
    for component_id in range(1, count + 1):
        component = labels == component_id
        pixels = int(component.sum())
        if pixels * pixel_area_ha < config.bright_recovered_minimum_component_area_ha:
            continue
        eligible = (hand <= config.max_hand_m) & near_damage
        if float((component & eligible).sum() / pixels) < (
            config.bright_recovered_minimum_corridor_fraction
        ):
            continue
        keep |= component
        kept += 1
    return keep, kept


def _clean(
    mask: np.ndarray,
    pixel_area_ha: float,
    close_to_stream: np.ndarray,
    config: OpticalConfig,
    *,
    apply_proximity: bool = True,
) -> np.ndarray:
    opened = binary_opening(
        mask,
        structure=np.ones(
            (config.opening_size_pixels, config.opening_size_pixels),
            dtype=bool,
        ),
    )
    labels, count = label(opened)
    keep = np.zeros(mask.shape, dtype=bool)
    for component_id in range(1, count + 1):
        component = labels == component_id
        if int(component.sum()) * pixel_area_ha < config.minimum_component_area_ha:
            continue
        if apply_proximity and not np.any(component & close_to_stream):
            continue
        keep |= component
    return keep


def _funnel(
    class_code: int,
    valid: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    pre_index: dict[str, np.ndarray],
    post_index: dict[str, np.ndarray],
    delta_ndvi: np.ndarray,
    config: OpticalConfig,
    ndvi_pre_min: float,
    delta_ndvi_max: float,
    close_to_stream: np.ndarray,
    pixel_area_ha: float,
) -> dict[str, int]:
    counts = {"valid": int(valid.sum())}
    mask = valid & (hand <= config.max_hand_m)
    counts["hand"] = int(mask.sum())
    if class_code == CLASS_SCOUR_DEBRIS:
        mask &= slope <= config.max_slope_deg
    counts["slope"] = int(mask.sum())
    if class_code == CLASS_SCOUR_DEBRIS:
        mask &= pre_index["ndvi"] >= ndvi_pre_min
        counts["NDVI_pre"] = int(mask.sum())
        mask &= post_index["ndvi"] <= config.scour_ndvi_post_max
        counts["NDVI_post"] = int(mask.sum())
        mask &= delta_ndvi <= delta_ndvi_max
        counts["delta_NDVI"] = int(mask.sum())
        mask &= post_index["brightness"] >= pre_index["brightness"]
        counts["brightness"] = int(mask.sum())
    elif class_code == CLASS_NEW_WATER:
        mask &= post_index["mndwi"] >= config.new_water_mndwi_post_min
        counts["MNDWI_post"] = int(mask.sum())
        mask &= pre_index["mndwi"] <= config.new_water_mndwi_pre_max
        counts["MNDWI_pre"] = int(mask.sum())
        mask &= post_index["ndvi"] < config.new_water_ndvi_post_max
        counts["NDVI_post"] = int(mask.sum())
    else:
        mask &= (pre_index["ndvi"] < config.existing_bare_ndvi_pre_max) | (
            pre_index["mndwi"] >= config.existing_water_mndwi_pre_min
        )
        counts["existing_channel_or_bare"] = int(mask.sum())
    opened = binary_opening(
        mask,
        structure=np.ones((config.opening_size_pixels, config.opening_size_pixels), dtype=bool),
    )
    counts["cleanup"] = int(opened.sum())
    minimum_area = _clean(
        mask,
        pixel_area_ha,
        close_to_stream,
        config,
        apply_proximity=False,
    )
    counts["minimum_area"] = int(minimum_area.sum())
    proximity = _clean(
        mask,
        pixel_area_ha,
        close_to_stream,
        config,
        apply_proximity=True,
    )
    counts["proximity"] = int(proximity.sum())
    return counts


def false_alarm_accounting(
    classes: np.ndarray,
    valid: np.ndarray,
    hand: np.ndarray,
    pixel_area_ha: float,
) -> dict[str, object]:
    """Summarize placebo classes, area, and false-alarm rates."""
    count_valid = valid | (classes == CLASS_BRIGHT_RECOVERED)
    count = int(count_valid.sum())
    class_counts = {
        name: int(((classes == code) & count_valid).sum()) for code, name in CLASS_NAMES.items()
    }
    return {
        "class_counts": class_counts,
        "class_areas_ha": {name: pixels * pixel_area_ha for name, pixels in class_counts.items()},
        "false_alarms_per_1000_valid_pixels": {
            name: pixels * 1000.0 / count if count else None
            for name, pixels in class_counts.items()
            if name != "none"
        },
        "corridor_false_alarms_per_1000_valid_pixels": {
            name: (
                int(((classes == code) & count_valid & (hand <= 40.0)).sum())
                * 1000.0
                / int((count_valid & (hand <= 40.0)).sum())
                if (count_valid & (hand <= 40.0)).any()
                else None
            )
            for code, name in CLASS_NAMES.items()
            if name != "none"
        },
    }
