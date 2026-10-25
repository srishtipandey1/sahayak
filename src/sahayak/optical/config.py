"""Validated optical-analysis configuration loaded from TOML."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpticalConfig:
    """Frozen optical classification, normalization, and cleanup parameters."""

    reflectance_min: float
    reflectance_max: float
    invalid_dilation_pixels: int
    clear_scl_classes: tuple[int, ...]
    stable_hand_min_m: float
    stable_slope_max_deg: float
    change_tail_fraction: float
    scour_ndvi_pre_min: float
    scour_ndvi_post_max: float
    scour_delta_ndvi_max: float
    new_water_mndwi_post_min: float
    new_water_mndwi_pre_max: float
    new_water_ndvi_post_max: float
    max_hand_m: float
    max_slope_deg: float
    existing_bare_ndvi_pre_max: float
    existing_water_mndwi_pre_min: float
    bright_recovered_scl_classes: tuple[int, ...]
    bright_recovered_max_elevation_m: float
    bright_recovered_delta_b04_min: float
    bright_recovered_ndvi_post_max: float
    bright_recovered_b02_max: float
    opening_size_pixels: int
    minimum_component_area_ha: float
    stream_proximity_m: float
    bright_recovered_minimum_component_area_ha: float
    bright_recovered_minimum_corridor_fraction: float
    bright_recovered_proximity_m: float
    delta_ndvi_saturation: float
    invalid_distance_saturation_m: float
    z_saturation: float
    z_normalization_weight: float
    sensitivity_max_hand_m: tuple[float, ...]
    ndvi_pre_offsets: tuple[float, ...]
    delta_ndvi_offsets: tuple[float, ...]
    bright_recovered_b02_limits: tuple[float, ...]
    bright_recovered_delta_b04_limits: tuple[float, ...]
    random_seed: int


def load_optical_config(path: Path) -> OpticalConfig:
    """Load and validate the optical TOML configuration."""
    with path.open("rb") as file:
        raw = tomllib.load(file)

    pre = _section(raw, "preprocessing")
    norm = _section(raw, "normalization")
    classify = _section(raw, "classification")
    cleanup = _section(raw, "cleanup")
    recovery = _section(raw, "bright_recovery")
    confidence = _section(raw, "confidence")
    analysis = _section(raw, "analysis")
    cfg = OpticalConfig(
        reflectance_min=_number(pre, "reflectance_min"),
        reflectance_max=_number(pre, "reflectance_max"),
        invalid_dilation_pixels=_integer(pre, "invalid_dilation_pixels"),
        clear_scl_classes=_integers(pre, "clear_scl_classes"),
        stable_hand_min_m=_number(norm, "stable_hand_min_m"),
        stable_slope_max_deg=_number(norm, "stable_slope_max_deg"),
        change_tail_fraction=_number(norm, "change_tail_fraction"),
        scour_ndvi_pre_min=_number(classify, "scour_ndvi_pre_min"),
        scour_ndvi_post_max=_number(classify, "scour_ndvi_post_max"),
        scour_delta_ndvi_max=_number(classify, "scour_delta_ndvi_max"),
        new_water_mndwi_post_min=_number(classify, "new_water_mndwi_post_min"),
        new_water_mndwi_pre_max=_number(classify, "new_water_mndwi_pre_max"),
        new_water_ndvi_post_max=_number(classify, "new_water_ndvi_post_max"),
        max_hand_m=_number(classify, "max_hand_m"),
        max_slope_deg=_number(classify, "max_slope_deg"),
        existing_bare_ndvi_pre_max=_number(classify, "existing_bare_ndvi_pre_max"),
        existing_water_mndwi_pre_min=_number(classify, "existing_water_mndwi_pre_min"),
        bright_recovered_scl_classes=_integers(recovery, "scl_classes"),
        bright_recovered_max_elevation_m=_number(recovery, "max_elevation_m"),
        bright_recovered_delta_b04_min=_number(recovery, "delta_b04_min"),
        bright_recovered_ndvi_post_max=_number(recovery, "ndvi_post_max"),
        bright_recovered_b02_max=_number(recovery, "b02_max"),
        opening_size_pixels=_integer(cleanup, "opening_size_pixels"),
        minimum_component_area_ha=_number(cleanup, "minimum_component_area_ha"),
        stream_proximity_m=_number(cleanup, "stream_proximity_m"),
        bright_recovered_minimum_component_area_ha=_number(
            recovery, "minimum_component_area_ha"
        ),
        bright_recovered_minimum_corridor_fraction=_number(
            recovery, "minimum_corridor_fraction"
        ),
        bright_recovered_proximity_m=_number(recovery, "proximity_m"),
        delta_ndvi_saturation=_number(confidence, "delta_ndvi_saturation"),
        invalid_distance_saturation_m=_number(confidence, "invalid_distance_saturation_m"),
        z_saturation=_number(confidence, "z_saturation"),
        z_normalization_weight=_number(confidence, "z_normalization_weight"),
        sensitivity_max_hand_m=_numbers(analysis, "sensitivity_max_hand_m"),
        ndvi_pre_offsets=_numbers(analysis, "ndvi_pre_offsets"),
        delta_ndvi_offsets=_numbers(analysis, "delta_ndvi_offsets"),
        bright_recovered_b02_limits=_numbers(analysis, "bright_recovered_b02_limits"),
        bright_recovered_delta_b04_limits=_numbers(
            analysis, "bright_recovered_delta_b04_limits"
        ),
        random_seed=_integer(analysis, "random_seed"),
    )
    if cfg.invalid_dilation_pixels < 0 or cfg.opening_size_pixels < 1:
        raise ValueError("Optical morphology sizes must be non-negative and positive.")
    if cfg.opening_size_pixels % 2 == 0:
        raise ValueError("Optical opening size must be odd.")
    if not 0 <= cfg.change_tail_fraction < 0.5:
        raise ValueError("Optical change-tail fraction must be in [0, 0.5).")
    if not cfg.clear_scl_classes or cfg.minimum_component_area_ha < 0:
        raise ValueError("Optical class list must be non-empty and component area non-negative.")
    if cfg.reflectance_min >= cfg.reflectance_max:
        raise ValueError("Optical reflectance bounds must be ordered.")
    if not 0 <= cfg.bright_recovered_minimum_corridor_fraction <= 1:
        raise ValueError("Bright-recovery corridor fraction must be in [0, 1].")
    return cfg


def _section(raw: dict[str, object], name: str) -> dict[str, object]:
    section = raw.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Optical config is missing [{name}].")
    return section


def _number(section: dict[str, object], key: str) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Optical config value {key} must be numeric.")
    return float(value)


def _integer(section: dict[str, object], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Optical config value {key} must be an integer.")
    return value


def _numbers(section: dict[str, object], key: str) -> tuple[float, ...]:
    value = section.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Optical config value {key} must be a non-empty array.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"Optical config values in {key} must be numeric.")
    return tuple(float(item) for item in value)


def _integers(section: dict[str, object], key: str) -> tuple[int, ...]:
    value = section.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Optical config value {key} must be a non-empty integer array.")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"Optical config values in {key} must be integers.")
    return tuple(value)
