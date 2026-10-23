"""Flood-stage configuration loaded from a project TOML file."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FloodConfig:
    """Validated parameters for radar preprocessing and flood classification."""

    minimum_incidence_angle_deg: float
    maximum_incidence_angle_deg: float
    vv_ceiling_db: float
    vh_ceiling_db: float
    lee_window_size: int
    terrain_resolution_m: float
    stream_area_threshold_km2: float
    minimum_baseline_std_db: float
    vv_water_threshold_min_db: float
    vv_water_threshold_max_db: float
    vh_water_threshold_db: float
    baseline_water_margin_db: float
    new_water_delta_db: float
    new_water_max_slope_deg: float
    disturbed_change_db: float
    disturbed_z_minimum: float
    disturbed_max_slope_deg: float
    default_max_hand_m: float
    sensitivity_max_hand_m: tuple[float, ...]
    minimum_component_area_ha: float
    proximity_distance_m: float
    structuring_element_size: int
    z_saturation: float
    invalid_distance_saturation_m: float
    z_weight: float
    agreement_weight: float
    invalid_distance_weight: float
    slope_weight: float


def load_flood_config(path: Path) -> FloodConfig:
    """Load TOML settings and reject missing, malformed, or inconsistent values."""
    with path.open("rb") as file:
        raw = tomllib.load(file)

    preprocessing = _section(raw, "preprocessing")
    terrain = _section(raw, "terrain")
    change = _section(raw, "change")
    cleanup = _section(raw, "cleanup")
    confidence = _section(raw, "confidence")
    sensitivity = change.get("sensitivity_max_hand_m")
    if not isinstance(sensitivity, list) or not sensitivity:
        raise ValueError("Flood config change.sensitivity_max_hand_m must be a non-empty list.")
    weights = (
        _number(confidence, "z_weight"),
        _number(confidence, "agreement_weight"),
        _number(confidence, "invalid_distance_weight"),
        _number(confidence, "slope_weight"),
    )
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Flood confidence weights must be non-negative and sum to 1.")

    config = FloodConfig(
        minimum_incidence_angle_deg=_number(preprocessing, "minimum_incidence_angle_deg"),
        maximum_incidence_angle_deg=_number(preprocessing, "maximum_incidence_angle_deg"),
        vv_ceiling_db=_number(preprocessing, "vv_ceiling_db"),
        vh_ceiling_db=_number(preprocessing, "vh_ceiling_db"),
        lee_window_size=_integer(preprocessing, "lee_window_size"),
        terrain_resolution_m=_number(terrain, "terrain_resolution_m"),
        stream_area_threshold_km2=_number(terrain, "stream_area_threshold_km2"),
        minimum_baseline_std_db=_number(change, "minimum_baseline_std_db"),
        vv_water_threshold_min_db=_number(change, "vv_water_threshold_min_db"),
        vv_water_threshold_max_db=_number(change, "vv_water_threshold_max_db"),
        vh_water_threshold_db=_number(change, "vh_water_threshold_db"),
        baseline_water_margin_db=_number(change, "baseline_water_margin_db"),
        new_water_delta_db=_number(change, "new_water_delta_db"),
        new_water_max_slope_deg=_number(change, "new_water_max_slope_deg"),
        disturbed_change_db=_number(change, "disturbed_change_db"),
        disturbed_z_minimum=_number(change, "disturbed_z_minimum"),
        disturbed_max_slope_deg=_number(change, "disturbed_max_slope_deg"),
        default_max_hand_m=_number(change, "default_max_hand_m"),
        sensitivity_max_hand_m=tuple(
            _number_value(value, "sensitivity_max_hand_m") for value in sensitivity
        ),
        minimum_component_area_ha=_number(cleanup, "minimum_component_area_ha"),
        proximity_distance_m=_number(cleanup, "proximity_distance_m"),
        structuring_element_size=_integer(cleanup, "structuring_element_size"),
        z_saturation=_number(confidence, "z_saturation"),
        invalid_distance_saturation_m=_number(confidence, "invalid_distance_saturation_m"),
        z_weight=weights[0],
        agreement_weight=weights[1],
        invalid_distance_weight=weights[2],
        slope_weight=weights[3],
    )
    _validate_config(config)
    return config


def _section(raw: dict[str, object], name: str) -> dict[str, object]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Flood config is missing [{name}].")
    return value


def _number(section: dict[str, object], key: str) -> float:
    return _number_value(section.get(key), key)


def _number_value(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Flood config value {key} must be numeric.")
    return float(value)


def _integer(section: dict[str, object], key: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Flood config value {key} must be an integer.")
    return value


def _validate_config(config: FloodConfig) -> None:
    if config.minimum_incidence_angle_deg >= config.maximum_incidence_angle_deg:
        raise ValueError("Minimum incidence angle must be below the maximum.")
    if config.lee_window_size < 3 or config.lee_window_size % 2 == 0:
        raise ValueError("Lee filter window size must be an odd integer of at least 3.")
    if config.structuring_element_size < 1 or config.structuring_element_size % 2 == 0:
        raise ValueError("Cleanup structuring element size must be a positive odd integer.")
    if config.vv_water_threshold_min_db > config.vv_water_threshold_max_db:
        raise ValueError("Minimum VV water threshold must not exceed its maximum.")
    if any(value <= 0 for value in (
        config.stream_area_threshold_km2,
        config.terrain_resolution_m,
        config.minimum_baseline_std_db,
        config.default_max_hand_m,
        config.z_saturation,
        config.invalid_distance_saturation_m,
    )):
        raise ValueError(
            "Flood area, baseline standard deviation, HAND, and confidence limits must be positive."
        )
