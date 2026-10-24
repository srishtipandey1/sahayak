"""Candidate-funnel and null-pair diagnostics for an existing flood run."""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from sahayak.config import Settings
from sahayak.flood.config import FloodConfig, load_flood_config
from sahayak.flood.detect import (
    CLASS_DARK_LOW_CONFIDENCE,
    CLASS_DISTURBED,
    CLASS_INVALID,
    CLASS_NEW_WATER,
    CLASS_NONE,
    CLASS_PERMANENT_WATER,
    classification_funnels,
    detect_flood,
)
from sahayak.flood.preprocess import _find_band, preprocess_scene
from sahayak.flood.run import (
    _pre_scene_records,
    _read_aligned_slope,
    _record_outputs,
    _require_run_file,
    _write_json,
)
from sahayak.ingest.run import validate_run_id
from sahayak.provenance import RunManifest, require_input

DIAGNOSTIC_PERCENTILES = (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)
SLOPE_BANDS = ((0.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 45.0))
CLASS_NAMES_BY_CODE = {
    CLASS_NONE: "none",
    CLASS_NEW_WATER: "new_water",
    CLASS_DISTURBED: "disturbed_surface",
    CLASS_PERMANENT_WATER: "permanent_water",
    CLASS_DARK_LOW_CONFIDENCE: "dark_low_confidence",
    CLASS_INVALID: "invalid",
}


def run_flood_diagnostics(
    *,
    run_id: str,
    config_path: Path = Path("configs/flood.toml"),
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Write candidate funnels and a pre-event null-pair calibration for one run."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    manifest_path = require_input(run_dir / "manifest.json", active_settings)
    selection_path = require_input(run_dir / "selection.json", active_settings)
    manifest = RunManifest.read(manifest_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    s1_selection = selection.get("s1")
    if not isinstance(s1_selection, dict):
        raise ValueError("Run selection.json has no Sentinel-1 selection.")

    resolved_config_path = config_path
    if not resolved_config_path.is_absolute():
        resolved_config_path = active_settings.env_file_path.parent / resolved_config_path
    config = load_flood_config(resolved_config_path.resolve())
    records = _pre_scene_records(s1_selection, manifest.event_date.isoformat())
    if len(records) < 3:
        raise ValueError(
            f"Flood diagnostics require 3 pre-event scenes; selection.json lists {len(records)}."
        )
    records = records[-3:]
    scene_paths = [
        _require_run_file(run_dir, str(record["file"]), active_settings)
        for record in records
    ]
    post_path = _require_run_file(run_dir, "s1_post.tif", active_settings)
    slope_path = _require_run_file(run_dir, "slope.tif", active_settings)
    hand_path = _require_flood_output(run_dir, "hand.tif", active_settings)
    streams_path = _require_flood_output(run_dir, "streams.gpkg", active_settings)
    main_river_path = _require_flood_output(run_dir, "main_river.tif", active_settings)
    pre_scenes = [preprocess_scene(path, config, active_settings) for path in scene_paths]
    post_scene = preprocess_scene(post_path, config, active_settings)
    with rasterio.open(post_path) as post_source:
        profile = post_source.profile.copy()
        post_transform = post_source.transform
        post_crs = post_source.crs
        pixel_size = abs(post_transform.a)
        pixel_area_m2 = abs(post_transform.a * post_transform.e)
    if post_crs is None:
        raise ValueError("Post-event Sentinel-1 raster has no CRS.")
    hand = _read_aligned_raster(hand_path, profile, "HAND")
    slope = _read_aligned_slope(slope_path, profile)
    hand[~np.isfinite(slope)] = np.nan
    streams = _rasterize_streams(streams_path, profile)
    main_river_values = _read_aligned_raster(main_river_path, profile, "Main river")
    main_river = np.isfinite(main_river_values) & (main_river_values > 0)

    scene_dates = {
        str(record["file"]): str(record["datetime"])
        for record in records
    }
    post_record = s1_selection["pair"]["post"]
    if isinstance(post_record, dict) and isinstance(post_record.get("datetime"), str):
        scene_dates[post_path.name] = post_record["datetime"]
    input_files = [
        *(path.name for path in scene_paths),
        post_path.name,
        slope_path.name,
        hand_path.name,
        streams_path.name,
    ]

    real_funnel = classification_funnels(
        pre_scenes,
        post_scene,
        hand,
        slope,
        streams,
        config,
        max_hand_m=config.default_max_hand_m,
        pixel_size_m=pixel_size,
    )
    real_arrays = real_funnel.pop("arrays")
    real_detection = detect_flood(
        pre_scenes,
        post_scene,
        hand,
        slope,
        streams,
        config,
        max_hand_m=config.default_max_hand_m,
        pixel_size_m=pixel_size,
    )
    valley = np.isfinite(hand) & (hand <= 40.0)
    stream_count = int(streams.sum())
    real_valid = real_arrays["valid"]
    real_funnel["valid_fraction_in_hand_40m_corridor"] = (
        int((real_valid & valley).sum()) / int(valley.sum()) if valley.any() else None
    )
    real_funnel["valid_fraction_whole_tile"] = float(real_valid.mean())
    real_funnel["valid_stream_cell_fraction"] = (
        int((streams & real_valid).sum()) / stream_count if stream_count else None
    )
    real_funnel["class_areas_ha"] = _class_areas(real_detection.classes, pixel_area_m2)

    null_baseline = pre_scenes[:2]
    null_post = pre_scenes[2]
    null_funnel = classification_funnels(
        null_baseline,
        null_post,
        hand,
        slope,
        streams,
        config,
        max_hand_m=config.default_max_hand_m,
        pixel_size_m=pixel_size,
    )
    null_arrays = null_funnel.pop("arrays")
    null_detection = detect_flood(
        null_baseline,
        null_post,
        hand,
        slope,
        streams,
        config,
        max_hand_m=config.default_max_hand_m,
        pixel_size_m=pixel_size,
    )
    null_valid_count = int(null_detection.valid_mask.sum())
    false_alarm_stats = false_alarm_accounting(
        null_detection.classes,
        pixel_area_m2=pixel_area_m2,
        valid_pixel_count=null_valid_count,
    )
    null_funnel["class_areas_ha"] = false_alarm_stats["class_areas_ha"]
    null_funnel["false_alarm_pixels"] = false_alarm_stats["false_alarm_pixels"]
    null_funnel["false_alarms_per_1000_valid_pixels"] = false_alarm_stats[
        "false_alarms_per_1000_valid_pixels"
    ]
    null_funnel["sigma_null_db"] = _null_noise_estimates(
        null_arrays["delta_vv_db"],
        null_arrays["delta_vh_db"],
        null_arrays["valid"],
        slope,
    )
    null_funnel["valid_fraction_whole_tile"] = float(
        null_valid_count / null_detection.valid_mask.size
    )
    null_funnel["valid_fraction_in_hand_40m_corridor"] = (
        int((null_detection.valid_mask & valley).sum()) / int(valley.sum())
        if valley.any()
        else None
    )
    null_funnel["valid_stream_cell_fraction"] = (
        int((null_detection.valid_mask & streams).sum()) / stream_count
        if stream_count
        else None
    )
    corridor_validity = corridor_validity_breakdown(
        [*scene_paths, post_path],
        hand,
        streams,
        main_river,
        config,
    )

    provenance = {
        "input_files": input_files,
        "scene_dates": scene_dates,
        "thresholds_used": _threshold_provenance(config),
        "null_pair_definition": {
            "baseline_files": [scene_paths[0].name, scene_paths[1].name],
            "pseudo_post_file": scene_paths[2].name,
        },
    }
    diagnostics = {
        "module": "sahayak.flood.diagnostics",
        "provenance": provenance,
        "real_pair": _wrap_pair_facts(real_funnel),
        "null_pair": _wrap_pair_facts(null_funnel),
        "corridor_validity": corridor_validity,
        "percentiles": list(DIAGNOSTIC_PERCENTILES),
        "slope_bands_degrees": [
            {
                "minimum_inclusive": low,
                "maximum_inclusive" if index == len(SLOPE_BANDS) - 1 else "maximum_exclusive": high,
            }
            for index, (low, high) in enumerate(SLOPE_BANDS)
        ],
    }
    diagnostics_path = run_dir / "flood" / "diagnostics.json"
    _write_json(diagnostics_path, diagnostics)
    _merge_diagnostics_facts(run_dir / "facts.json", diagnostics)
    _record_outputs(manifest, [diagnostics_path, run_dir / "facts.json"], run_dir, active_settings)
    manifest.write(manifest_path)
    return diagnostics


def _read_aligned_raster(
    path: Path,
    reference_profile: dict[str, Any],
    label: str,
) -> np.ndarray:
    """Read a raster only when it exactly matches the reference grid."""
    with rasterio.open(path) as source:
        if (
            source.shape != (reference_profile["height"], reference_profile["width"])
            or source.transform != reference_profile["transform"]
            or source.crs != reference_profile["crs"]
        ):
            raise ValueError(f"{label} raster is not aligned to the post-event Sentinel-1 grid.")
        values = source.read(1).astype(np.float32)
        if source.nodata is not None:
            values[values == source.nodata] = np.nan
    return values


def _require_flood_output(run_dir: Path, filename: str, settings: Settings) -> Path:
    path = require_input(run_dir / "flood" / filename, settings)
    if not path.is_file():
        raise FileNotFoundError(f"Required flood output does not exist: {path}.")
    return path


def _rasterize_streams(
    path: Path,
    reference_profile: dict[str, Any],
) -> np.ndarray:
    """Rasterize the saved stream polygons onto the Sentinel-1 grid."""
    streams = gpd.read_file(path, layer="streams")
    if streams.crs is None:
        raise ValueError("Saved streams layer has no CRS.")
    reference_crs = reference_profile["crs"]
    if reference_crs is None:
        raise ValueError("Post-event Sentinel-1 raster has no CRS.")
    if streams.crs != reference_crs:
        streams = streams.to_crs(reference_crs)
    geometries = [
        geometry
        for geometry in streams.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        raise ValueError("Saved streams layer contains no geometries.")
    return rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=(reference_profile["height"], reference_profile["width"]),
        transform=reference_profile["transform"],
        fill=0,
        dtype="uint8",
    ).astype(bool)


def _null_noise_estimates(
    delta_vv: np.ndarray,
    delta_vh: np.ndarray,
    valid: np.ndarray,
    slope: np.ndarray,
) -> dict[str, Any]:
    """Estimate robust null-pair noise overall and in local-slope intervals."""
    results: dict[str, Any] = {
        "overall": {
            "vv": _sigma_mad(delta_vv[valid]),
            "vh": _sigma_mad(delta_vh[valid]),
            "valid_pixel_count": int(valid.sum()),
        },
        "by_slope_band": {},
    }
    for index, (low, high) in enumerate(SLOPE_BANDS):
        upper_bound = slope <= high if index == len(SLOPE_BANDS) - 1 else slope < high
        band = valid & np.isfinite(slope) & (slope >= low) & upper_bound
        key = f"{low:g}-{high:g}_degrees"
        results["by_slope_band"][key] = {
            "vv": _sigma_mad(delta_vv[band]),
            "vh": _sigma_mad(delta_vh[band]),
            "valid_pixel_count": int(band.sum()),
        }
    return results


def _sigma_mad(values: np.ndarray) -> float | None:
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        return None
    median = np.median(finite)
    return float(1.4826 * np.median(np.abs(finite - median)))


def corridor_validity_breakdown(
    scene_paths: list[Path],
    hand: np.ndarray,
    streams: np.ndarray,
    main_river: np.ndarray,
    config: FloodConfig,
) -> dict[str, Any]:
    """Count exclusive invalid-mask reasons in terrain and drainage zones."""
    if not (hand.shape == streams.shape == main_river.shape):
        raise ValueError("HAND, stream, and main-river masks must share one grid.")
    zones = {
        "whole_tile": np.ones(hand.shape, dtype=bool),
        "hand_le_40m": np.isfinite(hand) & (hand <= 40.0),
        "stream_cells": streams,
        "main_river_cells": main_river,
    }
    reason_names = (
        "dataMask_zero",
        "shadowMask_flagged",
        "incidence_angle_outside_range",
        "physical_ceiling_exceeded",
        "nonfinite_or_nonpositive",
    )
    counts: dict[str, dict[str, dict[str, Any]]] = {}
    totals: dict[str, dict[str, Any]] = {}
    main_river_angles: dict[str, np.ndarray] = {}
    for scene_path in scene_paths:
        with rasterio.open(scene_path) as source:
            descriptions = tuple(source.descriptions)
            vv = source.read(_find_band(descriptions, "vv")).astype(np.float32)
            vh = source.read(_find_band(descriptions, "vh")).astype(np.float32)
            incidence = source.read(
                _find_band(descriptions, "incidence angle")
            ).astype(np.float32)
            shadow = source.read(_find_band(descriptions, "shadow mask")).astype(np.float32)
            data_mask = source.read(_find_band(descriptions, "data mask")).astype(np.float32)
            nodata = source.nodata
        if vv.shape != hand.shape:
            raise ValueError(f"Sentinel-1 scene {scene_path.name} is not aligned to HAND.")
        finite = (
            np.isfinite(vv)
            & np.isfinite(vh)
            & np.isfinite(incidence)
            & np.isfinite(shadow)
            & np.isfinite(data_mask)
        )
        valid = (
            finite
            & (data_mask > 0)
            & (shadow <= 0)
            & (incidence >= config.minimum_incidence_angle_deg)
            & (incidence <= config.maximum_incidence_angle_deg)
            & (vv > 0)
            & (vh > 0)
            & (vv <= 10.0 ** (config.vv_ceiling_db / 10.0))
            & (vh <= 10.0 ** (config.vh_ceiling_db / 10.0))
        )
        if nodata is not None:
            valid &= (vv != nodata) & (vh != nodata)
        invalid = ~valid
        candidates = {
            "dataMask_zero": np.isfinite(data_mask) & (data_mask <= 0),
            "shadowMask_flagged": np.isfinite(shadow) & (shadow > 0),
            "incidence_angle_outside_range": np.isfinite(incidence)
            & (
                (incidence < config.minimum_incidence_angle_deg)
                | (incidence > config.maximum_incidence_angle_deg)
            ),
            "physical_ceiling_exceeded": (
                (np.isfinite(vv) & (vv > 10.0 ** (config.vv_ceiling_db / 10.0)))
                | (np.isfinite(vh) & (vh > 10.0 ** (config.vh_ceiling_db / 10.0)))
            ),
            "nonfinite_or_nonpositive": (
                ~finite
                | (vv <= 0)
                | (vh <= 0)
                | ((vv == nodata) if nodata is not None else False)
                | ((vh == nodata) if nodata is not None else False)
            ),
        }
        remaining = invalid.copy()
        exclusive: dict[str, np.ndarray] = {}
        for reason in reason_names:
            exclusive[reason] = remaining & candidates[reason]
            remaining &= ~exclusive[reason]
        exclusive["nonfinite_or_nonpositive"] |= remaining

        image_name = scene_path.name
        counts[image_name] = {}
        for zone_name, zone_mask in zones.items():
            zone_size = int(zone_mask.sum())
            zone_invalid = invalid & zone_mask
            invalid_count = int(zone_invalid.sum())
            reason_counts = {
                reason: int((exclusive[reason] & zone_mask).sum())
                for reason in reason_names
            }
            counts[image_name][zone_name] = {
                "cell_count": zone_size,
                "valid_count": zone_size - invalid_count,
                "invalid_count": invalid_count,
                "invalid_fraction": invalid_count / zone_size if zone_size else None,
                "invalid_reason_counts": reason_counts,
                "invalid_reason_share": {
                    reason: (count / invalid_count if invalid_count else None)
                    for reason, count in reason_counts.items()
                },
            }
        angle_valid = (
            main_river
            & np.isfinite(incidence)
            & (incidence > 0)
            & (incidence <= 90)
            & (data_mask > 0)
        )
        if nodata is not None:
            angle_valid &= incidence != nodata
        main_angles = incidence[angle_valid]
        main_river_angles[image_name] = main_angles

    totals["combined"] = {}
    for zone_name in zones:
        cell_count = sum(
            zone_counts[zone_name]["cell_count"] for zone_counts in counts.values()
        )
        valid_count = sum(
            zone_counts[zone_name]["valid_count"] for zone_counts in counts.values()
        )
        invalid_count = sum(
            zone_counts[zone_name]["invalid_count"] for zone_counts in counts.values()
        )
        reason_counts = {
            reason: sum(
                zone_counts[zone_name]["invalid_reason_counts"][reason]
                for zone_counts in counts.values()
            )
            for reason in reason_names
        }
        totals["combined"][zone_name] = {
            "cell_count": cell_count,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "invalid_fraction": invalid_count / cell_count if cell_count else None,
            "invalid_reason_counts": reason_counts,
            "invalid_reason_share": {
                reason: (count / invalid_count if invalid_count else None)
                for reason, count in reason_counts.items()
            },
        }
    angle_values = np.concatenate(list(main_river_angles.values()))
    totals["combined"]["main_river_incidence_angle_degrees"] = _angle_summary(angle_values)
    for image_name, values in main_river_angles.items():
        counts[image_name]["main_river_incidence_angle_degrees"] = _angle_summary(values)
    return {"per_image": counts, **totals}


def _angle_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
            "below_30_fraction": None,
        }
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
        "below_30_fraction": float(np.mean(finite < 30.0)),
    }


def _class_areas(classes: np.ndarray, pixel_area_m2: float) -> dict[str, float]:
    return {
        name: float((classes == code).sum() * pixel_area_m2 / 10_000)
        for code, name in CLASS_NAMES_BY_CODE.items()
        if code != CLASS_INVALID
    }


def false_alarm_accounting(
    classes: np.ndarray,
    *,
    pixel_area_m2: float,
    valid_pixel_count: int,
) -> dict[str, Any]:
    """Measure every non-background null-pair class as a false alarm."""
    areas = _class_areas(classes, pixel_area_m2)
    counts = {
        name: int((classes == class_code).sum())
        for class_code, name in CLASS_NAMES_BY_CODE.items()
        if class_code not in {CLASS_NONE, CLASS_INVALID}
    }
    return {
        "class_areas_ha": areas,
        "false_alarm_pixels": counts,
        "false_alarms_per_1000_valid_pixels": {
            name: count * 1_000 / valid_pixel_count if valid_pixel_count else None
            for name, count in counts.items()
        },
    }


def _threshold_provenance(config: FloodConfig) -> dict[str, dict[str, object]]:
    """Store diagnostic thresholds once in the diagnostic provenance section."""
    values = {
        "default_max_hand": (config.default_max_hand_m, "m"),
        "vv_threshold_min": (config.vv_water_threshold_min_db, "dB"),
        "vv_threshold_max": (config.vv_water_threshold_max_db, "dB"),
        "vh_water_threshold": (config.vh_water_threshold_db, "dB"),
        "baseline_water_margin": (config.baseline_water_margin_db, "dB"),
        "new_water_delta": (config.new_water_delta_db, "dB"),
        "new_water_max_slope": (config.new_water_max_slope_deg, "degrees"),
        "disturbed_change": (config.disturbed_change_db, "dB"),
        "disturbed_z_minimum": (config.disturbed_z_minimum, "standard deviations"),
        "disturbed_max_slope": (config.disturbed_max_slope_deg, "degrees"),
        "permanent_water_max_hand": (15.0, "m"),
        "permanent_water_max_slope": (10.0, "degrees"),
        "minimum_component_area": (config.minimum_component_area_ha, "ha"),
        "proximity_distance": (config.proximity_distance_m, "m"),
    }
    return {
        name: {
            "value": value,
            "unit": unit,
            "module": "sahayak.flood.config",
            "provenance": "flood_diagnostics.provenance",
        }
        for name, (value, unit) in values.items()
    }


def _wrap_pair_facts(pair: dict[str, Any]) -> dict[str, Any]:
    """Attach units and a shared provenance reference to reported pair diagnostics."""
    units = {
        "delta_vv_db": "dB",
        "delta_vh_db": "dB",
        "z_vv": "z-score",
        "z_vh": "z-score",
        "post_vv_db": "dB",
        "post_vh_db": "dB",
        "baseline_mean_vv_db": "dB",
        "baseline_mean_vh_db": "dB",
    }
    wrapped = {
        "raw_otsu_threshold_db": _diagnostic_fact(pair["raw_otsu_threshold_db"], "dB"),
        "clamped_otsu_threshold_db": _diagnostic_fact(
            pair["clamped_otsu_threshold_db"],
            "dB",
        ),
        "valid_pixel_count": _diagnostic_fact(pair["valid_pixel_count"], "pixels"),
        "valid_pixel_fraction": _diagnostic_fact(pair["valid_pixel_fraction"], "fraction"),
        "valid_fraction_in_hand_40m_corridor": _diagnostic_fact(
            pair["valid_fraction_in_hand_40m_corridor"],
            "fraction",
        ),
        "valid_fraction_whole_tile": _diagnostic_fact(
            pair["valid_fraction_whole_tile"],
            "fraction",
        ),
        "valid_stream_cell_fraction": _diagnostic_fact(
            pair.get("valid_stream_cell_fraction"),
            "fraction",
        ),
        "distributions": {
            metric: {
                percentile: _diagnostic_fact(value, units[metric])
                for percentile, value in values.items()
            }
            for metric, values in pair["distributions"].items()
        },
        "water": _wrap_funnel_counts(pair["water"]),
        "disturbed_surface": _wrap_funnel_counts(pair["disturbed_surface"]),
        "permanent_water": _wrap_funnel_counts(pair["permanent_water"]),
        "permanent_hand_and_slope_percentiles": {
            name: (
                {
                    percentile: _diagnostic_fact(value, "m")
                    for percentile, value in values.items()
                }
                if name in {"before_restriction", "after_restriction"}
                else {
                    percentile: _diagnostic_fact(value, "degrees")
                    for percentile, value in values.items()
                }
                if name.startswith("slope_")
                else _diagnostic_fact(values, "pixels")
            )
            for name, values in pair["permanent_hand_and_slope_percentiles"].items()
        },
        "class_areas_ha": {
            name: _diagnostic_fact(value, "ha")
            for name, value in pair["class_areas_ha"].items()
        },
    }
    if "false_alarm_pixels" in pair:
        wrapped["false_alarm_pixels"] = {
            name: _diagnostic_fact(value, "pixels")
            for name, value in pair["false_alarm_pixels"].items()
        }
        wrapped["false_alarms_per_1000_valid_pixels"] = {
            name: _diagnostic_fact(value, "alarms per 1,000 valid pixels")
            for name, value in pair["false_alarms_per_1000_valid_pixels"].items()
        }
        wrapped["sigma_null_db"] = {
            "overall": _wrap_sigma(pair["sigma_null_db"]["overall"]),
            "by_slope_band": {
                band: _wrap_sigma(values)
                for band, values in pair["sigma_null_db"]["by_slope_band"].items()
            },
        }
    return wrapped


def _wrap_funnel_counts(values: dict[str, int]) -> dict[str, dict[str, object]]:
    return {
        name: _diagnostic_fact(count, "pixels")
        for name, count in values.items()
    }


def _wrap_sigma(values: dict[str, Any]) -> dict[str, dict[str, object]]:
    return {
        "vv": _diagnostic_fact(values["vv"], "dB"),
        "vh": _diagnostic_fact(values["vh"], "dB"),
        "valid_pixel_count": _diagnostic_fact(values["valid_pixel_count"], "pixels"),
    }


def _diagnostic_fact(value: object, unit: str) -> dict[str, object]:
    return {
        "value": value,
        "unit": unit,
        "module": "sahayak.flood.diagnostics",
        "provenance": "flood_diagnostics.provenance",
    }


def _merge_diagnostics_facts(path: Path, diagnostics: dict[str, Any]) -> None:
    """Merge diagnostic outputs without replacing unrelated fact sections."""
    facts: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Existing facts file {path} must contain a JSON object.")
        facts = loaded
    facts["flood_diagnostics"] = diagnostics
    _write_json(path, facts)


def format_flood_diagnostics(diagnostics: dict[str, Any]) -> str:
    """Render diagnostics as complete console tables from the stored JSON structure."""
    lines = ["Flood candidate diagnostics"]
    real_distributions = diagnostics["real_pair"]["distributions"]
    null_distributions = diagnostics["null_pair"]["distributions"]
    if "corridor_validity" in diagnostics:
        lines.extend(_format_corridor_validity(diagnostics["corridor_validity"]))
    paired_headers = [
        f"p{percentile} real/null"
        for percentile in DIAGNOSTIC_PERCENTILES
    ]
    lines.append("\nPercentiles (real pair beside pre-event null pair)")
    lines.append("Metric | " + " | ".join(paired_headers))
    for metric, real_values in real_distributions.items():
        null_values = null_distributions[metric]
        paired_values = " | ".join(
            f"{_format_optional(real_values[str(percentile)])} / "
            f"{_format_optional(null_values[str(percentile)])}"
            for percentile in DIAGNOSTIC_PERCENTILES
        )
        lines.append(f"{metric} | {paired_values}")

    for pair_name in ("real_pair", "null_pair"):
        pair = diagnostics[pair_name]
        lines.append(f"\n{pair_name.replace('_', ' ').title()}")
        lines.append(
            "Otsu raw/clamped (dB): "
            f"{_format_optional(_metric_value(pair['raw_otsu_threshold_db']))} / "
            f"{_format_optional(_metric_value(pair['clamped_otsu_threshold_db']))}"
        )
        valley_valid = _format_optional(
            _metric_value(pair["valid_fraction_in_hand_40m_corridor"])
        )
        stream_valid = _format_optional(_metric_value(pair["valid_stream_cell_fraction"]))
        valid_fraction = _metric_value(pair["valid_fraction_whole_tile"])
        lines.append(
            f"Valid pixels: {_metric_value(pair['valid_pixel_count'])} "
            f"({valid_fraction:.4%} tile); "
            f"valley HAND<=40 m fraction={valley_valid}; "
            f"stream-cell-valid fraction={stream_valid}"
        )
        lines.append("Candidate funnel (cumulative pixel count)")
        for class_name in ("water", "disturbed_surface", "permanent_water"):
            counts = pair[class_name]
            lines.append(
                class_name
                + " | "
                + " | ".join(
                    f"{stage}={_metric_value(count)}"
                    for stage, count in counts.items()
                )
            )
        lines.append("Class area (ha)")
        for name, area in pair["class_areas_ha"].items():
            lines.append(f"  {name}: {_format_optional(_metric_value(area))}")
        if pair_name == "null_pair":
            lines.append("False alarms per 1,000 valid pixels")
            for name, value in pair["false_alarms_per_1000_valid_pixels"].items():
                lines.append(f"  {name}: {_format_optional(_metric_value(value))}")
            lines.append("Null-pair robust noise sigma (1.4826 * MAD, dB)")
            for label, values in [("overall", pair["sigma_null_db"]["overall"]), *pair[
                "sigma_null_db"
            ]["by_slope_band"].items()]:
                lines.append(
                    f"  {label}: VV={_format_optional(_metric_value(values['vv']))}; "
                    f"VH={_format_optional(_metric_value(values['vh']))}; "
                    f"n={_metric_value(values['valid_pixel_count'])}"
                )
        permanent_stats = pair["permanent_hand_and_slope_percentiles"]
        lines.append("Permanent-water HAND/slope percentiles")
        for name, values in permanent_stats.items():
            if name.startswith(("before_restriction", "after_restriction", "slope_")):
                lines.append(
                    f"  {name}: "
                    + ", ".join(
                        f"p{key}={_format_optional(_metric_value(value))}"
                        for key, value in values.items()
                    )
                )
            else:
                lines.append(f"  {name}: {_format_optional(values)}")
    return "\n".join(lines)


def _format_corridor_validity(corridor: dict[str, Any]) -> list[str]:
    reason_names = (
        "dataMask_zero",
        "shadowMask_flagged",
        "incidence_angle_outside_range",
        "physical_ceiling_exceeded",
        "nonfinite_or_nonpositive",
    )
    lines = [
        "Corridor validity by acquisition (reason shares are of invalid cells)",
        "Image | Zone | invalid fraction | "
        + " | ".join(reason_names),
    ]
    for image_name, zones in corridor["per_image"].items():
        for zone_name, metrics in zones.items():
            if zone_name == "main_river_incidence_angle_degrees":
                angle = metrics
                lines.append(
                    f"{image_name} | main-river incidence degrees | "
                    f"n={angle['count']}; min={_format_optional(angle['minimum'])}; "
                    f"median={_format_optional(angle['median'])}; "
                    f"max={_format_optional(angle['maximum'])}; "
                    f"below 30={_format_optional(angle['below_30_fraction'])}"
                )
                continue
            shares = " | ".join(
                _format_optional(metrics["invalid_reason_share"][reason])
                for reason in reason_names
            )
            lines.append(
                f"{image_name} | {zone_name} | "
                f"{_format_optional(metrics['invalid_fraction'])} | {shares}"
            )
    for zone_name, metrics in corridor["combined"].items():
        if zone_name == "main_river_incidence_angle_degrees":
            angle = metrics
            lines.append(
                "combined | main-river incidence degrees | "
                f"n={angle['count']}; min={_format_optional(angle['minimum'])}; "
                f"median={_format_optional(angle['median'])}; "
                f"max={_format_optional(angle['maximum'])}; "
                f"below 30={_format_optional(angle['below_30_fraction'])}"
            )
            continue
        shares = " | ".join(
            _format_optional(metrics["invalid_reason_share"][reason])
            for reason in reason_names
        )
        lines.append(
            f"combined | {zone_name} | "
            f"{_format_optional(metrics['invalid_fraction'])} | {shares}"
        )
    return lines


def _metric_value(value: object) -> object:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _format_optional(value: object) -> str:
    value = _metric_value(value)
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.4f}" if isinstance(value, float) else str(value)
    return str(value)
