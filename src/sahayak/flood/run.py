"""Flood mapping orchestration, geospatial outputs, and run-facts integration."""

import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape
from shapely.ops import unary_union

from sahayak.config import Settings
from sahayak.flood.config import FloodConfig, load_flood_config
from sahayak.flood.detect import (
    CLASS_DISTURBED,
    CLASS_INVALID,
    CLASS_NEW_WATER,
    CLASS_NONE,
    CLASS_PERMANENT_WATER,
    FloodDetection,
    detect_flood,
)
from sahayak.flood.preprocess import RadarScene, preprocess_scene
from sahayak.flood.terrain import build_hand
from sahayak.ingest.run import validate_run_id
from sahayak.provenance import RunManifest, require_input

LOGGER = logging.getLogger(__name__)
FLOOD_ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026."
CLASS_NAMES = {
    CLASS_NONE: "none",
    CLASS_NEW_WATER: "new_water",
    CLASS_DISTURBED: "disturbed_surface",
    CLASS_PERMANENT_WATER: "permanent_water",
}


def run_flood(
    *,
    run_id: str,
    max_hand_m: float | None = None,
    config_path: Path = Path("configs/flood.toml"),
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run preprocessing, PySheds HAND, classification, and outputs for an ingested run."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    manifest_path = run_dir / "manifest.json"
    selection_path = run_dir / "selection.json"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(
            f"Run {run_id!r} requires existing manifest.json and selection.json files."
        )
    manifest = RunManifest.read(manifest_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    s1_selection = selection.get("s1")
    if not isinstance(s1_selection, dict):
        raise ValueError("Run selection.json has no Sentinel-1 selection.")

    resolved_config_path = config_path
    if not resolved_config_path.is_absolute():
        resolved_config_path = active_settings.env_file_path.parent / resolved_config_path
    config = load_flood_config(resolved_config_path.resolve())
    selected_max_hand = max_hand_m if max_hand_m is not None else config.default_max_hand_m
    if not math.isfinite(selected_max_hand) or selected_max_hand <= 0:
        raise ValueError("--max-hand must be a positive finite number of metres.")

    post_path = _require_run_file(run_dir, "s1_post.tif", active_settings)
    dem_path = _require_run_file(run_dir, "dem.tif", active_settings)
    slope_path = _require_run_file(run_dir, "slope.tif", active_settings)
    scene_records = _pre_scene_records(s1_selection, manifest.event_date.isoformat())
    if len(scene_records) < 3:
        raise ValueError(
            f"Flood mapping requires 3 pre-event scenes; selection.json lists {len(scene_records)}."
        )
    scene_records = scene_records[-3:]
    pre_paths = [
        _require_run_file(run_dir, str(record["file"]), active_settings)
        for record in scene_records
    ]
    scene_dates = {
        str(record["file"]): str(record["datetime"])
        for record in scene_records
    }
    post_record = s1_selection["pair"]["post"]
    if not isinstance(post_record, dict) or not isinstance(post_record.get("datetime"), str):
        raise ValueError("Run selection.json is missing the post-event acquisition datetime.")
    scene_dates[post_path.name] = post_record["datetime"]
    input_files = [path.name for path in (*pre_paths, post_path, dem_path, slope_path)]

    flood_dir = run_dir / "flood"
    flood_dir.mkdir(parents=True, exist_ok=True)
    pre_scenes = [preprocess_scene(path, config, active_settings) for path in pre_paths]
    post_scene = preprocess_scene(post_path, config, active_settings)
    with rasterio.open(post_path) as post_source:
        post_profile = post_source.profile.copy()
        post_crs = post_source.crs
        post_transform = post_source.transform
        if post_crs is None:
            raise ValueError("Post-event Sentinel-1 raster has no CRS.")
        pixel_size_x = abs(post_transform.a)
        pixel_size_y = abs(post_transform.e)
        pixel_area_m2 = pixel_size_x * pixel_size_y
    if not math.isclose(pixel_size_x, pixel_size_y, rel_tol=1e-6):
        raise ValueError("Flood mapping currently requires square pixels.")

    hand_path = flood_dir / "hand.tif"
    streams_path = flood_dir / "streams.gpkg"
    hand, stream_mask, terrain_metadata = build_hand(
        dem_path,
        hand_path,
        streams_path,
        config,
        active_settings,
    )
    slope = _read_aligned_slope(slope_path, post_profile)
    hand[~np.isfinite(slope)] = np.nan

    sensitivity_values = list(config.sensitivity_max_hand_m)
    sensitivity_values.extend((config.default_max_hand_m, selected_max_hand))
    sensitivity_values = list(dict.fromkeys(sensitivity_values))
    detections: dict[float, FloodDetection] = {}
    sensitivity_areas: dict[float, dict[str, float]] = {}
    for sensitivity_hand in sensitivity_values:
        detection = detect_flood(
            pre_scenes,
            post_scene,
            hand,
            slope,
            stream_mask,
            config,
            max_hand_m=sensitivity_hand,
            pixel_size_m=pixel_size_x,
        )
        detections[sensitivity_hand] = detection
        sensitivity_areas[sensitivity_hand] = _class_areas(detection.classes, pixel_area_m2)
    result = detections[selected_max_hand]
    threshold_values = _threshold_values(config, selected_max_hand, result.vv_water_threshold_db)
    stream_baseline_count = np.zeros(stream_mask.shape, dtype=np.uint8)
    stream_baseline_sum = np.zeros(stream_mask.shape, dtype=np.float32)
    for scene in pre_scenes:
        scene_valid = scene.valid_mask & np.isfinite(scene.vv_db)
        stream_baseline_count += scene_valid.astype(np.uint8)
        stream_baseline_sum += np.where(scene_valid, scene.vv_db, 0.0)
    stream_baseline_valid = stream_mask & (stream_baseline_count >= 2)
    stream_baseline_mean = np.divide(
        stream_baseline_sum,
        stream_baseline_count,
        out=np.full(stream_mask.shape, np.nan, dtype=np.float32),
        where=stream_baseline_count > 0,
    )
    stream_baseline_cells = int(stream_baseline_valid.sum())
    stream_below_water_cells = int(
        (
            stream_baseline_valid
            & (stream_baseline_mean < result.vv_water_threshold_db)
        ).sum()
    )
    stream_below_water_fraction = (
        stream_below_water_cells / stream_baseline_cells
        if stream_baseline_cells
        else None
    )

    outputs = {
        "flood_class": flood_dir / "flood_class.tif",
        "flood_confidence": flood_dir / "flood_confidence.tif",
        "change_vv_db": flood_dir / "change_vv_db.tif",
        "change_vh_db": flood_dir / "change_vh_db.tif",
        "valid_mask": flood_dir / "valid_mask.tif",
    }
    _write_raster(outputs["flood_class"], result.classes, post_profile, nodata=CLASS_INVALID)
    _write_raster(outputs["flood_confidence"], result.confidence, post_profile, nodata=-9999.0)
    _write_raster(outputs["change_vv_db"], result.change_vv_db, post_profile, nodata=-9999.0)
    _write_raster(outputs["change_vh_db"], result.change_vh_db, post_profile, nodata=-9999.0)
    _write_raster(
        outputs["valid_mask"],
        result.valid_mask.astype(np.uint8),
        post_profile,
        dtype="uint8",
        nodata=255,
    )
    polygons_path = flood_dir / "flood_polygons.gpkg"
    polygon_count = _write_polygons(
        polygons_path,
        result,
        post_transform,
        post_crs,
    )
    _write_quicklooks(
        flood_dir,
        post_scene,
        result,
        hand,
        post_transform,
    )

    invalid_distance = _distance_to_stream(stream_mask, pixel_size_x)
    new_water = result.classes == CLASS_NEW_WATER
    new_water_count = int(new_water.sum())
    near_stream_fraction = _fraction_or_none(
        int((new_water & (invalid_distance <= 100.0)).sum()),
        new_water_count,
    )
    hand_fraction = _fraction_or_none(
        int((new_water & (hand <= 10.0)).sum()),
        new_water_count,
    )
    provenance = {
        "input_files": input_files,
        "scene_dates": scene_dates,
        "thresholds_used": {
            key: _measurement(value, unit, input_files, scene_dates)
            for key, (value, unit) in threshold_values.items()
        },
    }
    sensitivity_stats = [
        {
            "max_hand": _measurement(
                value,
                "m",
                input_files,
                scene_dates,
                thresholds={**threshold_values, "max_hand": (value, "m")},
            ),
            "area_ha_by_class": {
                name: _measurement(
                    areas[name],
                    "ha",
                    input_files,
                    scene_dates,
                    thresholds={**threshold_values, "max_hand": (value, "m")},
                )
                for name in CLASS_NAMES.values()
            },
        }
        for value, areas in sorted(sensitivity_areas.items())
    ]
    flood_facts = {
        "module": "sahayak.flood.run",
        **provenance,
        "areas_ha_by_class": {
            name: _measurement(
                sensitivity_areas[selected_max_hand][name],
                "ha",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            )
            for name in CLASS_NAMES.values()
        },
        "valid_pixel_fraction": _measurement(
            float(result.valid_mask.mean()),
            "fraction",
            input_files,
            scene_dates,
            thresholds=threshold_values,
        ),
        "polygon_count": _measurement(
            polygon_count,
            "polygons",
            input_files,
            scene_dates,
            thresholds=threshold_values,
        ),
        "mean_confidence_by_class": {
            name: _measurement(
                float(np.mean(result.confidence[result.classes == class_code]))
                if np.any(result.classes == class_code)
                else 0.0,
                "score",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            )
            for class_code, name in CLASS_NAMES.items()
        },
        "new_water_area_within_hand_10m_fraction": _measurement(
            hand_fraction,
            "fraction",
            input_files,
            scene_dates,
            thresholds=threshold_values,
        ),
        "new_water_within_100m_of_stream_fraction": _measurement(
            near_stream_fraction,
            "fraction",
            input_files,
            scene_dates,
            thresholds=threshold_values,
        ),
        "sensitivity": sensitivity_stats,
        "terrain": {
            "module": "sahayak.flood.terrain",
            "algorithm": terrain_metadata["algorithm"],
            "terrain_resolution": _measurement(
                float(terrain_metadata["terrain_resolution_m"]),
                "m",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "stream_cell_count": _measurement(
                terrain_metadata["stream_cell_count"],
                "cells",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "hand_valid_pixel_fraction": _measurement(
                terrain_metadata["hand_valid_pixel_fraction"],
                "fraction",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "stream_area_threshold_cells": _measurement(
                int(terrain_metadata["stream_area_threshold_cells"]),
                "cells",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "pixel_area": _measurement(
                float(terrain_metadata["pixel_area_m2"]),
                "m2",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "pixel_size_x": _measurement(
                float(terrain_metadata["pixel_size_m"][0]),
                "m",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "pixel_size_y": _measurement(
                float(terrain_metadata["pixel_size_m"][1]),
                "m",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "stream_length": _measurement(
                float(terrain_metadata["stream_length_km"]),
                "km",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "stream_cells_below_water_threshold_fraction": _measurement(
                stream_below_water_fraction,
                "fraction",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "stream_cells_with_pre_event_baseline": _measurement(
                stream_baseline_cells,
                "cells",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "stream_cells_below_water_threshold": _measurement(
                stream_below_water_cells,
                "cells",
                input_files,
                scene_dates,
                thresholds=threshold_values,
            ),
            "hand_qc": {
                grid_name: {
                    "negative_count": _measurement(
                        int(qc["negative_count"]),
                        "cells",
                        input_files,
                        scene_dates,
                        thresholds=threshold_values,
                    ),
                    "valid_count": _measurement(
                        int(qc["valid_count"]),
                        "cells",
                        input_files,
                        scene_dates,
                        thresholds=threshold_values,
                    ),
                    "negative_fraction": _measurement(
                        float(qc["negative_fraction"]),
                        "fraction",
                        input_files,
                        scene_dates,
                        thresholds=threshold_values,
                    ),
                    "minimum_negative": _measurement(
                        qc["minimum_negative_m"],
                        "m",
                        input_files,
                        scene_dates,
                        thresholds=threshold_values,
                    ),
                }
                for grid_name, qc in terrain_metadata["hand_qc"].items()
            },
        },
        "filter_formula": (
            "Lee in linear power: local mean + max((local variance - noise variance) / "
            "local variance, 0) * (pixel - local mean); noise variance is the 20th "
            "percentile of valid local variances. Invalid samples are excluded from "
            "normalized window statistics."
        ),
        "confidence_formula": (
            "0.40 * min(max(abs(z_vv), abs(z_vh)) / 6, 1) + 0.25 * same-sign-change "
            "+ 0.20 * min(distance_to_invalid / 30 m, 1) + 0.15 * "
            "max(1 - slope / 35 degrees, 0), clipped component-wise to [0, 1]."
        ),
        "attribution": FLOOD_ATTRIBUTION,
    }
    _merge_facts(run_dir / "facts.json", {"flood": flood_facts})
    outputs["flood_polygons"] = polygons_path
    outputs["streams"] = streams_path
    outputs["hand"] = hand_path
    outputs["flood_overlay"] = flood_dir / "flood_overlay.png"
    outputs["change_vv"] = flood_dir / "change_vv.png"
    outputs["terrain_hand"] = flood_dir / "terrain_hand.png"
    outputs["stats"] = flood_dir / "stats.json"
    _write_json(outputs["stats"], flood_facts)
    facts_path = run_dir / "facts.json"
    output_paths = [*outputs.values(), facts_path]
    _record_outputs(manifest, output_paths, run_dir, active_settings)
    manifest.write(manifest_path)
    LOGGER.info(
        "Flood mapping completed with VV water threshold %.2f dB and max HAND %.1f m.",
        result.vv_water_threshold_db,
        selected_max_hand,
    )
    return {
        "run_dir": run_dir,
        "flood_dir": flood_dir,
        "facts_path": facts_path,
        "thresholds": threshold_values,
        "facts": flood_facts,
        "outputs": output_paths,
    }


def _pre_scene_records(s1_selection: dict[str, Any], event_date: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    baseline = s1_selection.get("baseline", [])
    if isinstance(baseline, list):
        for record in baseline:
            if isinstance(record, dict):
                _append_pre_record(records, seen, record, event_date)
    pair = s1_selection.get("pair")
    if isinstance(pair, dict) and isinstance(pair.get("pre"), dict):
        _append_pre_record(records, seen, pair["pre"], event_date)
    return sorted(records, key=lambda record: record["datetime"])


def _append_pre_record(
    records: list[dict[str, str]],
    seen: set[str],
    record: dict[str, Any],
    event_date: str,
) -> None:
    filename = record.get("file")
    timestamp = record.get("datetime")
    if (
        isinstance(filename, str)
        and isinstance(timestamp, str)
        and timestamp[:10] < event_date
        and filename not in seen
    ):
        records.append({"file": filename, "datetime": timestamp})
        seen.add(filename)


def _require_run_file(run_dir: Path, filename: str, settings: Settings) -> Path:
    path = require_input(run_dir / filename, settings)
    if not path.is_file():
        raise FileNotFoundError(f"Required flood input does not exist: {path}.")
    return path


def _read_aligned_slope(path: Path, reference_profile: dict[str, Any]) -> np.ndarray:
    with rasterio.open(path) as source:
        if (
            source.shape != (reference_profile["height"], reference_profile["width"])
            or source.transform != reference_profile["transform"]
            or source.crs != reference_profile["crs"]
        ):
            raise ValueError("Slope raster is not aligned to the post-event Sentinel-1 grid.")
        data = source.read(1).astype(np.float32)
        if source.nodata is not None:
            data[data == source.nodata] = np.nan
    return data


def _class_areas(classes: np.ndarray, pixel_area_m2: float) -> dict[str, float]:
    return {
        name: float((classes == code).sum() * pixel_area_m2 / 10_000)
        for code, name in CLASS_NAMES.items()
    }


def _threshold_values(
    config: FloodConfig,
    max_hand_m: float,
    otsu_threshold_db: float,
) -> dict[str, tuple[float, str]]:
    return {
        "minimum_incidence_angle": (config.minimum_incidence_angle_deg, "degrees"),
        "maximum_incidence_angle": (config.maximum_incidence_angle_deg, "degrees"),
        "vv_physical_ceiling": (config.vv_ceiling_db, "dB"),
        "vh_physical_ceiling": (config.vh_ceiling_db, "dB"),
        "lee_window_size": (float(config.lee_window_size), "pixels"),
        "stream_area": (config.stream_area_threshold_km2, "km2"),
        "baseline_standard_deviation_floor": (config.minimum_baseline_std_db, "dB"),
        "vv_otsu_water_threshold": (otsu_threshold_db, "dB"),
        "vh_water_threshold": (config.vh_water_threshold_db, "dB"),
        "baseline_water_margin": (config.baseline_water_margin_db, "dB"),
        "new_water_vv_change": (config.new_water_delta_db, "dB"),
        "new_water_max_slope": (config.new_water_max_slope_deg, "degrees"),
        "disturbed_surface_change": (config.disturbed_change_db, "dB"),
        "disturbed_surface_z_minimum": (config.disturbed_z_minimum, "standard deviations"),
        "disturbed_surface_max_slope": (config.disturbed_max_slope_deg, "degrees"),
        "max_hand": (max_hand_m, "m"),
        "minimum_component_area": (config.minimum_component_area_ha, "ha"),
        "stream_or_water_proximity": (config.proximity_distance_m, "m"),
        "cleanup_structuring_element_size": (
            float(config.structuring_element_size),
            "pixels",
        ),
        "confidence_z_saturation": (config.z_saturation, "standard deviations"),
        "confidence_invalid_distance_saturation": (
            config.invalid_distance_saturation_m,
            "m",
        ),
        "confidence_z_weight": (config.z_weight, "fraction"),
        "confidence_agreement_weight": (config.agreement_weight, "fraction"),
        "confidence_invalid_distance_weight": (
            config.invalid_distance_weight,
            "fraction",
        ),
        "confidence_slope_weight": (config.slope_weight, "fraction"),
    }


def _measurement(
    value: float | int | None,
    unit: str,
    input_files: list[str],
    scene_dates: dict[str, str],
    *,
    thresholds: dict[str, tuple[float, str]] | None = None,
) -> dict[str, object]:
    return {
        "value": value,
        "unit": unit,
        "module": "sahayak.flood.run",
        "input_files": input_files,
        "scene_dates": scene_dates,
        "thresholds_used": {
            key: {"value": threshold, "unit": threshold_unit, "module": "sahayak.flood.config"}
            for key, (threshold, threshold_unit) in (thresholds or {}).items()
        },
    }


def _fraction_or_none(numerator: int, denominator: int) -> float | None:
    """Return a fraction only when its denominator is non-zero."""
    return numerator / denominator if denominator else None


def _write_raster(
    path: Path,
    data: np.ndarray,
    base_profile: dict[str, Any],
    *,
    nodata: int | float,
    dtype: str | None = None,
) -> None:
    profile = base_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype or str(data.dtype),
        nodata=nodata,
        compress="deflate",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.where(np.isfinite(data), data, nodata)
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(output.astype(profile["dtype"]), 1)


def _write_polygons(
    path: Path,
    result: FloodDetection,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
) -> int:
    records: list[dict[str, object]] = []
    for class_code, class_name in CLASS_NAMES.items():
        mask = result.classes == class_code
        if not np.any(mask):
            continue
        geometries = [
            shape(geometry)
            for geometry, value in shapes(
                result.classes,
                mask=mask,
                transform=transform,
            )
            if value == class_code
        ]
        if geometries:
            geometry = unary_union(geometries)
            pixel_confidence = result.confidence[mask]
            records.append(
                {
                    "class_code": class_code,
                    "class_name": class_name,
                    "area_ha": float(geometry.area / 10_000),
                    "mean_confidence": float(np.nanmean(pixel_confidence)),
                    "geometry": geometry,
                }
            )
    polygons = gpd.GeoDataFrame(
        records,
        columns=["class_code", "class_name", "area_ha", "mean_confidence", "geometry"],
        geometry="geometry",
        crs=crs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    polygons.to_file(path, driver="GPKG", index=False)
    return len(records)


def _distance_to_stream(stream_mask: np.ndarray, pixel_size_m: float) -> np.ndarray:
    if not np.any(stream_mask):
        return np.full(stream_mask.shape, np.inf, dtype=np.float32)
    return distance_transform_edt(
        ~stream_mask,
        sampling=(pixel_size_m, pixel_size_m),
    ).astype(np.float32)


def _write_quicklooks(
    directory: Path,
    post_scene: RadarScene,
    result: FloodDetection,
    hand: np.ndarray,
    transform: rasterio.Affine,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    bounds = (
        transform.c,
        transform.c + transform.a * result.classes.shape[1],
        transform.f + transform.e * result.classes.shape[0],
        transform.f,
    )
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.imshow(post_scene.vv_db, cmap="gray", vmin=-30, vmax=5, extent=bounds)
    colors = ListedColormap(["#1877d2", "#f28e2b", "#36c5d2"])
    overlay = np.ma.masked_where(
        (result.classes == CLASS_NONE) | (result.classes == CLASS_INVALID),
        result.classes,
    )
    axis.imshow(
        overlay,
        cmap=colors,
        norm=BoundaryNorm((0.5, 1.5, 2.5, 3.5), colors.N),
        alpha=0.4,
        extent=bounds,
    )
    axis.set_title("Flood and disturbed-surface screening")
    _add_scale_bar(axis, result.classes.shape[1], transform.a)
    axis.text(
        0.01,
        0.015,
        FLOOD_ATTRIBUTION,
        transform=axis.transAxes,
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    figure.tight_layout()
    figure.savefig(directory / "flood_overlay.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    change_image = np.ma.masked_invalid(result.change_vv_db)
    image = axis.imshow(change_image, cmap="RdBu_r", vmin=-10, vmax=10, extent=bounds)
    axis.set_title("VV change (post-event minus pre-event baseline)")
    figure.colorbar(image, ax=axis, label="dB")
    _add_scale_bar(axis, result.classes.shape[1], transform.a)
    figure.tight_layout()
    figure.savefig(directory / "change_vv.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 8))
    hand_image = np.ma.masked_invalid(hand)
    image = axis.imshow(hand_image, cmap="terrain", vmin=0, vmax=200, extent=bounds)
    axis.set_title("Height above nearest downstream drainage")
    figure.colorbar(image, ax=axis, label="m")
    _add_scale_bar(axis, result.classes.shape[1], transform.a)
    figure.tight_layout()
    figure.savefig(directory / "terrain_hand.png", dpi=150)
    plt.close(figure)


def _add_scale_bar(axis: Any, width_pixels: int, pixel_size_m: float) -> None:
    width_m = width_pixels * pixel_size_m
    bar_m = min(2_000.0, width_m * 0.2)
    start_x = 0.05
    start_y = 0.06
    axis.plot(
        [start_x, start_x + bar_m / width_m],
        [start_y, start_y],
        color="white",
        linewidth=3,
        transform=axis.transAxes,
    )
    axis.text(
        start_x,
        start_y + 0.015,
        f"{bar_m / 1_000:g} km",
        color="white",
        fontsize=8,
        transform=axis.transAxes,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )


def _merge_facts(path: Path, updates: dict[str, object]) -> None:
    facts: dict[str, object] = {}
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise ValueError(f"Existing facts file {path} must contain a JSON object.")
        facts = existing
    facts.update(updates)
    _write_json(path, facts)


def _write_json(path: Path, content: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _record_outputs(
    manifest: RunManifest,
    paths: list[Path],
    run_dir: Path,
    settings: Settings,
) -> None:
    names = {path.relative_to(run_dir).as_posix() for path in paths}
    manifest.inputs = [
        record
        for record in manifest.inputs
        if not (record.role == "output" and record.name in names)
    ]
    for path in paths:
        manifest.add_input(
            name=path.relative_to(run_dir).as_posix(),
            role="output",
            source="Sahayak flood mapping",
            license="Generated analytical output",
            retrieved_utc=datetime.now(UTC),
            path=path,
            settings=settings,
        )
