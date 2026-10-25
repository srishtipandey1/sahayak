"""Diagnose Sentinel-2 invalid pixels in the optical corridor."""

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, label

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id
from sahayak.optical.preprocess import BANDS
from sahayak.optical.run import (
    ATTRIBUTION,
    _find_settlement,
    _read_aligned_single_band,
    _read_s2_scene,
    _truecolor,
)
from sahayak.quicklook import write_png_atomically


def diagnose_optical(
    *,
    run_id: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Write diagnostic statistics and an invalid-pixel quicklook for one run."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    selection_path = run_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"Run {run_id!r} lacks selection.json.")
    selected = json.loads(selection_path.read_text(encoding="utf-8")).get("s2", {}).get(
        "selected", {}
    )
    if not all(name in selected for name in ("pre", "post")):
        raise ValueError("Run selection.json must contain selected S2 pre and post scenes.")
    scenes = {
        name: _read_s2_scene(run_dir / selected[name]["file"])
        for name in ("pre", "post")
    }
    profile = scenes["pre"]["profile"]
    shape = scenes["pre"]["arrays"]["SCL"].shape
    if any(scene["arrays"]["SCL"].shape != shape for scene in scenes.values()):
        raise ValueError("Sentinel-2 diagnostic scenes are not aligned to a common grid.")
    hand_path = run_dir / "flood" / "hand.tif"
    slope_path = run_dir / "slope.tif"
    dem_path = run_dir / "dem.tif"
    stream_path = run_dir / "flood" / "streams.gpkg"
    main_river_path = run_dir / "flood" / "main_river.tif"
    for path in (hand_path, slope_path, dem_path, stream_path, main_river_path):
        if not path.is_file():
            raise FileNotFoundError(f"Optical diagnostic input is missing: {path.name}")
    hand = _read_aligned_single_band(hand_path, profile, *shape)
    slope = _read_aligned_single_band(slope_path, profile, *shape)
    elevation = _read_aligned_single_band(dem_path, profile, *shape)
    main_river_values = _read_aligned_single_band(main_river_path, profile, *shape)
    main_river = np.isfinite(main_river_values) & (main_river_values > 0)

    streams = gpd.read_file(stream_path, layer="streams")
    if streams.crs is None:
        raise ValueError("Flood stream layer has no CRS.")
    streams = streams.to_crs(profile["crs"])
    stream_mask = rasterize(
        ((geometry, 1) for geometry in streams.geometry if geometry is not None),
        out_shape=shape,
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)
    pixel_size = abs(profile["transform"].a)
    distance_to_stream = distance_transform_edt(
        ~stream_mask,
        sampling=(pixel_size, pixel_size),
    )
    corridor = np.isfinite(hand) & (hand <= 40.0)
    zones = {
        "corridor_hand_le_40m": corridor,
        "within_100m_of_streams": distance_to_stream <= 100.0,
        "main_river_cells": main_river,
    }
    histograms = {
        scene_name: {
            zone_name: _scl_histogram(scene["arrays"]["SCL"], zone)
            for zone_name, zone in zones.items()
        }
        for scene_name, scene in scenes.items()
    }

    pre_arrays = scenes["pre"]["arrays"]
    post_arrays = scenes["post"]["arrays"]
    stable = (
        np.isin(pre_arrays["SCL"], (4, 5, 6, 7))
        & np.isin(post_arrays["SCL"], (4, 5, 6, 7))
        & (pre_arrays["dataMask"] > 0)
        & (post_arrays["dataMask"] > 0)
        & np.isfinite(hand)
        & (hand > 100.0)
        & np.isfinite(slope)
        & (slope <= 35.0)
    )
    for band in BANDS:
        stable &= _valid_reflectance(pre_arrays[band]) & _valid_reflectance(post_arrays[band])
    if not stable.any():
        raise ValueError("No stable clear pixels are available for B04 diagnostic normalization.")
    b04_change = post_arrays["B04"] - pre_arrays["B04"]
    samples = b04_change[stable]
    low, high = np.quantile(samples, [0.02, 0.98])
    fit = stable & (b04_change >= low) & (b04_change <= high)
    positive = fit & (post_arrays["B04"] > 1e-6) & (pre_arrays["B04"] > 1e-6)
    scale = (
        float(np.median(pre_arrays["B04"][positive] / post_arrays["B04"][positive]))
        if positive.any()
        else 1.0
    )
    offset = float(np.median(pre_arrays["B04"][fit] - scale * post_arrays["B04"][fit]))
    normalized_b04 = post_arrays["B04"] * scale + offset
    delta_b04 = normalized_b04 - pre_arrays["B04"]
    bright_candidate = corridor & np.isfinite(delta_b04) & (delta_b04 >= 0.08)
    candidate_histogram = _scl_histogram(post_arrays["SCL"], bright_candidate)

    pre_valid = _diagnostic_valid(pre_arrays)
    post_valid = _diagnostic_valid(post_arrays)
    post_invalid_pre_valid = corridor & ~post_valid & pre_valid
    invalid_share = (
        float(post_invalid_pre_valid.sum() / corridor.sum()) if corridor.any() else None
    )

    suspicious_scl = np.isin(post_arrays["SCL"], (8, 9, 10, 11))
    corridor_suspicious = corridor & suspicious_scl
    genuine_cloud = (
        suspicious_scl
        & np.isfinite(hand)
        & (hand > 100.0)
        & np.isfinite(elevation)
        & (elevation < 3500.0)
    )
    band_means = {
        band: {
            "corridor_scl_8_11_mean": _finite_mean(post_arrays[band], corridor_suspicious),
            "hand_gt_100m_elevation_lt_3500m_scl_8_11_mean": _finite_mean(
                post_arrays[band], genuine_cloud
            ),
        }
        for band in BANDS
    }
    scl11 = (post_arrays["SCL"] == 11) & np.isfinite(hand) & np.isfinite(elevation)
    elevation_hand = {
        "count": int(scl11.sum()),
        "elevation_m": _distribution(elevation[scl11]),
        "hand_m": _distribution(hand[scl11]),
    }
    reflectance_present = post_arrays["dataMask"] > 0
    for band in BANDS:
        reflectance_present &= np.isfinite(post_arrays[band])
    near_stream_suspicious = suspicious_scl & (distance_to_stream <= 150.0) & reflectance_present
    components = _component_statistics(
        near_stream_suspicious,
        pixel_size * pixel_size / 10_000.0,
    )

    output_dir = run_dir / "optical"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnosis = {
        "module": "sahayak.optical.diagnose",
        "scene_dates": {
            name: selected[name]["acquired_at"] for name in ("pre", "post")
        },
        "zone_definitions": {
            "corridor_hand_le_40m": "Finite HAND <= 40 m.",
            "within_100m_of_streams": "Euclidean distance <= 100 m from rasterized stream cells.",
            "main_river_cells": "Positive cells in flood/main_river.tif.",
            "genuine_looking_cloud": "SCL 8-11, HAND > 100 m, elevation < 3500 m.",
            "near_stream_components": "Valid post SCL 8-11 cells within 150 m of streams.",
        },
        "scl_histograms": histograms,
        "bright_change_candidate": {
            "normalized_b04_scale": scale,
            "normalized_b04_offset": offset,
            "threshold_reflectance": 0.08,
            "hand_le_40m_scl_histogram": candidate_histogram,
        },
        "post_invalid_pre_valid_corridor_share": invalid_share,
        "post_scl_8_11_band_means": band_means,
        "post_scl_11_elevation_hand": elevation_hand,
        "post_scl_8_11_near_stream_components": components,
        "unavailable_reflectance_bands": ["B05", "B06", "B07", "B8A", "B09"],
        "attribution": ATTRIBUTION,
    }
    diagnosis_path = output_dir / "diagnosis.json"
    diagnosis_path.write_text(
        json.dumps(diagnosis, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    facts_path = run_dir / "facts.json"
    _merge_diagnosis_facts(facts_path, diagnosis)
    image_path = output_dir / "optical_invalid_diagnosis.png"
    if not write_png_atomically(
        image_path,
        lambda path: _write_diagnosis_figure(
            path,
            scenes["pre"],
            scenes["post"],
            post_valid,
            profile,
            run_dir / "osm_pre_event.gpkg",
        ),
    ):
        raise OSError("Could not atomically write optical_invalid_diagnosis.png.")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    return {
        "diagnosis": facts["optical"]["invalid_diagnosis"],
        "files": [diagnosis_path, image_path],
    }


def _merge_diagnosis_facts(path: Path, diagnosis: dict[str, Any]) -> None:
    """Add diagnostics without replacing other measurements in facts.json."""
    facts: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Facts file {path} must contain a JSON object.")
        facts = loaded
    optical = facts.setdefault("optical", {})
    if not isinstance(optical, dict):
        raise ValueError("The optical section of facts.json must be a JSON object.")
    optical["invalid_diagnosis"] = diagnosis
    path.write_text(json.dumps(facts, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _valid_reflectance(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values >= 0.0) & (values <= 1.5)


def _diagnostic_valid(arrays: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.isin(arrays["SCL"], (4, 5, 6, 7)) & (arrays["dataMask"] > 0)
    for band in BANDS:
        valid &= _valid_reflectance(arrays[band])
    return valid


def _scl_histogram(scl: np.ndarray, zone: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scl)[zone]
    counts = {str(int(code)): int((values == code).sum()) for code in range(12)}
    total = int(values.size)
    return {
        "pixel_count": total,
        "class_counts": counts,
        "class_fraction": {
            class_code: count / total if total else None for class_code, count in counts.items()
        },
    }


def _finite_mean(values: np.ndarray, mask: np.ndarray) -> dict[str, float | int | None]:
    samples = values[mask & np.isfinite(values)]
    return {
        "count": int(samples.size),
        "mean": float(samples.mean()) if samples.size else None,
    }


def _distribution(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {"min": None, "p10": None, "median": None, "p90": None, "max": None}
    quantiles = np.quantile(finite, [0.1, 0.5, 0.9])
    return {
        "min": float(finite.min()),
        "p10": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p90": float(quantiles[2]),
        "max": float(finite.max()),
    }


def _component_statistics(mask: np.ndarray, pixel_area_ha: float) -> dict[str, Any]:
    labels, count = label(mask)
    records = []
    for component_id in range(1, count + 1):
        rows, columns = np.nonzero(labels == component_id)
        if rows.size == 0:
            continue
        height = int(rows.max() - rows.min() + 1)
        width = int(columns.max() - columns.min() + 1)
        elongation = float(max(height, width) / min(height, width))
        records.append(
            {
                "pixels": int(rows.size),
                "area_ha": float(rows.size * pixel_area_ha),
                "elongation": elongation,
            }
        )
    return {"component_count": len(records), "components": records}


def _write_diagnosis_figure(
    path: Path,
    pre: dict[str, Any],
    post: dict[str, Any],
    post_valid: np.ndarray,
    profile: dict[str, Any],
    osm_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    settlements = gpd.read_file(osm_path, layer="settlements")
    if settlements.crs is None:
        raise ValueError("OSM settlement layer has no CRS.")
    settlements = settlements.to_crs(profile["crs"])
    center_x, center_y = _find_settlement(settlements)
    transform = profile["transform"]
    radius = int(1500 / abs(transform.a))
    center_row = int((transform.f - center_y) / abs(transform.e))
    center_column = int((center_x - transform.c) / transform.a)
    height, width = post_valid.shape
    r0, r1 = max(0, center_row - radius), min(height, center_row + radius)
    c0, c1 = max(0, center_column - radius), min(width, center_column + radius)
    extent = (
        transform.c + c0 * transform.a,
        transform.c + c1 * transform.a,
        transform.f + r1 * transform.e,
        transform.f + r0 * transform.e,
    )
    post_arrays = post["arrays"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(
        _truecolor(post_arrays, pre["arrays"])[r0:r1, c0:c1],
        extent=extent,
    )
    axes[0].set_title("Post true colour")
    axes[1].imshow(
        post_arrays["SCL"][r0:r1, c0:c1],
        extent=extent,
        vmin=0,
        vmax=11,
        cmap="tab20",
    )
    axes[1].set_title("Post SCL class")
    axes[2].imshow(
        (~post_valid[r0:r1, c0:c1]).astype(np.uint8),
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="gray_r",
    )
    axes[2].set_title("Post invalid mask")
    axes[0].text(
        0.01,
        0.01,
        ATTRIBUTION,
        transform=axes[0].transAxes,
        color="white",
        fontsize=8,
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    for axis in axes:
        axis.set_xlabel(str(profile["crs"]))
        axis.set_ylabel("Northing")
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)
