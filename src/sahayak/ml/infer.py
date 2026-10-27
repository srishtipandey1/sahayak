"""Inference for the optional Sen1Floods11 Sentinel-1 flood model."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy.ndimage import label
from shapely.geometry import shape
from skimage.filters import threshold_otsu

from sahayak.config import Settings
from sahayak.flood.config import FloodConfig, load_flood_config
from sahayak.flood.detect import _remove_small_components
from sahayak.flood.preprocess import _find_band, radar_valid_mask
from sahayak.ingest.dem import DEM_ATTRIBUTION
from sahayak.ingest.run import validate_run_id
from sahayak.ingest.s1 import S1_ATTRIBUTION
from sahayak.provenance import RunManifest, require_input, sha256_file

TILE_SIZE = 256
TILE_OVERLAP = 64
MINIMUM_COMPONENT_AREA_HA = 0.5
MAXIMUM_HAND_M = 40.0
MAXIMUM_SLOPE_DEG = 30.0
CLASS_INVALID = 255
PROBABILITY_NODATA = -9999.0
TRANSFER_STATEMENT = (
    "The model was trained on Sen1Floods11 sigma0 open-water flood labels. "
    "This event input is gamma0 terrain backscatter from a debris-flow event; "
    "transfer to this sensor product and hazard type is unvalidated."
)


@dataclass(frozen=True)
class RadarInput:
    """Unfiltered scene dB values, the shared flood validity mask, and raster profile."""

    vv_db: np.ndarray
    vh_db: np.ndarray
    valid_mask: np.ndarray
    profile: dict[str, Any]
    descriptions: tuple[str | None, ...]


def build_model_features(
    vv_db: np.ndarray,
    vh_db: np.ndarray,
    valid_mask: np.ndarray,
    *,
    means_db: tuple[float, float],
    standard_deviations_db: tuple[float, float],
    clip_db: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the training clip, per-band normalization, and normalized difference input."""
    if vv_db.shape != vh_db.shape or vv_db.shape != valid_mask.shape:
        raise ValueError("VV, VH, and validity arrays must share one grid.")
    if clip_db[0] >= clip_db[1]:
        raise ValueError("The model dB clipping interval must be increasing.")
    if any(not math.isfinite(value) for value in (*means_db, *standard_deviations_db)):
        raise ValueError("Model normalization values must be finite.")
    if any(value <= 0 for value in standard_deviations_db):
        raise ValueError("Model standard deviations must be positive.")
    if not valid_mask.any():
        raise ValueError("Cannot build model features without valid radar pixels.")

    vv_clipped = np.clip(vv_db, *clip_db).astype(np.float32, copy=False)
    vh_clipped = np.clip(vh_db, *clip_db).astype(np.float32, copy=False)
    normalized_vv = (vv_clipped - means_db[0]) / standard_deviations_db[0]
    normalized_vh = (vh_clipped - means_db[1]) / standard_deviations_db[1]
    features = np.stack(
        (normalized_vv, normalized_vh, normalized_vv - normalized_vh),
    ).astype(np.float32, copy=False)
    features[:, ~valid_mask] = 0.0
    return features, vv_clipped, vh_clipped


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= 0:
        raise ValueError("Raster dimensions must be positive.")
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("Tile size must be positive and overlap must be smaller than tile size.")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_tiled(
    model: Any,
    features: np.ndarray,
    *,
    device: Any = "cpu",
    tile_size: int = TILE_SIZE,
    overlap: int = TILE_OVERLAP,
) -> np.ndarray:
    """Predict probabilities by cosine blending overlapping fixed-size model tiles."""
    import torch

    if features.ndim != 3 or features.shape[0] != 3:
        raise ValueError("Model features must have shape (3, height, width).")
    height, width = features.shape[1:]
    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)
    window_1d = np.hanning(tile_size + 2)[1:-1].astype(np.float32)
    weights = np.maximum(np.outer(window_1d, window_1d), np.finfo(np.float32).eps)
    accumulated = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    model.eval()

    with torch.inference_mode():
        for top in y_starts:
            for left in x_starts:
                tile = features[
                    :,
                    top : min(top + tile_size, height),
                    left : min(left + tile_size, width),
                ]
                tile_height, tile_width = tile.shape[1:]
                if tile_height < tile_size or tile_width < tile_size:
                    pad_mode = "reflect" if min(tile_height, tile_width) > 1 else "edge"
                    tile = np.pad(
                        tile,
                        (
                            (0, 0),
                            (0, tile_size - tile_height),
                            (0, tile_size - tile_width),
                        ),
                        mode=pad_mode,
                    )
                tensor = torch.from_numpy(np.ascontiguousarray(tile[None])).to(
                    device=device,
                    dtype=torch.float32,
                )
                logits = model(tensor)
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 4
                    or logits.shape[0] != 1
                    or logits.shape[1] != 1
                    or logits.shape[2:] != (tile_size, tile_size)
                ):
                    raise ValueError("Flood model must return one logit per pixel and tile.")
                probability = torch.sigmoid(logits[0, 0]).cpu().numpy()
                tile_weights = weights[:tile_height, :tile_width]
                y_slice = slice(top, top + tile_height)
                x_slice = slice(left, left + tile_width)
                accumulated[y_slice, x_slice] += probability[:tile_height, :tile_width] * (
                    tile_weights
                )
                weight_sum[y_slice, x_slice] += tile_weights

    if np.any(weight_sum <= 0):
        raise RuntimeError("Cosine tile blending left one or more pixels without a prediction.")
    return accumulated / weight_sum


def threshold_mask(
    probability: np.ndarray,
    valid_mask: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Threshold a probability raster while excluding invalid pixels."""
    if probability.shape != valid_mask.shape:
        raise ValueError("Probability and validity arrays must share one grid.")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Probability threshold must be between zero and one.")
    return (probability >= threshold) & valid_mask & np.isfinite(probability)


def apply_terrain_mask(
    prediction: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    *,
    maximum_hand_m: float = MAXIMUM_HAND_M,
    maximum_slope_deg: float = MAXIMUM_SLOPE_DEG,
) -> np.ndarray:
    """Restrict a prediction to finite pixels within the requested terrain limits."""
    if prediction.shape != hand.shape or prediction.shape != slope.shape:
        raise ValueError("Prediction, HAND, and slope arrays must share one grid.")
    return prediction & terrain_valid_mask(
        hand,
        slope,
        maximum_hand_m=maximum_hand_m,
        maximum_slope_deg=maximum_slope_deg,
    )


def terrain_valid_mask(
    hand: np.ndarray,
    slope: np.ndarray,
    *,
    maximum_hand_m: float = MAXIMUM_HAND_M,
    maximum_slope_deg: float = MAXIMUM_SLOPE_DEG,
) -> np.ndarray:
    """Return pixels eligible under finite HAND, HAND-height, and slope limits."""
    if hand.shape != slope.shape:
        raise ValueError("HAND and slope arrays must share one grid.")
    if maximum_hand_m <= 0 or maximum_slope_deg <= 0:
        raise ValueError("Terrain limits must be positive.")
    return (
        np.isfinite(hand)
        & np.isfinite(slope)
        & (hand <= maximum_hand_m)
        & (slope <= maximum_slope_deg)
    )


def remove_small_components(
    prediction: np.ndarray,
    *,
    pixel_area_m2: float,
    minimum_area_ha: float = MINIMUM_COMPONENT_AREA_HA,
) -> np.ndarray:
    """Apply the flood module's eight-connected component-size filter."""
    if pixel_area_m2 <= 0 or minimum_area_ha <= 0:
        raise ValueError("Pixel and minimum component areas must be positive.")
    pixel_size_m = math.sqrt(pixel_area_m2)
    return _remove_small_components(prediction, minimum_area_ha, pixel_size_m)


def pixel_area_m2_from_transform(transform: rasterio.Affine) -> float:
    """Return one raster cell's area from the affine determinant in square metres."""
    pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("Raster transform must have a finite, positive pixel area.")
    return pixel_area_m2


def overlap_summary(
    prediction: np.ndarray,
    reference: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pixel_area_m2: float,
) -> dict[str, int | float | None]:
    """Report area overlap and both directional fractions on shared valid pixels."""
    if prediction.shape != reference.shape or prediction.shape != valid_mask.shape:
        raise ValueError("Prediction, reference, and validity arrays must share one grid.")
    if pixel_area_m2 <= 0:
        raise ValueError("Pixel area must be positive.")
    predicted = prediction & valid_mask
    reference_valid = reference & valid_mask
    predicted_count = int(predicted.sum())
    reference_count = int(reference_valid.sum())
    intersection_count = int((predicted & reference_valid).sum())
    hectares_per_pixel = pixel_area_m2 / 10_000
    return {
        "prediction_pixels": predicted_count,
        "reference_pixels": reference_count,
        "intersection_pixels": intersection_count,
        "prediction_area_ha": predicted_count * hectares_per_pixel,
        "reference_area_ha": reference_count * hectares_per_pixel,
        "intersection_area_ha": intersection_count * hectares_per_pixel,
        "prediction_inside_reference_percent": (
            100.0 * intersection_count / predicted_count if predicted_count else None
        ),
        "reference_inside_prediction_percent": (
            100.0 * intersection_count / reference_count if reference_count else None
        ),
    }


def false_alarm_accounting(
    prediction: np.ndarray,
    valid_mask: np.ndarray,
    *,
    pixel_area_m2: float,
) -> dict[str, int | float]:
    """Measure placebo detections and false alarms per 1,000 valid radar pixels."""
    if prediction.shape != valid_mask.shape or pixel_area_m2 <= 0:
        raise ValueError("Placebo arrays must align and pixel area must be positive.")
    valid_count = int(valid_mask.sum())
    detections = prediction & valid_mask
    detection_count = int(detections.sum())
    return {
        "detection_pixels": detection_count,
        "valid_pixels": valid_count,
        "area_ha": detection_count * pixel_area_m2 / 10_000,
        "false_alarms_per_1000_valid_pixels": (
            1000.0 * detection_count / valid_count if valid_count else 0.0
        ),
    }


def _read_model_metadata(config_path: Path, metrics_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if config.get("dataset") != "sen1floods11" or metrics.get("dataset") != "sen1floods11":
        raise ValueError("Inference requires a Sen1Floods11 model configuration and metrics.")
    if config.get("synthetic") or metrics.get("synthetic"):
        raise ValueError("Synthetic training artifacts cannot be used for event inference.")
    if not config.get("final_metrics") or not metrics.get("final_metrics"):
        raise ValueError("Inference requires finalized, non-synthetic model metrics.")
    hyperparameters = config.get("hyperparameters", {})
    if hyperparameters.get("input_channels") != ["VV", "VH", "VV_minus_VH"]:
        raise ValueError("Model input channels differ from the supported VV/VH feature contract.")
    normalization = config.get("normalization", {})
    means = (normalization.get("VV_mean_db"), normalization.get("VH_mean_db"))
    standard_deviations = (
        normalization.get("VV_std_db"),
        normalization.get("VH_std_db"),
    )
    clip = normalization.get("clip_db")
    if (
        any(not isinstance(value, (int, float)) for value in (*means, *standard_deviations))
        or not isinstance(clip, list)
        or len(clip) != 2
        or any(not isinstance(value, (int, float)) for value in clip)
    ):
        raise ValueError("Model config.json has incomplete dB normalization values.")
    threshold = metrics.get("validation_threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ValueError("Model metrics.json has no finite validation-selected threshold.")
    if not 0 <= threshold <= 1:
        raise ValueError("Validation-selected model threshold must be between zero and one.")
    return {
        "config": config,
        "metrics": metrics,
        "training_distributions": config.get("training_distributions"),
        "means_db": tuple(float(value) for value in means),
        "standard_deviations_db": tuple(float(value) for value in standard_deviations),
        "clip_db": (float(clip[0]), float(clip[1])),
        "threshold": float(threshold),
    }


def _load_model(weights_path: Path, device: Any) -> Any:
    """Load this project's tensor checkpoint using PyTorch's restricted unpickler."""
    import torch

    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    safe_globals = [
        np.dtype,
        np.ndarray,
        numpy_core.multiarray._reconstruct,
    ]
    uint32_dtype = getattr(getattr(np, "dtypes", None), "UInt32DType", None)
    if uint32_dtype is not None:
        safe_globals.append(uint32_dtype)
    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )
    if not isinstance(checkpoint, dict):
        raise ValueError("Model checkpoint must contain a state dictionary.")
    state_dict = checkpoint.get("model_state", checkpoint)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Model checkpoint contains no model state dictionary.")
    from sahayak.ml.model import build_model

    model = build_model(encoder_weights=None)
    model.load_state_dict(state_dict)
    return model.to(device)


def _read_radar_input(
    path: Path,
    config: FloodConfig,
    settings: Settings,
    *,
    reference_profile: dict[str, Any] | None = None,
) -> RadarInput:
    source_path = require_input(path, settings)
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError(f"Sentinel-1 scene {source_path.name} has no CRS.")
        if reference_profile is not None and (
            source.shape != (reference_profile["height"], reference_profile["width"])
            or source.transform != reference_profile["transform"]
            or source.crs != reference_profile["crs"]
        ):
            raise ValueError(f"Sentinel-1 scene {source_path.name} is not aligned to s1_post.tif.")
        descriptions = tuple(source.descriptions)
        vv = source.read(_find_band(descriptions, "vv")).astype(np.float32, copy=False)
        vh = source.read(_find_band(descriptions, "vh")).astype(np.float32, copy=False)
        incidence = source.read(
            _find_band(descriptions, "incidence angle")
        ).astype(np.float32, copy=False)
        shadow = source.read(_find_band(descriptions, "shadow mask")).astype(
            np.float32,
            copy=False,
        )
        data_mask = source.read(_find_band(descriptions, "data mask")).astype(
            np.float32,
            copy=False,
        )
        nodata = source.nodata
        profile = source.profile.copy()
    valid = radar_valid_mask(
        vv,
        vh,
        incidence,
        shadow,
        data_mask,
        config,
        nodata=nodata,
    )
    if not valid.any():
        raise ValueError(f"No valid Sentinel-1 pixels remain in {source_path.name}.")
    vv_db = np.full(vv.shape, np.nan, dtype=np.float32)
    vh_db = np.full(vh.shape, np.nan, dtype=np.float32)
    vv_db[valid] = 10.0 * np.log10(vv[valid])
    vh_db[valid] = 10.0 * np.log10(vh[valid])
    return RadarInput(vv_db, vh_db, valid, profile, descriptions)


def _read_aligned_terrain(path: Path, reference_profile: dict[str, Any]) -> np.ndarray:
    with rasterio.open(path) as source:
        if (
            source.count != 1
            or source.shape != (reference_profile["height"], reference_profile["width"])
            or source.transform != reference_profile["transform"]
            or source.crs != reference_profile["crs"]
        ):
            raise ValueError(f"Terrain raster {path.name} is not aligned to s1_post.tif.")
        values = source.read(1).astype(np.float32)
        if source.nodata is not None:
            values[values == source.nodata] = np.nan
    return values


def _write_raster(
    path: Path,
    data: np.ndarray,
    base_profile: dict[str, Any],
    *,
    dtype: str,
    nodata: int | float,
) -> None:
    profile = base_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype.startswith("float") else 2,
    )
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(data.astype(dtype, copy=False), 1)


def _write_polygons(
    path: Path,
    prediction: np.ndarray,
    probability: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    pixel_area_m2: float,
) -> int:
    import geopandas as gpd

    labels, component_count = label(prediction, structure=np.ones((3, 3), dtype=bool))
    minimum_pixels = math.ceil(MINIMUM_COMPONENT_AREA_HA * 10_000 / pixel_area_m2)
    records: list[dict[str, Any]] = []
    component_sizes = np.bincount(labels.ravel(), minlength=component_count + 1)
    for geometry, component_id in shapes(
        labels,
        mask=labels > 0,
        transform=transform,
        connectivity=8,
    ):
        identifier = int(component_id)
        if component_sizes[identifier] < minimum_pixels:
            continue
        component_mask = labels == identifier
        records.append(
            {
                "component_id": identifier,
                "pixel_count": int(component_sizes[identifier]),
                "area_ha": float(component_sizes[identifier] * pixel_area_m2 / 10_000),
                "mean_probability": float(np.mean(probability[component_mask])),
                "geometry": shape(geometry),
            }
        )
    polygons = gpd.GeoDataFrame(
        records,
        columns=[
            "component_id",
            "pixel_count",
            "area_ha",
            "mean_probability",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )
    polygons.to_file(path, layer="ml_flood", driver="GPKG", index=False)
    return len(records)


def _read_true_colour(
    path: Path,
    reference_profile: dict[str, Any],
) -> np.ndarray:
    with rasterio.open(path) as source:
        if (
            source.shape != (reference_profile["height"], reference_profile["width"])
            or source.transform != reference_profile["transform"]
            or source.crs != reference_profile["crs"]
        ):
            raise ValueError("Post-event true-colour raster is not aligned to s1_post.tif.")
        descriptions = tuple(source.descriptions)
        band_indices = {
            description.split(maxsplit=1)[0]: index
            for index, description in enumerate(descriptions, start=1)
            if description
        }
        required = ("B04", "B03", "B02")
        if any(name not in band_indices for name in required):
            raise ValueError("Post-event true-colour raster lacks B04/B03/B02 bands.")
        rgb = np.stack(
            [source.read(band_indices[name]).astype(np.float32) for name in required],
            axis=-1,
        )
        valid = np.all(np.isfinite(rgb), axis=-1)
        if source.nodata is not None:
            valid &= np.all(rgb != source.nodata, axis=-1)
        data_mask_index = next(
            (
                index
                for index, description in enumerate(descriptions, start=1)
                if description and "data mask" in description.casefold()
            ),
            None,
        )
        if data_mask_index is not None:
            valid &= source.read(data_mask_index) > 0
    if not valid.any():
        raise ValueError("Post-event true-colour raster has no valid RGB pixels.")
    rendered = np.zeros(rgb.shape, dtype=np.float32)
    for band in range(3):
        samples = rgb[..., band][valid]
        low, high = np.percentile(samples, (2, 98))
        rendered[..., band] = (
            np.clip((rgb[..., band] - low) / (high - low), 0.0, 1.0)
            if high > low
            else 0.0
        )
    rendered[~valid] = 0.0
    return rendered


def _write_overlay(
    path: Path,
    run_dir: Path,
    post: RadarInput,
    prediction: np.ndarray,
    probability: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    true_colour = run_dir / "s2_post.tif"
    if true_colour.is_file():
        background = _read_true_colour(true_colour, post.profile)
        background_title = "Post-event Sentinel-2 true colour"
    else:
        vv = np.clip(np.nan_to_num(post.vv_db, nan=-30.0), -30.0, 5.0)
        background = plt.get_cmap("gray")((vv + 30.0) / 35.0)[..., :3]
        background_title = "Post-event Sentinel-1 VV backscatter (dB)"

    height, width = prediction.shape
    transform = post.profile["transform"]
    extent = (
        transform.c,
        transform.c + transform.a * width,
        transform.f + transform.e * height,
        transform.f,
    )
    overlay = np.zeros((height, width, 4), dtype=np.float32)
    overlay[prediction] = (0.90, 0.18, 0.12, 0.58)
    figure, axis = plt.subplots(figsize=(11, 9))
    axis.imshow(background, extent=extent, origin="upper")
    axis.imshow(overlay, extent=extent, origin="upper", interpolation="nearest")
    axis.set_title(f"Sen1Floods11 flood model — {background_title}")
    axis.set_xlabel("Easting (m)")
    axis.set_ylabel("Northing (m)")
    axis.legend(
        handles=[
            Patch(facecolor="#e62e1f", alpha=0.58, label="Model flood detection"),
        ],
        loc="lower right",
    )
    figure.text(
        0.01,
        0.01,
        f"{S1_ATTRIBUTION} {DEM_ATTRIBUTION} "
        "Sen1Floods11 (Bonafilia et al., CVPR Workshops 2020): open-water flood model; "
        "not validated for debris flow.",
        fontsize=7,
        wrap=True,
    )
    figure.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _merge_facts(path: Path, ml_facts: dict[str, Any]) -> None:
    facts = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(facts, dict):
        raise ValueError(f"Existing facts file {path} must contain a JSON object.")
    facts["ml"] = ml_facts
    path.write_text(
        json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _record_manifest(
    manifest_path: Path,
    run_id: str,
    project_root: Path,
    settings: Settings,
    inputs: list[tuple[Path, str, str]],
    outputs: list[Path],
) -> None:
    manifest = RunManifest.read(manifest_path)
    for path, source, license_name in inputs:
        name = (
            path.relative_to(project_root).as_posix()
            if path.is_relative_to(project_root)
            else str(path)
        )
        manifest.inputs = [
            record
            for record in manifest.inputs
            if not (record.name == name and record.role == "input")
        ]
        manifest.add_input(
            name=name,
            role="input",
            source=source,
            license=license_name,
            retrieved_utc=datetime.now(UTC),
            path=path,
            settings=settings,
        )
    for path in outputs:
        name = path.relative_to(settings.resolved_runs_dir / run_id).as_posix()
        manifest.inputs = [
            record
            for record in manifest.inputs
            if not (record.name == name and record.role == "output")
        ]
        manifest.add_input(
            name=name,
            role="output",
            source="Sahayak Sen1Floods11 model inference",
            license="Generated analytical output",
            retrieved_utc=datetime.now(UTC),
            path=path,
            settings=settings,
        )
    manifest.write(manifest_path)


def run_ml_flood(
    *,
    run_id: str,
    weights: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run the validated Sen1Floods11 model on post-event and pre-event placebo scenes."""
    import torch

    started = time.perf_counter()
    validate_run_id(run_id)
    active_settings = settings or Settings()
    project_root = active_settings.env_file_path.parent
    run_dir = active_settings.resolved_runs_dir / run_id
    manifest_path = run_dir / "manifest.json"
    selection_path = run_dir / "selection.json"
    facts_path = run_dir / "facts.json"
    for path in (manifest_path, selection_path, facts_path):
        if not path.is_file():
            raise FileNotFoundError(f"ML flood inference requires {path.name} in run {run_id!r}.")

    model_dir = project_root / "models" / "flood_unet_sen1floods11"
    config_path = require_input(model_dir / "config.json", active_settings)
    metrics_path = require_input(model_dir / "metrics.json", active_settings)
    weights_path = require_input(
        weights.expanduser().resolve()
        if weights is not None and weights.is_absolute()
        else project_root / weights
        if weights is not None
        else model_dir / "best.pt",
        active_settings,
    )
    for path in (config_path, metrics_path, weights_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required ML model artifact does not exist: {path}")
    metadata = _read_model_metadata(config_path, metrics_path)
    flood_config = load_flood_config(project_root / "configs" / "flood.toml")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    s1_selection = selection.get("s1")
    pair = s1_selection.get("pair") if isinstance(s1_selection, dict) else None
    post_record = pair.get("post") if isinstance(pair, dict) else None
    pre_record = pair.get("pre") if isinstance(pair, dict) else None
    if not isinstance(post_record, dict) or not isinstance(pre_record, dict):
        raise ValueError("selection.json must contain selected pre- and post-event S1 scenes.")
    if not isinstance(post_record.get("file"), str) or not isinstance(pre_record.get("file"), str):
        raise ValueError("Selected S1 scenes must include input filenames.")
    if (
        post_record.get("relative_orbit") != pre_record.get("relative_orbit")
        or post_record.get("direction") != pre_record.get("direction")
    ):
        raise ValueError(
            "Selected Sentinel-1 pair does not share relative orbit and pass direction."
        )

    post_path = require_input(run_dir / post_record["file"], active_settings)
    placebo_path = require_input(run_dir / pre_record["file"], active_settings)
    if post_path.name != "s1_post.tif" or placebo_path.name != "s1_pre.tif":
        raise ValueError("Selected Sentinel-1 pair does not match s1_post.tif and s1_pre.tif.")
    post = _read_radar_input(post_path, flood_config, active_settings)
    placebo = _read_radar_input(
        placebo_path,
        flood_config,
        active_settings,
        reference_profile=post.profile,
    )
    hand_path = require_input(run_dir / "flood" / "hand.tif", active_settings)
    slope_path = require_input(run_dir / "slope.tif", active_settings)
    hand = _read_aligned_terrain(hand_path, post.profile)
    slope = _read_aligned_terrain(slope_path, post.profile)
    terrain_valid = terrain_valid_mask(hand, slope)
    post_detection_valid = post.valid_mask & terrain_valid
    placebo_detection_valid = placebo.valid_mask & terrain_valid
    post_features, post_vv_clipped, post_vh_clipped = build_model_features(
        post.vv_db,
        post.vh_db,
        post.valid_mask,
        means_db=metadata["means_db"],
        standard_deviations_db=metadata["standard_deviations_db"],
        clip_db=metadata["clip_db"],
    )
    placebo_features, placebo_vv_clipped, placebo_vh_clipped = build_model_features(
        placebo.vv_db,
        placebo.vh_db,
        placebo.valid_mask,
        means_db=metadata["means_db"],
        standard_deviations_db=metadata["standard_deviations_db"],
        clip_db=metadata["clip_db"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(weights_path, device)
    post_started = time.perf_counter()
    post_probability = predict_tiled(model, post_features, device=device)
    post_inference_seconds = time.perf_counter() - post_started
    placebo_started = time.perf_counter()
    placebo_probability = predict_tiled(model, placebo_features, device=device)
    placebo_inference_seconds = time.perf_counter() - placebo_started

    threshold = metadata["threshold"]
    post_valid = post_detection_valid
    pixel_area_m2 = pixel_area_m2_from_transform(post.profile["transform"])
    if (
        post.profile["crs"] is None
        or not post.profile["crs"].is_projected
    ):
        raise ValueError(
            "ML flood area calculation requires a projected grid with positive pixel area."
        )
    post_unmasked = remove_small_components(
        threshold_mask(post_probability, post.valid_mask, threshold),
        pixel_area_m2=pixel_area_m2,
    )
    post_final = remove_small_components(
        apply_terrain_mask(
            threshold_mask(post_probability, post.valid_mask, threshold),
            hand,
            slope,
            maximum_hand_m=MAXIMUM_HAND_M,
            maximum_slope_deg=MAXIMUM_SLOPE_DEG,
        ),
        pixel_area_m2=pixel_area_m2,
    )
    placebo_final = remove_small_components(
        apply_terrain_mask(
            threshold_mask(placebo_probability, placebo.valid_mask, threshold),
            hand,
            slope,
            maximum_hand_m=MAXIMUM_HAND_M,
            maximum_slope_deg=MAXIMUM_SLOPE_DEG,
        ),
        pixel_area_m2=pixel_area_m2,
    )
    transform = post.profile["transform"]

    optical_path = require_input(
        run_dir / "optical" / "optical_class.tif",
        active_settings,
    )
    with rasterio.open(optical_path) as optical_source:
        if (
            optical_source.count != 1
            or optical_source.shape != post.valid_mask.shape
            or optical_source.transform != transform
            or optical_source.crs != post.profile["crs"]
        ):
            raise ValueError("Optical class raster is not aligned to the Sentinel-1 grid.")
        optical_class = optical_source.read(1)
        optical_nodata = optical_source.nodata
    optical_valid = np.ones(optical_class.shape, dtype=bool)
    if optical_nodata is not None:
        optical_valid &= optical_class != optical_nodata
    optical_footprint = np.isin(optical_class, (1, 2, 4))
    optical_comparison_valid = post_valid & optical_valid
    optical_overlap = overlap_summary(
        post_final,
        optical_footprint,
        optical_comparison_valid,
        pixel_area_m2=pixel_area_m2,
    )

    otsu_population = post.valid_mask & (slope < flood_config.disturbed_max_slope_deg)
    if not otsu_population.any():
        raise ValueError("No common valid low-slope pixels remain for the Otsu comparison.")
    otsu_threshold_db = float(threshold_otsu(post.vv_db[otsu_population]))
    otsu_mask = post.valid_mask & (post.vv_db < otsu_threshold_db)
    ml_unmasked_shared = remove_small_components(
        threshold_mask(post_probability, post.valid_mask, threshold),
        pixel_area_m2=pixel_area_m2,
    )
    otsu_mask = remove_small_components(otsu_mask, pixel_area_m2=pixel_area_m2)
    otsu_overlap = overlap_summary(
        ml_unmasked_shared,
        otsu_mask,
        post.valid_mask,
        pixel_area_m2=pixel_area_m2,
    )

    output_dir = run_dir / "ml"
    output_dir.mkdir(parents=True, exist_ok=True)
    probability_path = output_dir / "ml_flood_prob.tif"
    class_path = output_dir / "ml_flood_class.tif"
    polygons_path = output_dir / "ml_polygons.gpkg"
    overlay_path = output_dir / "ml_overlay.png"
    probability_output = np.full(post_probability.shape, PROBABILITY_NODATA, dtype=np.float32)
    probability_output[post.valid_mask] = post_probability[post.valid_mask]
    class_output = np.full(post.valid_mask.shape, CLASS_INVALID, dtype=np.uint8)
    class_output[post.valid_mask] = 0
    class_output[post_final] = 1
    _write_raster(
        probability_path,
        probability_output,
        post.profile,
        dtype="float32",
        nodata=PROBABILITY_NODATA,
    )
    _write_raster(
        class_path,
        class_output,
        post.profile,
        dtype="uint8",
        nodata=CLASS_INVALID,
    )
    polygon_count = _write_polygons(
        polygons_path,
        post_final,
        post_probability,
        transform,
        post.profile["crs"],
        pixel_area_m2,
    )
    _write_overlay(overlay_path, run_dir, post, post_final, post_probability)

    post_vv_mean = float(np.mean(post_vv_clipped[post.valid_mask]))
    post_vh_mean = float(np.mean(post_vh_clipped[post.valid_mask]))
    placebo_vv_mean = float(np.mean(placebo_vv_clipped[placebo.valid_mask]))
    placebo_vh_mean = float(np.mean(placebo_vh_clipped[placebo.valid_mask]))
    training_means = metadata["means_db"]
    band_shift = {
        "training_distribution_source": (
            "training-only means of clipped dB values in config.json; "
            "training_distributions is null"
        ),
        "scene_mean_basis": "model input dB after applying the recorded clip interval",
        "VV": {
            "training_mean_db": training_means[0],
            "post_event_mean_db": post_vv_mean,
            "post_minus_training_mean_db": post_vv_mean - training_means[0],
            "post_event_raw_mean_db": float(np.mean(post.vv_db[post.valid_mask])),
            "placebo_mean_db": placebo_vv_mean,
            "placebo_minus_training_mean_db": placebo_vv_mean - training_means[0],
            "placebo_raw_mean_db": float(np.mean(placebo.vv_db[placebo.valid_mask])),
        },
        "VH": {
            "training_mean_db": training_means[1],
            "post_event_mean_db": post_vh_mean,
            "post_minus_training_mean_db": post_vh_mean - training_means[1],
            "post_event_raw_mean_db": float(np.mean(post.vh_db[post.valid_mask])),
            "placebo_mean_db": placebo_vh_mean,
            "placebo_minus_training_mean_db": placebo_vh_mean - training_means[1],
            "placebo_raw_mean_db": float(np.mean(placebo.vh_db[placebo.valid_mask])),
        },
    }
    placebo_accounting = false_alarm_accounting(
        placebo_final,
        placebo_detection_valid,
        pixel_area_m2=pixel_area_m2,
    )
    output_paths = [probability_path, class_path, polygons_path, overlay_path]
    output_sizes = {path.name: path.stat().st_size for path in output_paths}
    facts = {
        "module": "sahayak.ml.infer",
        "provenance": {
            "post_event_scene": post_path.name,
            "post_event_datetime": post_record.get("datetime"),
            "placebo_scene": placebo_path.name,
            "placebo_datetime": pre_record.get("datetime"),
            "relative_orbit": post_record.get("relative_orbit"),
            "pass_direction": post_record.get("direction"),
            "model_weights": weights_path.relative_to(project_root).as_posix()
            if weights_path.is_relative_to(project_root)
            else str(weights_path),
            "model_weights_sha256": sha256_file(weights_path),
            "model_config": config_path.relative_to(project_root).as_posix(),
            "model_config_sha256": sha256_file(config_path),
            "model_metrics": metrics_path.relative_to(project_root).as_posix(),
            "model_metrics_sha256": sha256_file(metrics_path),
            "input_files": [
                post_path.name,
                placebo_path.name,
                hand_path.name,
                slope_path.name,
                optical_path.relative_to(run_dir).as_posix(),
            ],
        },
        "model_transfer_statement": TRANSFER_STATEMENT,
        "training_metrics": {
            "source": metrics_path.relative_to(project_root).as_posix(),
            "metrics": metadata["metrics"],
        },
        "threshold_used": {
            "value": threshold,
            "selection": "Pooled validation F1; copied from metrics.json",
            "source": metrics_path.relative_to(project_root).as_posix(),
        },
        "normalization": {
            "clip_db": list(metadata["clip_db"]),
            "VV_mean_db": training_means[0],
            "VH_mean_db": training_means[1],
            "VV_std_db": metadata["standard_deviations_db"][0],
            "VH_std_db": metadata["standard_deviations_db"][1],
            "input_channels": ["VV", "VH", "VV_minus_VH"],
            "training_distribution": {
                "available": metadata["training_distributions"] is not None,
                "source": config_path.relative_to(project_root).as_posix(),
                "recorded_means_db": {
                    "VV": training_means[0],
                    "VH": training_means[1],
                },
                "note": (
                    "Full training distributions are unavailable because "
                    "config.json training_distributions is null."
                    if metadata["training_distributions"] is None
                    else "Full training distributions are recorded in config.json."
                ),
            },
            "band_mean_shift_db": band_shift,
        },
        "validity": {
            "module": "sahayak.flood.preprocess.radar_valid_mask",
            "post_valid_pixel_count": int(post.valid_mask.sum()),
            "placebo_valid_pixel_count": int(placebo.valid_mask.sum()),
            "terrain_valid_pixel_count": int(post_detection_valid.sum()),
            "placebo_terrain_valid_pixel_count": int(placebo_detection_valid.sum()),
            "incidence_angle_range_degrees": [
                flood_config.minimum_incidence_angle_deg,
                flood_config.maximum_incidence_angle_deg,
            ],
            "data_mask_required": "greater than zero",
            "shadow_mask_required": "less than or equal to zero",
            "terrain_limits": {
                "max_hand_m": MAXIMUM_HAND_M,
                "max_slope_deg": MAXIMUM_SLOPE_DEG,
            },
        },
        "area_ha": {
            "without_terrain_mask": float(post_unmasked.sum() * pixel_area_m2 / 10_000),
            "with_terrain_mask": float(post_final.sum() * pixel_area_m2 / 10_000),
        },
        "placebo_area_ha": float(placebo_final.sum() * pixel_area_m2 / 10_000),
        "placebo": {
            "scene": placebo_path.name,
            "detection_pixels": placebo_accounting["detection_pixels"],
            "valid_pixels": placebo_accounting["valid_pixels"],
            "false_alarms_per_1000_valid_pixels": placebo_accounting[
                "false_alarms_per_1000_valid_pixels"
            ],
            "interpretation": "Pre-event predictions are false alarms by construction.",
        },
        "false_alarms_per_1000_valid_pixels": placebo_accounting[
            "false_alarms_per_1000_valid_pixels"
        ],
        "comparisons": {
            "optical_default": {
                **optical_overlap,
                "optical_classes": [1, 2, 4],
                "shared_valid_pixel_count": int(optical_comparison_valid.sum()),
                "ml_mask": "terrain-restricted, minimum-area-filtered detections",
            },
            "otsu_baseline": {
                **otsu_overlap,
                "threshold_db": otsu_threshold_db,
                "shared_valid_pixel_count": int(post.valid_mask.sum()),
                "ml_mask": "thresholded probability without terrain mask, component-filtered",
                "otsu_mask": "post-event VV dB below scene-specific Otsu threshold",
            },
        },
        "timings_seconds": {
            "post_event_inference": post_inference_seconds,
            "placebo_inference": placebo_inference_seconds,
            "total": time.perf_counter() - started,
        },
        "outputs": {
            path.name: {
                "path": path.relative_to(run_dir).as_posix(),
                "size_bytes": output_sizes[path.name],
            }
            for path in output_paths
        },
        "polygon_count": polygon_count,
        "attribution": f"{S1_ATTRIBUTION} {DEM_ATTRIBUTION}",
    }
    _merge_facts(facts_path, facts)
    manifest_inputs = [
        (
            weights_path,
            "Sen1Floods11-trained Sahayak U-Net checkpoint",
            "Project model artifact; Sen1Floods11 terms should be confirmed before redistribution.",
        ),
        (
            config_path,
            "Sen1Floods11 Sahayak model configuration",
            "Project training metadata",
        ),
        (
            metrics_path,
            "Sen1Floods11 Sahayak model evaluation metrics",
            "Project training evaluation output",
        ),
    ]
    _record_manifest(
        manifest_path,
        run_id,
        project_root,
        active_settings,
        manifest_inputs,
        [*output_paths, facts_path],
    )
    return {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "facts": facts,
        "outputs": output_paths,
        "timings_seconds": facts["timings_seconds"],
        "output_sizes": output_sizes,
    }
