"""Sentinel-1 validity masking, masked Lee filtering, and dB conversion."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

from sahayak.config import Settings
from sahayak.flood.config import FloodConfig
from sahayak.provenance import require_input


@dataclass(frozen=True)
class RadarScene:
    """Filtered VV/VH dB arrays and their common per-pixel validity mask."""

    vv_db: np.ndarray
    vh_db: np.ndarray
    valid_mask: np.ndarray
    descriptions: tuple[str, ...]


def preprocess_scene(
    path: Path,
    config: FloodConfig,
    settings: Settings,
) -> RadarScene:
    """Read named Sentinel-1 bands, mask artifacts, Lee-filter linear power, and convert to dB."""
    source_path = require_input(path, settings)
    with rasterio.open(source_path) as source:
        descriptions = tuple(source.descriptions)
        indexes = {
            "vv": _find_band(descriptions, "vv"),
            "vh": _find_band(descriptions, "vh"),
            "incidence": _find_band(descriptions, "incidence angle"),
            "shadow": _find_band(descriptions, "shadow mask"),
            "data": _find_band(descriptions, "data mask"),
        }
        vv = source.read(indexes["vv"]).astype(np.float32, copy=False)
        vh = source.read(indexes["vh"]).astype(np.float32, copy=False)
        incidence = source.read(indexes["incidence"]).astype(np.float32, copy=False)
        shadow = source.read(indexes["shadow"]).astype(np.float32, copy=False)
        data_mask = source.read(indexes["data"]).astype(np.float32, copy=False)
        nodata = source.nodata

    valid = (
        np.isfinite(vv)
        & np.isfinite(vh)
        & np.isfinite(incidence)
        & np.isfinite(shadow)
        & np.isfinite(data_mask)
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
    if not np.any(valid):
        raise ValueError(
            f"No valid Sentinel-1 pixels remain after preprocessing {source_path.name}."
        )

    filtered_vv = lee_filter(vv, valid, config.lee_window_size)
    filtered_vh = lee_filter(vh, valid, config.lee_window_size)
    vv_db = np.full(vv.shape, np.nan, dtype=np.float32)
    vh_db = np.full(vh.shape, np.nan, dtype=np.float32)
    vv_db[valid] = 10.0 * np.log10(filtered_vv[valid])
    vh_db[valid] = 10.0 * np.log10(filtered_vh[valid])
    return RadarScene(vv_db, vh_db, valid, descriptions)


def lee_filter(image: np.ndarray, valid_mask: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply a masked Lee filter to linear power without mixing invalid samples.

    The local mean and variance use only valid samples. The Lee gain is
    max((local_variance - noise_variance) / local_variance, 0), where the
    global noise estimate is the 20th percentile of valid local variances.
    """
    if image.shape != valid_mask.shape:
        raise ValueError("Image and validity mask must have matching shapes.")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("Lee window size must be an odd integer of at least 3.")
    valid = valid_mask & np.isfinite(image) & (image > 0)
    if not np.any(valid):
        raise ValueError("Lee filter received no valid positive finite samples.")

    weights = valid.astype(np.float32)
    safe_image = np.where(valid, image, 0.0).astype(np.float32, copy=False)
    count = uniform_filter(weights, size=window_size, mode="constant", cval=0.0)
    sum_values = uniform_filter(safe_image, size=window_size, mode="constant", cval=0.0)
    sum_squares = uniform_filter(
        safe_image * safe_image,
        size=window_size,
        mode="constant",
        cval=0.0,
    )
    local_mean = np.divide(sum_values, count, out=np.zeros_like(sum_values), where=count > 0)
    local_mean_square = np.divide(
        sum_squares,
        count,
        out=np.zeros_like(sum_squares),
        where=count > 0,
    )
    local_variance = np.maximum(local_mean_square - local_mean * local_mean, 0.0)

    valid_variances = local_variance[valid]
    noise_variance = float(np.percentile(valid_variances, 20))
    gain = np.divide(
        np.maximum(local_variance - noise_variance, 0.0),
        local_variance,
        out=np.zeros_like(local_variance),
        where=local_variance > np.finfo(np.float32).eps,
    )
    result = np.full(image.shape, np.nan, dtype=np.float32)
    result[valid] = local_mean[valid] + gain[valid] * (image[valid] - local_mean[valid])
    return result


def _find_band(descriptions: tuple[str | None, ...], label: str) -> int:
    normalized = [description.casefold() if description else "" for description in descriptions]
    if label in {"vv", "vh"}:
        pattern = re.compile(rf"\b{label}\b", re.IGNORECASE)
        candidates = [
            index
            for index, description in enumerate(normalized, start=1)
            if pattern.search(description)
        ]
    elif label == "incidence angle":
        candidates = [
            index
            for index, description in enumerate(normalized, start=1)
            if "incidence" in description and "angle" in description
        ]
    else:
        candidates = [
            index
            for index, description in enumerate(normalized, start=1)
            if label in description
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one band description for {label!r}, found {len(candidates)}."
        )
    return candidates[0]
