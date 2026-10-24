"""Optical band validation, indices, and robust radiometric normalization."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from sahayak.optical.config import OpticalConfig

BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")


@dataclass(frozen=True)
class OpticalPair:
    """Prepared pre/post reflectance arrays, validity, and normalization facts."""

    pre: dict[str, np.ndarray]
    post: dict[str, np.ndarray]
    valid: np.ndarray
    stable: np.ndarray
    coefficients: dict[str, dict[str, float | int]]


def prepare_pair(
    pre_data: dict[str, np.ndarray],
    post_data: dict[str, np.ndarray],
    pre_scl: np.ndarray,
    post_scl: np.ndarray,
    pre_data_mask: np.ndarray,
    post_data_mask: np.ndarray,
    hand: np.ndarray,
    slope: np.ndarray,
    config: OpticalConfig,
) -> OpticalPair:
    """Validate both dates, dilate invalid boundaries, and normalize post to pre."""
    arrays = [
        *pre_data.values(),
        *post_data.values(),
        pre_scl,
        post_scl,
        pre_data_mask,
        post_data_mask,
        hand,
        slope,
    ]
    shape = np.asarray(pre_scl).shape
    if any(np.asarray(array).shape != shape for array in arrays):
        raise ValueError("Optical bands, masks, and terrain must share one grid.")

    valid = (
        np.isin(pre_scl, config.clear_scl_classes)
        & np.isin(post_scl, config.clear_scl_classes)
        & (pre_data_mask > 0)
        & (post_data_mask > 0)
        & np.isfinite(hand)
        & np.isfinite(slope)
    )
    for band in BANDS:
        valid &= (
            np.isfinite(pre_data[band])
            & np.isfinite(post_data[band])
            & (pre_data[band] >= config.reflectance_min)
            & (pre_data[band] <= config.reflectance_max)
            & (post_data[band] >= config.reflectance_min)
            & (post_data[band] <= config.reflectance_max)
        )
    if config.invalid_dilation_pixels:
        invalid = binary_dilation(
            ~valid,
            structure=np.ones((3, 3), dtype=bool),
            iterations=config.invalid_dilation_pixels,
        )
        valid &= ~invalid
    if not valid.any():
        raise ValueError("No jointly valid clear Sentinel-2 pixels remain after dilation.")

    stable = valid & (hand > config.stable_hand_min_m) & (slope <= config.stable_slope_max_deg)
    pre = {band: np.asarray(pre_data[band], dtype=np.float32).copy() for band in BANDS}
    post = {band: np.asarray(post_data[band], dtype=np.float32).copy() for band in BANDS}
    coefficients: dict[str, dict[str, float | int]] = {}
    for band in BANDS:
        change = post[band] - pre[band]
        stable_samples = change[stable]
        if stable_samples.size < 2:
            raise ValueError(f"Too few stable pixels to normalize band {band}.")
        low, high = np.quantile(
            stable_samples,
            [config.change_tail_fraction, 1.0 - config.change_tail_fraction],
        )
        fit = stable & (change >= low) & (change <= high)
        positive = fit & (post[band] > 1e-6) & (pre[band] > 1e-6)
        if positive.any():
            scale = float(np.median(pre[band][positive] / post[band][positive]))
        else:
            scale = 1.0
        offset = float(np.median(pre[band][fit] - scale * post[band][fit]))
        post[band] = post[band] * scale + offset
        coefficients[band] = {
            "scale": scale,
            "offset": offset,
            "stable_pixel_count": int(fit.sum()),
            "tail_low": float(low),
            "tail_high": float(high),
        }
    return OpticalPair(pre, post, valid, stable, coefficients)


def spectral_indices(bands: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Calculate NDVI, MNDWI, BSI, and visible brightness from reflectance."""
    b02, b03, b04 = bands["B02"], bands["B03"], bands["B04"]
    b08, b11 = bands["B08"], bands["B11"]
    return {
        "ndvi": _normalized_difference(b08, b04),
        "mndwi": _normalized_difference(b03, b11),
        "bsi": _normalized_difference(b11 + b04, b08 + b02),
        "brightness": (b02 + b03 + b04) / 3.0,
    }


def _normalized_difference(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(numerator.shape, np.nan, dtype=np.float32)
    total = numerator + denominator
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(total) > 1e-8)
    np.divide(numerator - denominator, total, out=output, where=valid)
    return output


def robust_sigma(values: np.ndarray, valid: np.ndarray) -> float:
    """Estimate robust standard deviation as 1.4826 times the MAD."""
    samples = values[valid & np.isfinite(values)]
    if samples.size == 0:
        return 0.0
    median = float(np.median(samples))
    return float(1.4826 * np.median(np.abs(samples - median)))
