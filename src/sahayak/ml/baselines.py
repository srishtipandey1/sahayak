"""VV-only flood baseline predictions on the labeled pixel population."""

import numpy as np

from sahayak.ml.metrics import confusion_counts, metrics_from_counts


def otsu_vv_threshold(vv_db: np.ndarray, labels: np.ndarray) -> float:
    """Compute a histogram Otsu threshold from finite valid VV dB pixels."""
    if vv_db.shape != labels.shape:
        raise ValueError("VV and label arrays must have matching shapes.")
    valid = np.isin(labels, (0, 1)) & np.isfinite(vv_db)
    values = vv_db[valid]
    if values.size == 0:
        raise ValueError("Otsu VV baseline requires finite labeled pixels.")
    minimum, maximum = float(values.min()), float(values.max())
    if maximum == minimum:
        return minimum
    histogram, edges = np.histogram(values, bins=256, range=(minimum, maximum))
    probabilities = histogram.astype(np.float64) / histogram.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * centers)
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between_class_variance = np.divide(
        np.square(global_mean * cumulative_probability - cumulative_mean),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    threshold_bin = int(np.argmax(between_class_variance))
    return float(edges[threshold_bin + 1])


def otsu_vv_metrics(vv_db: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    """Evaluate per-chip Otsu, with lower VV dB treated as water."""
    threshold = otsu_vv_threshold(vv_db, labels)
    valid = np.isin(labels, (0, 1)) & np.isfinite(vv_db)
    counts = confusion_counts(vv_db < threshold, np.where(valid, labels, 255))
    return {
        "threshold_db": threshold,
        "counts": counts,
        **metrics_from_counts(counts),
    }


def fixed_vv_metrics(
    vv_db: np.ndarray,
    labels: np.ndarray,
    threshold_db: float = -15.0,
) -> dict[str, object]:
    """Evaluate the fixed VV < -15 dB baseline on the same valid pixels."""
    valid = np.isin(labels, (0, 1)) & np.isfinite(vv_db)
    counts = confusion_counts(vv_db < threshold_db, np.where(valid, labels, 255))
    return {
        "threshold_db": float(threshold_db),
        "counts": counts,
        **metrics_from_counts(counts),
    }
