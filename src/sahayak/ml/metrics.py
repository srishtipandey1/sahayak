"""Flood segmentation metrics, validation thresholding, and chip bootstrap intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

IGNORE_VALUES = (-1, 255)
THRESHOLD_GRID_SIZE = 1000


def confusion_counts(prediction: np.ndarray, labels: np.ndarray) -> dict[str, int]:
    """Return binary confusion counts while excluding -1 and 255 labels."""
    if prediction.shape != labels.shape:
        raise ValueError("Prediction and label arrays must have matching shapes.")
    valid = np.isin(labels, (0, 1))
    predicted = np.asarray(prediction, dtype=bool)[valid]
    truth = np.asarray(labels)[valid] == 1
    return {
        "tn": int((~predicted & ~truth).sum()),
        "fp": int((predicted & ~truth).sum()),
        "fn": int((~predicted & truth).sum()),
        "tp": int((predicted & truth).sum()),
        "valid_pixels": int(valid.sum()),
        "positive_pixels": int(truth.sum()),
    }


def metrics_from_counts(counts: dict[str, int]) -> dict[str, object]:
    """Calculate IoU, F1, precision, recall, accuracy, and confusion matrix."""
    tn, fp, fn, tp = (int(counts[key]) for key in ("tn", "fp", "fn", "tp"))
    union = tp + fp + fn
    predicted_positive = tp + fp
    actual_positive = tp + fn
    total = tn + fp + fn + tp
    iou = tp / union if union else 0.0
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else 0.0
    return {
        "flood_iou": iou,
        "f1": f1,
        "precision": tp / predicted_positive if predicted_positive else 0.0,
        "recall": tp / actual_positive if actual_positive else 0.0,
        "accuracy": (tn + tp) / total if total else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "pixel_counts": {
            "valid": total,
            "positive": actual_positive,
            "negative": tn + fp,
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
        },
    }


def select_threshold(
    probabilities: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
) -> float:
    """Choose the 0.001-grid validation threshold with maximum pooled F1."""
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("Threshold selection requires equally sized non-empty chip lists.")
    positive_histogram = np.zeros(THRESHOLD_GRID_SIZE, dtype=np.int64)
    negative_histogram = np.zeros(THRESHOLD_GRID_SIZE, dtype=np.int64)
    for probability, label in zip(probabilities, labels, strict=True):
        if probability.shape != label.shape:
            raise ValueError("Probability and label arrays must have matching shapes.")
        valid = np.isin(label, (0, 1)) & np.isfinite(probability)
        values = np.clip(probability[valid], 0.0, 1.0)
        bins = np.minimum((values * THRESHOLD_GRID_SIZE).astype(np.int64), THRESHOLD_GRID_SIZE - 1)
        positives = np.asarray(label)[valid] == 1
        positive_histogram += np.bincount(
            bins[positives],
            minlength=THRESHOLD_GRID_SIZE,
        )
        negative_histogram += np.bincount(
            bins[~positives],
            minlength=THRESHOLD_GRID_SIZE,
        )
    if not positive_histogram.sum() + negative_histogram.sum():
        raise ValueError("Threshold selection requires valid labeled pixels.")
    true_positive = np.cumsum(positive_histogram[::-1])[::-1]
    false_positive = np.cumsum(negative_histogram[::-1])[::-1]
    total_positive = int(positive_histogram.sum())
    false_negative = total_positive - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator > 0,
    )
    maximum = float(f1.max())
    best_bin = int(np.flatnonzero(f1 == maximum)[-1])
    return best_bin / THRESHOLD_GRID_SIZE


def chip_metric_distribution(
    counts_by_chip: Sequence[dict[str, int]],
) -> dict[str, dict[str, float]]:
    """Summarize per-chip IoU and F1 with median, p10, and p90."""
    values = {"flood_iou": [], "f1": []}
    for counts in counts_by_chip:
        metrics = metrics_from_counts(counts)
        for name in values:
            values[name].append(float(metrics[name]))
    return {
        name: {
            "median": float(np.median(samples)),
            "p10": float(np.percentile(samples, 10)),
            "p90": float(np.percentile(samples, 90)),
        }
        for name, samples in values.items()
    }


def bootstrap_intervals(
    counts_by_chip: Sequence[dict[str, int]],
    *,
    resamples: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap chips with replacement and report percentile intervals for IoU/F1."""
    if not counts_by_chip or resamples < 1:
        raise ValueError("Bootstrap requires chips and at least one resample.")
    rng = np.random.default_rng(seed)
    sample_count = len(counts_by_chip)
    values = {"flood_iou": np.empty(resamples), "f1": np.empty(resamples)}
    for index in range(resamples):
        sample_indices = rng.integers(0, sample_count, size=sample_count)
        total = {key: 0 for key in ("tn", "fp", "fn", "tp")}
        for chip_index in sample_indices:
            counts = counts_by_chip[int(chip_index)]
            for key in total:
                total[key] += int(counts[key])
        metrics = metrics_from_counts(total)
        for name in values:
            values[name][index] = float(metrics[name])
    return {
        name: {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
        }
        for name, samples in values.items()
    }


def aggregate_counts(counts_by_chip: Sequence[dict[str, int]]) -> dict[str, int]:
    """Sum confusion and pixel counts across chips."""
    return {
        key: sum(int(chip[key]) for chip in counts_by_chip)
        for key in (
            "tn",
            "fp",
            "fn",
            "tp",
            "valid_pixels",
            "positive_pixels",
        )
    }
