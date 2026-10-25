"""VV-only flood baseline predictions on the labeled pixel population."""

import logging

import numpy as np

from sahayak.ml.metrics import confusion_counts, metrics_from_counts

LOGGER = logging.getLogger(__name__)
MINIMUM_VALID_PIXELS = 100


class AllBaselineChipsSkipped(ValueError):
    """Raised when no chip has enough finite, non-constant VV pixels to score."""

    def __init__(self, method: str, skipped_chips: list[dict[str, object]]) -> None:
        self.skipped_chips = skipped_chips
        super().__init__(
            f"{method} baseline skipped every chip ({len(skipped_chips)} total)."
        )


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


def evaluate_baseline(
    evaluation: dict[str, object],
    method: str,
) -> dict[str, object]:
    """Score one VV baseline, skipping chips with unusable labeled VV values."""
    if method not in {"otsu", "fixed_minus_15_db"}:
        raise ValueError(f"Unknown VV baseline method: {method}")
    vv_chips = evaluation["vv_db"]
    label_chips = evaluation["labels"]
    chip_ids = evaluation.get("chip_ids", [])
    if len(vv_chips) != len(label_chips):
        raise ValueError("VV and label chip lists must have matching lengths.")
    if chip_ids and len(chip_ids) != len(vv_chips):
        raise ValueError("Chip IDs and arrays must have matching lengths.")

    counts_by_chip: list[dict[str, int]] = []
    skipped_chips: list[dict[str, object]] = []
    for index, (vv_db, labels) in enumerate(zip(vv_chips, label_chips, strict=True)):
        chip_id = str(chip_ids[index]) if chip_ids else f"chip-{index:04d}"
        finite_vv = np.isfinite(vv_db)
        valid = np.isin(labels, (0, 1)) & finite_vv
        valid_pixel_count = int(valid.sum())
        finite_vv_count = int(finite_vv.sum())
        is_constant = valid_pixel_count > 0 and float(np.ptp(vv_db[valid])) == 0.0
        if valid_pixel_count < MINIMUM_VALID_PIXELS or is_constant:
            reason = (
                "constant VV"
                if is_constant
                else f"fewer than {MINIMUM_VALID_PIXELS} valid pixels"
            )
            record = {
                "chip_id": chip_id,
                "valid_pixel_count": valid_pixel_count,
                "finite_vv_count": finite_vv_count,
                "reason": reason,
            }
            skipped_chips.append(record)
            LOGGER.warning(
                "Skipping %s baseline chip %s: valid_pixels=%d finite_vv_pixels=%d reason=%s",
                method,
                chip_id,
                valid_pixel_count,
                finite_vv_count,
                reason,
            )
            continue

        baseline = (
            otsu_vv_metrics(vv_db, labels)
            if method == "otsu"
            else fixed_vv_metrics(vv_db, labels)
        )
        counts_by_chip.append(baseline["counts"])

    if not counts_by_chip:
        raise AllBaselineChipsSkipped(method, skipped_chips)
    aggregate = {
        key: sum(int(counts[key]) for counts in counts_by_chip)
        for key in ("tn", "fp", "fn", "tp", "valid_pixels", "positive_pixels")
    }
    return {
        **metrics_from_counts(aggregate),
        "chip_count": len(counts_by_chip),
        "skipped_chips": {
            "count": len(skipped_chips),
            "chip_ids": skipped_chips,
        },
        "counts_by_chip": counts_by_chip,
    }
