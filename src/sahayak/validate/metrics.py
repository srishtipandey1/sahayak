"""Pure geometry and feature-matching metrics for validation."""

from __future__ import annotations

from typing import Any

from shapely.geometry.base import BaseGeometry


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return a ratio, leaving undefined zero-denominator results explicit."""
    return numerator / denominator if denominator > 0 else None


def area_metrics(
    predicted: BaseGeometry,
    reference: BaseGeometry,
) -> dict[str, float | None]:
    """Calculate area-based IoU, precision, recall, and F1 for projected geometries."""
    intersection_area = float(predicted.intersection(reference).area)
    predicted_area = float(predicted.area)
    reference_area = float(reference.area)
    union_area = float(predicted.union(reference).area)
    precision = _ratio(intersection_area, predicted_area)
    recall = _ratio(intersection_area, reference_area)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision == 0 and recall == 0 else None)
    )
    return {
        "intersection_area_m2": intersection_area,
        "predicted_area_m2": predicted_area,
        "reference_area_m2": reference_area,
        "union_area_m2": union_area,
        "iou": _ratio(intersection_area, union_area),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def point_match_metrics(
    predicted_geometries: list[BaseGeometry],
    reference_points: list[BaseGeometry],
    *,
    tolerance_m: float = 25.0,
) -> dict[str, Any]:
    """Match reference points to predicted building geometries one-to-one."""
    if tolerance_m < 0:
        raise ValueError("Point-match tolerance must be non-negative.")
    candidates_by_reference: list[list[tuple[float, int]]] = []
    distances: dict[tuple[int, int], float] = {}
    for reference_index, point in enumerate(reference_points):
        candidates: list[tuple[float, int]] = []
        for prediction_index, prediction in enumerate(predicted_geometries):
            distance = float(prediction.distance(point))
            if distance <= tolerance_m:
                candidates.append((distance, prediction_index))
                distances[(prediction_index, reference_index)] = distance
        candidates_by_reference.append(sorted(candidates))

    prediction_to_reference: dict[int, int] = {}

    def assign(reference_index: int, visited_predictions: set[int]) -> bool:
        for _, prediction_index in candidates_by_reference[reference_index]:
            if prediction_index in visited_predictions:
                continue
            visited_predictions.add(prediction_index)
            assigned_reference = prediction_to_reference.get(prediction_index)
            if assigned_reference is None or assign(
                assigned_reference,
                visited_predictions,
            ):
                prediction_to_reference[prediction_index] = reference_index
                return True
        return False

    reference_order = sorted(
        range(len(reference_points)),
        key=lambda index: (len(candidates_by_reference[index]), index),
    )
    for reference_index in reference_order:
        assign(reference_index, set())

    matches = [
        {
            "prediction_index": prediction_index,
            "reference_index": reference_index,
            "distance_m": distances[(prediction_index, reference_index)],
        }
        for prediction_index, reference_index in sorted(prediction_to_reference.items())
    ]
    true_positives = len(prediction_to_reference)
    precision = _ratio(true_positives, len(predicted_geometries))
    recall = _ratio(true_positives, len(reference_points))
    return {
        "prediction_count": len(predicted_geometries),
        "reference_count": len(reference_points),
        "matched_count": true_positives,
        "unmatched_prediction_count": len(predicted_geometries) - true_positives,
        "unmatched_reference_count": len(reference_points) - true_positives,
        "tolerance_m": tolerance_m,
        "precision": precision,
        "recall": recall,
        "matches": matches,
    }


def not_comparable(reason: str) -> dict[str, str]:
    """Represent an unsupported validation comparison without a numeric score."""
    if not reason.strip():
        raise ValueError("A not-comparable result requires a reason.")
    return {"status": "not comparable", "reason": reason}
