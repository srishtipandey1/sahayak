"""Append-only human review records for dashboard hazard detections."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

REVIEWABLE_LAYERS = frozenset({"hazard_strict", "hazard_default", "hazard_lenient"})
REVIEW_STORE_VERSION = 1
_REVIEW_LOCK = Lock()


class ReviewStoreError(ValueError):
    """Raised when review metadata cannot be read or safely persisted."""


class ReviewSubmission(BaseModel):
    """Validated status and reviewer details submitted from the local dashboard."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["confirmed", "rejected", "skipped"]
    reviewer: str = Field(min_length=1, max_length=80)
    reason: str | None = Field(default=None, max_length=500)

    @field_validator("reviewer")
    @classmethod
    def _strip_reviewer(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Reviewer label must not be blank.")
        return value

    @field_validator("reason")
    @classmethod
    def _normalize_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ReviewDecision(ReviewSubmission):
    """One immutable, timestamped human review decision."""

    timestamp_utc: datetime


class ReviewRequest(ReviewSubmission):
    """Review status plus the layer and content ID selected by the reviewer."""

    layer_name: str
    feature_id: str


def review_key(layer_name: str, feature_id: str) -> str:
    """Return the review-store key for one feature in a dashboard layer."""
    return f"{layer_name}:{feature_id}"


def review_feature_id(layer_name: str, feature: dict[str, Any]) -> str:
    """Create a stable content-derived ID without changing the source feature."""
    if layer_name not in REVIEWABLE_LAYERS:
        raise ValueError(f"Layer {layer_name!r} is not available for review.")
    properties = feature.get("properties")
    geometry = feature.get("geometry")
    if feature.get("type") != "Feature" or not isinstance(properties, dict):
        raise ReviewStoreError("Reviewable GeoJSON contains an invalid feature.")
    if not isinstance(geometry, dict):
        raise ReviewStoreError("Reviewable feature geometry must be an object.")
    identity = {
        "layer": layer_name,
        "geometry": geometry,
        "properties": {
            key: value for key, value in properties.items() if key != "review_id"
        },
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"rv_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


def annotate_review_ids(layer_name: str, geojson: dict[str, Any]) -> dict[str, Any]:
    """Add response-only review IDs to reviewable features in a GeoJSON copy."""
    if layer_name not in REVIEWABLE_LAYERS:
        raise ValueError(f"Layer {layer_name!r} is not available for review.")
    features = geojson.get("features")
    if not isinstance(features, list):
        raise ReviewStoreError("Reviewable GeoJSON must contain a features list.")
    annotated = dict(geojson)
    annotated_features = []
    for feature in features:
        if not isinstance(feature, dict):
            raise ReviewStoreError("Reviewable GeoJSON contains an invalid feature.")
        if feature.get("type") != "Feature":
            raise ReviewStoreError("Reviewable GeoJSON contains an invalid feature.")
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            raise ReviewStoreError("Reviewable feature properties must be an object.")
        copied_feature = dict(feature)
        copied_feature["properties"] = {
            **properties,
            "review_id": review_feature_id(layer_name, feature),
        }
        annotated_features.append(copied_feature)
    annotated["features"] = annotated_features
    return annotated


def read_review_history(path: str | Path) -> dict[str, list[ReviewDecision]]:
    """Read and validate the separate append-only review store."""
    store_path = Path(path)
    if not store_path.is_file():
        return {}
    try:
        document = json.loads(store_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReviewStoreError("Review store is not valid JSON.") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != REVIEW_STORE_VERSION
        or not isinstance(document.get("reviews"), dict)
    ):
        raise ReviewStoreError("Review store schema is unsupported or malformed.")

    history: dict[str, list[ReviewDecision]] = {}
    try:
        for key, entries in document["reviews"].items():
            if not isinstance(key, str) or not isinstance(entries, list):
                raise ReviewStoreError("Review history entries must be lists keyed by strings.")
            history[key] = [ReviewDecision.model_validate(entry) for entry in entries]
    except ValidationError as error:
        raise ReviewStoreError("Review store contains an invalid decision record.") from error
    return history


def _write_review_history(
    path: Path,
    history: dict[str, list[ReviewDecision]],
) -> None:
    """Atomically replace the separate review file after serializing validated records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": REVIEW_STORE_VERSION,
        "reviews": {
            key: [decision.model_dump(mode="json") for decision in decisions]
            for key, decisions in history.items()
        },
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(document, temporary, ensure_ascii=False, indent=2, allow_nan=False)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def record_review(
    *,
    run_dir: str | Path,
    layer_name: str,
    feature_id: str,
    submission: ReviewSubmission,
) -> ReviewDecision:
    """Append a review for a feature present in the built hazard layer."""
    if layer_name not in REVIEWABLE_LAYERS:
        raise ValueError(f"Layer {layer_name!r} is not available for review.")
    if len(feature_id) != 27 or not feature_id.startswith("rv_"):
        raise ValueError("Review feature ID is invalid.")
    run_root = Path(run_dir)
    layer_path = run_root / "dashboard" / "layers" / f"{layer_name}.geojson"
    if not layer_path.is_file():
        raise FileNotFoundError(f"Review layer {layer_name!r} is unavailable.")
    try:
        geojson = json.loads(layer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReviewStoreError(f"Review layer {layer_name!r} is not valid JSON.") from error
    annotated = annotate_review_ids(layer_name, geojson)
    valid_ids = {
        feature["properties"]["review_id"] for feature in annotated["features"]
    }
    if feature_id not in valid_ids:
        raise ValueError("Review feature is not present in the selected hazard layer.")

    store_path = run_root / "reviews.json"
    key = review_key(layer_name, feature_id)
    decision = ReviewDecision(
        status=submission.status,
        reviewer=submission.reviewer,
        reason=submission.reason,
        timestamp_utc=datetime.now(UTC),
    )
    with _REVIEW_LOCK:
        history = read_review_history(store_path)
        history.setdefault(key, []).append(decision)
        _write_review_history(store_path, history)
    return decision
