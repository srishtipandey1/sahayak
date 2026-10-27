"""Tests for separate, append-only human review records."""

import json
from pathlib import Path

import pytest

from sahayak.config import Settings
from sahayak.review import (
    ReviewStoreError,
    ReviewSubmission,
    annotate_review_ids,
    read_review_history,
    record_review,
    review_feature_id,
    review_key,
)

FEATURE = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[[85.1, 27.1], [85.2, 27.1], [85.2, 27.2], [85.1, 27.1]]],
    },
    "properties": {"scenario": "default", "source": "optical"},
}


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sahayak_root=tmp_path,
        data_dir=Path("data"),
        runs_dir=Path("runs"),
        _env_file=None,
    )


def _write_hazard_layer(run_dir: Path) -> Path:
    layer_path = run_dir / "dashboard" / "layers" / "hazard_default.geojson"
    layer_path.parent.mkdir(parents=True)
    layer_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": [FEATURE]}),
        encoding="utf-8",
    )
    return layer_path


def test_review_feature_id_is_stable_and_bound_to_layer() -> None:
    assert review_feature_id("hazard_default", FEATURE) == review_feature_id(
        "hazard_default", json.loads(json.dumps(FEATURE))
    )
    assert review_feature_id("hazard_default", FEATURE) != review_feature_id(
        "hazard_strict", FEATURE
    )


def test_annotated_geojson_does_not_mutate_original_feature() -> None:
    source = {"type": "FeatureCollection", "features": [json.loads(json.dumps(FEATURE))]}

    annotated = annotate_review_ids("hazard_default", source)

    assert "review_id" not in source["features"][0]["properties"]
    assert annotated["features"][0]["properties"]["review_id"].startswith("rv_")


def test_annotated_geojson_rejects_missing_geometry() -> None:
    source = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": None, "properties": {}}],
    }

    with pytest.raises(ReviewStoreError, match="geometry must be an object"):
        annotate_review_ids("hazard_default", source)


def test_review_submission_normalizes_reviewer_and_optional_reason() -> None:
    submission = ReviewSubmission(status="confirmed", reviewer="  Field reviewer  ", reason="  ")

    assert submission.reviewer == "Field reviewer"
    assert submission.reason is None
    with pytest.raises(ValueError, match="must not be blank"):
        ReviewSubmission(status="confirmed", reviewer="   ")


def test_reviews_append_separately_without_modifying_detection_output(tmp_path: Path) -> None:
    run_dir = _settings(tmp_path).resolved_runs_dir / "review"
    layer_path = _write_hazard_layer(run_dir)
    original = layer_path.read_bytes()
    feature_id = review_feature_id("hazard_default", FEATURE)

    confirmed = record_review(
        run_dir=run_dir,
        layer_name="hazard_default",
        feature_id=feature_id,
        submission=ReviewSubmission(
            status="confirmed",
            reviewer="Reviewer A",
            reason="Visible in the post-event image.",
        ),
    )
    skipped = record_review(
        run_dir=run_dir,
        layer_name="hazard_default",
        feature_id=feature_id,
        submission=ReviewSubmission(status="skipped", reviewer="Reviewer B"),
    )

    history = read_review_history(run_dir / "reviews.json")
    entries = history[review_key("hazard_default", feature_id)]
    assert [entry.status for entry in entries] == ["confirmed", "skipped"]
    assert entries[0].reason == "Visible in the post-event image."
    assert entries[1].reason is None
    assert confirmed.timestamp_utc.tzinfo is not None
    assert skipped.timestamp_utc >= confirmed.timestamp_utc
    assert layer_path.read_bytes() == original


def test_review_requires_existing_feature_and_reviewable_layer(tmp_path: Path) -> None:
    run_dir = _settings(tmp_path).resolved_runs_dir / "review"
    _write_hazard_layer(run_dir)
    submission = ReviewSubmission(status="rejected", reviewer="Reviewer")

    with pytest.raises(ValueError, match="not present"):
        record_review(
            run_dir=run_dir,
            layer_name="hazard_default",
            feature_id="rv_" + "0" * 24,
            submission=submission,
        )
    with pytest.raises(ValueError, match="not available for review"):
        record_review(
            run_dir=run_dir,
            layer_name="buildings_hit",
            feature_id=review_feature_id("hazard_default", FEATURE),
            submission=submission,
        )


def test_corrupt_review_store_is_reported_instead_of_overwritten(tmp_path: Path) -> None:
    store_path = tmp_path / "reviews.json"
    store_path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ReviewStoreError, match="not valid JSON"):
        read_review_history(store_path)


def test_review_api_persists_decisions_and_sample_is_read_only(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sahayak.api import create_app

    settings = _settings(tmp_path)
    run_dir = settings.resolved_runs_dir / "api-review"
    _write_hazard_layer(run_dir)
    original = (run_dir / "dashboard" / "layers" / "hazard_default.geojson").read_bytes()
    client = TestClient(create_app(run_id="api-review", settings=settings))

    annotated = client.get("/api/reviewable/hazard_default.geojson")
    assert annotated.status_code == 200
    feature_id = annotated.json()["features"][0]["properties"]["review_id"]
    saved = client.post(
        "/api/reviews",
        json={
            "layer_name": "hazard_default",
            "feature_id": feature_id,
            "status": "rejected",
            "reviewer": "Reviewer A",
            "reason": "Not supported by visible imagery.",
        },
    )
    assert saved.status_code == 200
    assert saved.json()["review"]["status"] == "rejected"
    assert client.get("/api/reviews").json()["reviews"][
        review_key("hazard_default", feature_id)
    ][0]["reviewer"] == "Reviewer A"
    assert (run_dir / "dashboard" / "layers" / "hazard_default.geojson").read_bytes() == original
    invalid_feature = client.post(
        "/api/reviews",
        json={
            "layer_name": "hazard_default",
            "feature_id": "rv_" + "0" * 24,
            "status": "confirmed",
            "reviewer": "Reviewer",
        },
    )
    assert invalid_feature.status_code == 400
    invalid_status = client.post(
        "/api/reviews",
        json={
            "layer_name": "hazard_default",
            "feature_id": feature_id,
            "status": "uncertain",
            "reviewer": "Reviewer",
        },
    )
    assert invalid_status.status_code == 422
    unavailable_layer = client.get("/api/reviewable/hazard_strict.geojson")
    assert unavailable_layer.status_code == 404

    sample_dir = settings.resolved_data_dir.parent / "sample_data" / "read_only"
    _write_hazard_layer(sample_dir)
    sample_client = TestClient(
        create_app(run_id="read_only", settings=settings, sample=True)
    )
    assert sample_client.get("/api/reviews").json() == {"enabled": False, "reviews": {}}
    denied = sample_client.post(
        "/api/reviews",
        json={
            "layer_name": "hazard_default",
            "feature_id": feature_id,
            "status": "confirmed",
            "reviewer": "Reviewer",
        },
    )
    assert denied.status_code == 403
    assert not (sample_dir / "reviews.json").exists()
