"""Synthetic tests for validation metrics, scope guards, and frozen configs."""

import json
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from sahayak.config import Settings
from sahayak.provenance import ForbiddenInputError, open_reference
from sahayak.validate.metrics import area_metrics, not_comparable, point_match_metrics
from sahayak.validate.run import (
    _metric_crs,
    _search_osm_named_items,
    freeze_configs,
)


def test_metric_crs_accepts_projected_metre_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(500_000, 3_100_000)], crs="EPSG:32645")

    assert _metric_crs(frame).to_epsg() == 32645


def test_metric_crs_rejects_geographic_crs() -> None:
    frame = gpd.GeoDataFrame(geometry=[Point(85, 28)], crs="EPSG:4326")

    with pytest.raises(ValueError, match="projected CRS"):
        _metric_crs(frame)


def test_area_metrics_for_known_overlap() -> None:
    metrics = area_metrics(box(0, 0, 2, 2), box(1, 0, 3, 2))

    assert metrics["intersection_area_m2"] == 2
    assert metrics["predicted_area_m2"] == 4
    assert metrics["reference_area_m2"] == 4
    assert metrics["union_area_m2"] == 6
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_point_match_metrics_are_one_to_one_and_count_precision_recall() -> None:
    metrics = point_match_metrics(
        [box(0, 0, 10, 10), box(20, 0, 30, 10), box(100, 0, 110, 10)],
        [Point(5, 5), Point(25, 5), Point(60, 5)],
        tolerance_m=25,
    )

    assert metrics["prediction_count"] == 3
    assert metrics["reference_count"] == 3
    assert metrics["matched_count"] == 2
    assert metrics["unmatched_prediction_count"] == 1
    assert metrics["unmatched_reference_count"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_point_matching_maximizes_unique_matches_not_greedy_nearest_only() -> None:
    metrics = point_match_metrics(
        [Point(0, 0), Point(3, 0)],
        [Point(1, 0), Point(-2, 0)],
        tolerance_m=2,
    )

    assert metrics["matched_count"] == 2
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1


def test_open_reference_rejects_calls_outside_validation_package(tmp_path: Path) -> None:
    with pytest.raises(ForbiddenInputError, match="only be opened from sahayak.validate"):
        open_reference(tmp_path / "synthetic.gpkg", layer="synthetic")


def test_freeze_config_hashes_refuses_drift_without_refreeze(tmp_path: Path) -> None:
    root = tmp_path / "project"
    configs = root / "configs"
    configs.mkdir(parents=True)
    (configs / "optical.toml").write_text("threshold = 1\n", encoding="utf-8")
    (configs / "access.toml").write_text("speed = 2\n", encoding="utf-8")
    frozen_path = tmp_path / "runs" / "validate" / "frozen.json"

    first = freeze_configs(
        root,
        frozen_path,
        commit_hash="synthetic-commit",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    assert first["refrozen"] is False
    assert first["frozen"]["git_commit"] == "synthetic-commit"
    assert set(first["frozen"]["config_sha256"]) == {
        "configs/optical.toml",
        "configs/access.toml",
    }
    assert json.loads(frozen_path.read_text(encoding="utf-8")) == first["frozen"]

    (configs / "optical.toml").write_text("threshold = 3\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="--refreeze"):
        freeze_configs(root, frozen_path, commit_hash="synthetic-commit")

    refrozen = freeze_configs(
        root,
        frozen_path,
        refreeze=True,
        commit_hash="synthetic-commit",
        timestamp="2026-01-02T00:00:00+00:00",
    )
    assert refrozen["refrozen"] is True
    assert refrozen["frozen"]["timestamp_utc"] == "2026-01-02T00:00:00+00:00"
    assert (
        freeze_configs(root, frozen_path, commit_hash="later-commit")["refrozen"] is False
    )


def test_not_comparable_has_reason_and_no_score() -> None:
    result = not_comparable("No corresponding reference measurement.")

    assert result == {
        "status": "not comparable",
        "reason": "No corresponding reference measurement.",
    }
    assert "score" not in result
    with pytest.raises(ValueError, match="requires a reason"):
        not_comparable(" ")


def test_osm_named_item_search_handles_numeric_and_missing_attributes(
    tmp_path: Path,
) -> None:
    import geopandas as gpd
    import pyogrio

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    gpkg = run_dir / "network_osm_pre_event.gpkg"
    frame = gpd.GeoDataFrame(
        {
            "osm_id": [123],
            "name": [None],
            "population": [float("nan")],
            "tags_json": ['{"power":"plant","name":"Langtang Khola"}'],
        },
        geometry=[Point(85.35, 28.15)],
        crs="EPSG:4326",
    )
    pyogrio.write_dataframe(frame, gpkg, layer="health_facilities", driver="GPKG")
    settings = Settings(
        _env_file=None,
        sahayak_root=tmp_path,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
    )
    test_dir = settings.resolved_runs_dir / "test"
    test_dir.mkdir(parents=True)
    gpkg.replace(test_dir / gpkg.name)

    result = _search_osm_named_items(test_dir, box(85.3, 28.1, 85.4, 28.2))

    assert result["osm_pre_event_tag_search"]["hydroelectric_power_plant"][0]["osm_id"] == (
        "123"
    )


def test_open_reference_accepts_only_paths_under_reference_folder_from_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sahayak.validate import run as validation_run

    settings = Settings(
        _env_file=None,
        sahayak_root=tmp_path,
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
    )
    allowed = settings.reference_dir / "test.gpkg"
    elsewhere = tmp_path / "test.gpkg"
    called: list[Path] = []

    def fake_read_dataframe(path: Path, *, layer: str, **kwargs: object) -> str:
        called.append(Path(path))
        return layer

    monkeypatch.setattr("pyogrio.read_dataframe", fake_read_dataframe)
    monkeypatch.setattr(validation_run, "open_reference", open_reference)
    allowed.parent.mkdir(parents=True)
    assert validation_run._read_reference(allowed, "layer", settings) == "layer"
    assert called == [allowed.resolve()]
    with pytest.raises(ForbiddenInputError, match="inside"):
        validation_run._read_reference(elsewhere, "layer", settings)
