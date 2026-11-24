"""Synthetic tests for multi-run discovery."""

import json
from pathlib import Path

from sahayak.config import Settings
from sahayak.registry import build_registry, describe_run


def test_describe_run_summarizes_metadata_without_source_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "assam_flood"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "assam_flood",
                "name": "Assam Flood",
                "event_date": "2026-08-26",
                "bbox": [89.0, 25.0, 92.0, 27.0],
                "event_type": "flood",
                "state": "Assam",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "facts.json").write_text(
        json.dumps({"optical": {"reliability": "adequate"}}),
        encoding="utf-8",
    )
    (run_dir / "headline.json").write_text(
        json.dumps([{"id": "cut_off_default", "value": 1}]),
        encoding="utf-8",
    )

    result = describe_run(run_dir)

    assert result["run_id"] == "assam_flood"
    assert result["state"] == "Assam"
    assert result["evidence_status"] == "unvalidated"
    assert result["headline_count"] == 1
    assert "source" not in result


def test_build_registry_combines_runs_and_samples_once(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for run_id in ("trishuli_syapru", "assam_flood"):
        run_dir = runs / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "event_date": "2026-08-26", "bbox": [1, 2, 3, 4]}),
            encoding="utf-8",
        )
    sample = tmp_path / "sample_data" / "trishuli_syapru"
    sample.mkdir(parents=True)
    (sample / "manifest.json").write_text(
        json.dumps({"run_id": "trishuli_syapru", "bbox": [1, 2, 3, 4]}),
        encoding="utf-8",
    )

    settings = Settings(
        _env_file=None,
        sahayak_root=tmp_path,
        runs_dir=runs,
        data_dir=tmp_path / "data",
    )
    result = build_registry(settings)

    assert result["run_count"] == 2
    assert [item["run_id"] for item in result["runs"]] == [
        "assam_flood",
        "trishuli_syapru",
    ]
