"""Synthetic tests for benchmark and field-priority summaries."""

import json
from pathlib import Path

from sahayak.benchmark import build_benchmark
from sahayak.config import Settings
from sahayak.registry import build_priorities


def test_benchmark_marks_missing_validation_and_counts_runs(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "assam_flood"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"event_date": "2026-08-26", "event_type": "flood", "state": "Assam"}),
        encoding="utf-8",
    )
    (run_dir / "facts.json").write_text(
        json.dumps(
            {
                "optical": {"reliability": "adequate", "valid_pixel_fraction": {"value": 0.8}},
                "access": {"robust_cut_off_settlement_ids": [10], "not_assessable_count": 66},
                "flood": {
                    "areas_ha_by_class": {
                        "new_water": {"value": 0.0},
                        "disturbed_surface": {"value": 0.0},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        sahayak_root=tmp_path,
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path / "data",
    )

    result = build_benchmark(settings)

    assert result["run_count"] == 1
    assert result["rows"][0]["validation_status"] == "not-comparable"
    assert result["rows"][0]["validation_iou"] is None


def test_priorities_identify_robust_cutoff_and_network_gaps(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "facts.json").write_text(
        json.dumps(
            {
                "access": {
                    "robust_cut_off_settlements": ["Syapru Besi"],
                    "not_assessable_count": 66,
                    "settlement_routes": {"10": {"name": "Syapru Besi"}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = build_priorities(run_dir)

    assert result["route_record_count"] == 1
    assert result["priorities"][0] == {
        "name": "Syapru Besi",
        "reason": "cut_off_in_all_scenarios",
        "field_verification": "high",
    }
