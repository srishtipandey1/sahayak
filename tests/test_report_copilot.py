"""Synthetic tests for curated facts, report templates, and grounded Q&A."""

import json
import string
from pathlib import Path

from sahayak.config import Settings
from sahayak.copilot import answer_question, route_intent
from sahayak.report.facts import build_headline, write_headline
from sahayak.report.i18n import MESSAGES
from sahayak.report.run import _find_browser, generate_reports
from sahayak.report.verify import normalize_digits, verify_text


def _sample_facts() -> tuple[dict, dict]:
    scenarios = ("strict", "default", "lenient")
    return (
        {
            "hazard": {"levels": {level: {"area_ha": 1.5} for level in scenarios}},
            "damage": {
                "scenarios": {
                    level: {
                        "buildings_inside": {"value": 2, "unit": "buildings"},
                        "buildings_near_30m": {"value": 1, "unit": "buildings"},
                        "road_km_hit_by_class": {"primary": {"value": 0.5, "unit": "km"}},
                        "bridges_hit_within_20m": {"value": 1, "unit": "bridges"},
                        "settlements_affected": {"value": 1, "unit": "settlements"},
                    }
                    for level in scenarios
                }
            },
            "access": {
                "scenarios": {
                    level: {
                        "counts_by_class": {
                            "cut_off": {"value": 1, "unit": "settlements"},
                            "degraded": {"value": 2, "unit": "settlements"},
                            "unaffected": {"value": 3, "unit": "settlements"},
                            "not_assessable": {"value": 4, "unit": "settlements"},
                        },
                        "cut_off_settlements": ["स्याप्रु बेसी"],
                    }
                    for level in scenarios
                },
                "robust_cut_off_settlement_ids": [10],
                "robust_cut_off_settlements": ["स्याप्रु बेसी"],
                "not_assessable_count": 4,
                "settlement_routes": {
                    "10": {
                        "name": "स्याप्रु बेसी",
                        "scenarios": {
                            "default": {
                                "hospital_before_name": "Local Clinic",
                                "hospital_before_km": 2.4,
                                "hospital_before_minutes": 12.0,
                            }
                        },
                    }
                },
            },
            "optical": {
                "reliability": "adequate",
                "valid_pixel_fraction": {"value": 0.75, "unit": "fraction"},
                "provenance": {
                    "scene_dates": {"pre": "2026-08-12", "post": "2026-10-01"}
                },
            },
            "flood": {
                "areas_ha_by_class": {
                    "new_water": {"value": 0.0},
                    "disturbed_surface": {"value": 0.0},
                }
            },
        },
        {
            "event_date": "2026-08-26",
            "bbox": [85.3, 28.1, 85.42, 28.2],
            "inputs": [
                {"role": "input", "source": "Copernicus Sentinel-2"},
                {"role": "reference", "source": "Validation only"},
            ],
        },
    )


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        sahayak_root=tmp_path,
        runs_dir=Path("runs"),
        _env_file=None,
    )


def test_localized_templates_have_matching_keys_and_complete_placeholders() -> None:
    assert set(MESSAGES["en"]) == set(MESSAGES["hi"]) == set(MESSAGES["ne"])
    for messages in MESSAGES.values():
        for template in messages.values():
            placeholders = {
                field_name
                for _, field_name, _, _ in string.Formatter().parse(template)
                if field_name
            }
            template.format(**{name: "value" for name in placeholders})


def test_qa_routes_all_thirty_language_cases() -> None:
    cases_path = Path(__file__).parent / "data" / "qa_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert len(cases) == 30
    assert {
        language: sum(item["language"] == language for item in cases)
        for language in MESSAGES
    } == {"en": 10, "hi": 10, "ne": 10}
    assert all(route_intent(item["question"]) == item["intent"] for item in cases)


def test_numeric_verifier_rejects_invented_number_and_normalizes_devanagari() -> None:
    headline = [{"id": "count", "value": 5, "unit": "settlements"}]
    assert verify_text("There are 5 settlements.", headline)["verified"]
    result = verify_text("There are 500 settlements.", headline)
    assert result == {"verified": False, "unverified_numbers": ["500"]}
    assert normalize_digits("बस्ती १२") == "बस्ती 12"
    assert verify_text("बस्ती ५", [{"id": "count", "value": 5}])["verified"]


def test_missing_nan_names_are_not_materialized_as_the_literal_nan(tmp_path: Path) -> None:
    import geopandas as gpd
    import numpy as np
    import pyogrio
    from shapely.geometry import Point

    from sahayak.report.facts import _is_missing, _materialize_settlement_routes

    settings = _settings(tmp_path)
    run_dir = settings.resolved_runs_dir / "synthetic"
    run_dir.mkdir(parents=True)
    frame = gpd.GeoDataFrame(
        {
            "osm_id": [10],
            "name": ["Settlement"],
            "default_hospital_before_name": [np.nan],
            "default_hospital_before_km": [7.3543],
            "default_hospital_before_minutes": [37.7196],
        },
        geometry=[Point(0, 0)],
        crs="EPSG:32645",
    )
    gpkg_path = run_dir / "settlement_access.gpkg"
    pyogrio.write_dataframe(frame, gpkg_path, layer="settlements", driver="GPKG")
    facts: dict = {}

    _materialize_settlement_routes(run_dir, facts)

    route = facts["access"]["settlement_routes"]["10"]["scenarios"]["default"]
    assert route["hospital_before_name"] is None
    assert route["hospital_before_km"] == 7.3543
    assert route["hospital_before_minutes"] == 37.7196
    assert _is_missing(float("nan"))
    assert _is_missing("nan")
    assert not _is_missing("Unnamed health facility")


def test_numeric_verifier_accepts_rounded_percentage_from_fraction_fact() -> None:
    headline = [
        {"id": "valid_fraction", "value": 0.8691308393820957, "unit": "fraction"}
    ]

    assert verify_text("Valid optical pixels: 86.9%", headline)["verified"]
    assert verify_text("Valid optical pixels: 87.5%", headline) == {
        "verified": False,
        "unverified_numbers": ["87.5%"],
    }


def test_headline_keeps_missing_values_null_and_has_source_paths() -> None:
    headline = build_headline({}, {})
    assert headline
    assert all(
        {"id", "value", "unit", "label_en", "source_fact_path", "scenario", "confidence_note"}
        <= set(item)
        for item in headline
    )
    assert next(item for item in headline if item["id"] == "hazard_area_default")["value"] is None
    assert next(item for item in headline if item["id"] == "cut_off_default")["value"] is None


def test_write_headline_and_grounded_answers_use_headline_only(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_dir = settings.resolved_runs_dir / "synthetic"
    run_dir.mkdir(parents=True)
    facts, manifest = _sample_facts()
    (run_dir / "facts.json").write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    path = write_headline("synthetic", settings)
    assert path.is_file()

    result = answer_question(
        run_id="synthetic",
        language="en",
        question="Nearest hospital to स्याप्रु बेसी?",
        settings=settings,
    )
    assert result["verification"]["verified"]
    assert "Local Clinic" in result["answer"]
    assert "2.40 km" in result["answer"]
    assert "12 minutes" in result["answer"]


def test_null_answer_renders_not_available_not_zero(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_dir = settings.resolved_runs_dir / "synthetic"
    run_dir.mkdir(parents=True)
    (run_dir / "headline.json").write_text(
        json.dumps([{"id": "cut_off_default", "value": None, "unit": "settlements"}]),
        encoding="utf-8",
    )
    result = answer_question(
        run_id="synthetic",
        language="en",
        question="How many settlements are affected?",
        settings=settings,
    )
    assert "not available" in result["answer"]
    assert "0 cut off" not in result["answer"]
    assert result["verification"]["verified"]


def test_report_is_one_page_when_a_local_browser_exists(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    run_dir = settings.resolved_runs_dir / "synthetic"
    run_dir.mkdir(parents=True)
    facts, manifest = _sample_facts()
    headline = build_headline(facts, manifest)
    (run_dir / "headline.json").write_text(
        json.dumps(headline, ensure_ascii=False), encoding="utf-8"
    )

    results = generate_reports(
        run_id="synthetic",
        languages=("en",),
        settings=settings,
    )
    assert results[0]["verification"]["verified"]
    if _find_browser() is not None:
        assert results[0]["page_count"] == 1
    else:
        assert results[0]["page_count"] is None
