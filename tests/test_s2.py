"""Network-free tests for Sentinel-2 selection and screening."""

from datetime import UTC, date, datetime

import numpy as np

from sahayak.ingest.s2 import (
    S2_BANDS,
    S2_EVALSCRIPT,
    choose_s2_scenes,
    deduplicate_s2_scenes,
    scl_fractions,
)


def _scene(
    scene_id: str,
    acquired: str,
    clear: float,
    cloud: float,
) -> dict[str, object]:
    return {
        "id": scene_id,
        "acquired_at": datetime.fromisoformat(acquired).replace(tzinfo=UTC),
        "clear_fraction": clear,
        "eo_cloud_cover": cloud,
    }


def test_scl_fractions_report_requested_categories() -> None:
    result = scl_fractions(
        np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.float32)
    )

    assert result["pixel_count"] == 12
    assert result["clear_fraction"] == 4 / 12
    assert result["fractions"]["dark_topographic_shadow"]["count"] == 1
    assert result["fractions"]["cloud_shadow"]["count"] == 1
    assert result["fractions"]["cloud_and_cirrus"]["count"] == 3
    assert result["fractions"]["snow_or_ice"]["count"] == 1
    assert result["fractions"]["no_data_or_defective"]["count"] == 2


def test_s2_same_date_duplicates_keep_lowest_cloud_item() -> None:
    scenes = [
        _scene("pre-cloudy", "2026-08-12T05:00:00", 0.4, 30),
        _scene("pre-clear", "2026-08-12T06:00:00", 0.8, 10),
    ]

    kept, duplicates = deduplicate_s2_scenes(scenes)

    assert [scene["id"] for scene in kept] == ["pre-clear"]
    assert duplicates == [
        {
            "date": "2026-08-12",
            "kept_id": "pre-clear",
            "duplicate_id": "pre-cloudy",
            "reason": "Same-date duplicate; item with lower eo:cloud_cover retained.",
        }
    ]


def test_s2_scene_choice_uses_earliest_and_latest_usable_scenes() -> None:
    scenes = [
        _scene("pre-early", "2026-08-01T05:00:00", 0.9, 20),
        _scene("pre-latest-usable", "2026-08-12T05:00:00", 0.6, 30),
        _scene("pre-latest-cloudy", "2026-08-20T05:00:00", 0.4, 5),
        _scene("post-first-usable", "2026-09-01T05:00:00", 0.8, 40),
        _scene("post-second", "2026-09-10T05:00:00", 0.9, 5),
    ]

    pre, post, reasons = choose_s2_scenes(scenes, date(2026, 8, 26))

    assert pre["id"] == "pre-latest-usable"
    assert post["id"] == "post-first-usable"
    assert "Latest pre-event" in reasons["pre"]
    assert "Earliest post-event" in reasons["post"]


def test_s2_scene_choice_fallback_uses_best_clear_fraction_and_ties() -> None:
    scenes = [
        _scene("pre-earlier", "2026-08-01T05:00:00", 0.4, 10),
        _scene("pre-later", "2026-08-12T05:00:00", 0.4, 10),
        _scene("post-later", "2026-09-10T05:00:00", 0.7, 10),
        _scene("post-earlier", "2026-09-01T05:00:00", 0.7, 10),
    ]

    pre, post, reasons = choose_s2_scenes(scenes, date(2026, 8, 26))

    assert pre["id"] == "pre-later"
    assert post["id"] == "post-earlier"
    assert reasons["pre"].startswith("Fallback")
    assert reasons["post"].startswith("Fallback")


def test_s2_evalscript_requests_reflectance_and_expected_bands() -> None:
    assert all(f"sample.{band}" in S2_EVALSCRIPT for band in S2_BANDS)
    assert 'sampleType: "FLOAT32"' in S2_EVALSCRIPT
    assert '"REFLECTANCE"' in S2_EVALSCRIPT
