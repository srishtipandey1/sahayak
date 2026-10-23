"""Tests for ohsome request construction, selection, and API versions."""

from datetime import UTC, date, datetime

import pytest

from sahayak.ingest.osm import (
    LAYER_FILTERS,
    OhsomeError,
    build_ohsome_request,
    detect_ohsome_api_version,
    select_osm_snapshot,
    tile_bboxes,
)
from sahayak.provenance import RuleViolation


def test_v1_request_uses_documented_geometry_endpoint_and_parameters() -> None:
    method, endpoint, params = build_ohsome_request(
        api_version="v1",
        bbox=(85.3, 28.1, 85.42, 28.2),
        snapshot_date=date(2026, 7, 27),
        filter_text=LAYER_FILTERS["roads"],
        base_url="https://api.ohsome.org/v1",
    )

    assert method == "GET"
    assert endpoint == "https://api.ohsome.org/v1/elements/geometry"
    assert params == {
        "bboxes": "85.30000000,28.10000000,85.42000000,28.20000000",
        "time": "2026-07-27",
        "filter": "highway=* and geometry:line",
        "properties": "tags",
    }


def test_v2_request_uses_extraction_features_and_aoi() -> None:
    method, endpoint, payload = build_ohsome_request(
        api_version="v2",
        bbox=(85.3, 28.1, 85.42, 28.2),
        snapshot_date=date(2026, 7, 27),
        filter_text=LAYER_FILTERS["buildings"],
        base_url="https://api.heigit.org/ohsome-api/v2-rc",
    )

    assert method == "POST"
    assert endpoint == "https://api.heigit.org/ohsome-api/v2-rc/extraction/features"
    assert payload == {
        "aoi": [85.3, 28.1, 85.42, 28.2],
        "time": "2026-07-27",
        "filter": "building=*",
    }


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.ohsome.org/v1", "v1"),
        ("https://api.heigit.org/ohsome-api/v2", "v2"),
        ("https://api.heigit.org/ohsome-api/v2-rc", "v2"),
    ],
)
def test_api_version_is_configurable_in_url(url: str, expected: str) -> None:
    assert detect_ohsome_api_version(url) == expected


def test_unsupported_api_url_fails_clearly() -> None:
    with pytest.raises(OhsomeError, match="must end in"):
        detect_ohsome_api_version("https://example.invalid")


def test_snapshot_default_uses_event_minus_one_day() -> None:
    latest = datetime(2026, 8, 30, 9, tzinfo=UTC)

    assert select_osm_snapshot(date(2026, 8, 26), latest) == date(2026, 8, 25)


def test_snapshot_default_is_capped_by_latest_available_date() -> None:
    latest = datetime(2026, 7, 27, 9, tzinfo=UTC)

    assert select_osm_snapshot(date(2026, 7, 28), latest) == date(2026, 7, 27)


def test_snapshot_guard_rejects_before_equal_and_after_event() -> None:
    latest = datetime(2026, 8, 30, tzinfo=UTC)
    assert select_osm_snapshot(date(2026, 8, 26), latest, date(2026, 8, 25))

    with pytest.raises(RuleViolation, match="earlier"):
        select_osm_snapshot(date(2026, 8, 26), latest, date(2026, 8, 26))
    with pytest.raises(RuleViolation, match="earlier"):
        select_osm_snapshot(date(2026, 8, 26), latest, date(2026, 8, 27))


def test_snapshot_guard_rejects_date_later_than_latest() -> None:
    latest = datetime(2026, 7, 27, 9, tzinfo=UTC)

    with pytest.raises(ValueError, match="later than latest available"):
        select_osm_snapshot(date(2026, 8, 26), latest, date(2026, 7, 28))


def test_large_bbox_is_tiled_without_gaps() -> None:
    tiles = tile_bboxes((0.0, 0.0, 0.6, 0.5))

    assert len(tiles) == 120
    assert all(
        east - west <= 0.05 + 1e-12 and north - south <= 0.05 + 1e-12
        for west, south, east, north in tiles
    )
