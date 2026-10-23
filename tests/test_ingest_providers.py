"""Offline tests for the Overpass and ohsome OSM providers."""

import json
import logging
import re
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import geopandas as gpd
import pyogrio
import pytest
import requests
from pydantic import SecretStr
from shapely.geometry import LineString

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.ingest.osm import (
    LAYER_NAMES,
    OhsomeError,
    OhsomeProvider,
    OverpassProvider,
    TileResult,
    _overpass_result_layers,
    _write_osm_layers,
    build_overpass_query,
    parse_overpass_payload,
    select_osm_provider,
    tile_bboxes,
    verify_osm_timestamp_compliance,
)
from sahayak.provenance import RuleViolation, RunManifest


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    """Create isolated settings and cache paths for provider tests."""
    return Settings(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        _env_file=None,
        **overrides,
    )


def test_overpass_query_pins_date_and_tile_splitting() -> None:
    query = build_overpass_query((85.3, 28.1, 85.42, 28.2), date(2026, 8, 25))
    tiles = tile_bboxes((85.3, 28.1, 85.42, 28.2))

    assert '[out:json]' in query
    assert '[date:"2026-08-25T00:00:00Z"]' in query
    assert "out meta geom;" in query
    assert 'nwr["building"]' in query
    assert 'way["highway"]' in query
    assert 'nwr["bridge"]' in query
    assert 'nwr["man_made"="bridge"]' in query
    assert 'nwr["place"~"^(village|hamlet|town|city|suburb|locality)$"]' in query
    assert 'nwr["amenity"~"^(hospital|clinic|doctors)$"]' in query
    assert 'nwr["healthcare"~"^(hospital|clinic)$"]' in query
    assert len(tiles) == 6
    assert all(
        east - west <= 0.05 + 1e-12 and north - south <= 0.05 + 1e-12
        for west, south, east, north in tiles
    )


def test_overpass_parser_handles_nodes_ways_and_skipped_relation() -> None:
    payload = {
        "osm3s": {"timestamp_osm_base": "2026-08-25T00:00:00Z"},
        "elements": [
            {
                "type": "node",
                "id": 1,
                "timestamp": "2026-08-24T15:00:00Z",
                "version": 4,
                "lon": 85.31,
                "lat": 28.11,
                "tags": {"place": "village", "name": "Example"},
            },
            {
                "type": "way",
                "id": 2,
                "timestamp": "2026-08-24T16:00:00Z",
                "version": 8,
                "geometry": [{"lon": 85.31, "lat": 28.11}, {"lon": 85.32, "lat": 28.12}],
                "tags": {"highway": "track"},
            },
            {
                "type": "relation",
                "id": 3,
                "timestamp": "2026-08-24T17:00:00Z",
                "version": 1,
                "members": [],
                "tags": {"building": "yes"},
            },
        ],
    }

    result = parse_overpass_payload(json.dumps(payload))

    assert result.osm_base_timestamp == "2026-08-25T00:00:00Z"
    assert result.layers["settlements"][0]["geometry"].geom_type == "Point"
    assert result.layers["roads"][0]["road_class"] == "track"
    assert result.layers["roads"][0]["osm_version"] == 8
    assert result.skipped_relations["buildings"] == 1
    assert result.skipped_relation_ids["buildings"] == {("relation", 3)}
    assert result.max_element_timestamp.isoformat() == "2026-08-24T17:00:00+00:00"


def test_overpass_deduplicates_elements_repeated_across_tiles() -> None:
    record = {
        "osm_type": "way",
        "osm_id": 7,
        "geometry": None,
        "highway": "path",
    }
    results = [
        TileResult(
            {layer: ([record] if layer == "roads" else []) for layer in LAYER_NAMES},
            skipped_relation_ids={
                layer: ({("relation", 42)} if layer == "buildings" else set())
                for layer in LAYER_NAMES
            },
        )
        for _ in range(2)
    ]
    provider = SimpleNamespace(
        extract_tile=Mock(side_effect=results),
        skipped_relation_ids={layer: set() for layer in LAYER_NAMES},
        element_count=0,
        max_element_timestamp=None,
        per_layer_max_timestamps={layer: None for layer in LAYER_NAMES},
    )

    features, skipped = _overpass_result_layers(provider, (0.0, 0.0, 0.1, 0.05), date(2026, 1, 1))

    assert len(features["roads"]) == 1
    assert skipped["buildings"] == 1


def test_multiclass_features_are_classified_for_every_matching_layer() -> None:
    payload = {
        "elements": [
            {
                "type": "way",
                "id": 10,
                "timestamp": "2026-08-24T23:00:00Z",
                "version": 2,
                "geometry": [
                    {"lon": 85.31, "lat": 28.11},
                    {"lon": 85.32, "lat": 28.12},
                ],
                "tags": {"highway": "primary", "bridge": "yes", "tunnel": "no"},
            },
            {
                "type": "way",
                "id": 11,
                "timestamp": "2026-08-24T22:00:00Z",
                "version": 3,
                "geometry": [
                    {"lon": 85.31, "lat": 28.11},
                    {"lon": 85.32, "lat": 28.11},
                    {"lon": 85.32, "lat": 28.12},
                    {"lon": 85.31, "lat": 28.11},
                ],
                "tags": {"building": "yes", "amenity": "hospital", "name": "Clinic"},
            },
        ]
    }

    result = parse_overpass_payload(json.dumps(payload))

    assert len(result.layers["roads"]) == 1
    assert len(result.layers["bridges"]) == 1
    assert result.layers["roads"][0]["is_bridge"] is True
    assert result.layers["roads"][0]["is_tunnel"] is False
    assert len(result.layers["buildings"]) == 1
    assert len(result.layers["health_facilities"]) == 1
    assert result.layers["health_facilities"][0]["geometry"].geom_type == "Point"


def test_overpass_retries_and_honours_retry_after(tmp_path) -> None:
    sleeps: list[float] = []
    first = SimpleNamespace(status_code=429, headers={"Retry-After": "5"}, content=b"", text="")
    second = SimpleNamespace(status_code=200, headers={}, content=b'{"elements":[]}', text="")
    session = Mock(post=Mock(side_effect=[first, second]))
    settings = _settings(tmp_path)
    provider = OverpassProvider(
        settings,
        RequestCache(settings),
        session=session,
        sleep=sleeps.append,
        request_interval=0,
    )

    assert provider._request_bytes("query") == b'{"elements":[]}'
    assert sleeps == [5.0]
    assert provider.request_count == 2
    assert session.post.call_args.kwargs["headers"]["User-Agent"] == "sahayak-hackathon-prototype"


def test_provider_selection_defaults_to_overpass(tmp_path) -> None:
    settings = _settings(tmp_path)

    provider = select_osm_provider(settings, RequestCache(settings), offline=True)

    assert isinstance(provider, OverpassProvider)


def test_ohsome_requires_key_and_reports_signup_url(tmp_path) -> None:
    settings = _settings(tmp_path, osm_provider="ohsome", ohsome_api_version="v2")

    with pytest.raises(OhsomeError, match="OHSOME_API_KEY"):
        select_osm_provider(settings, RequestCache(settings), offline=True)
    with pytest.raises(OhsomeError, match="https://account.heigit.org/signup"):
        OhsomeProvider(settings, RequestCache(settings), offline=True)


def test_ohsome_v1_restriction_is_explanatory(tmp_path) -> None:
    settings = _settings(tmp_path, ohsome_api_version="v1")

    with pytest.raises(OhsomeError, match="HTTP 403"):
        OhsomeProvider(settings, RequestCache(settings), offline=True)


def test_ohsome_key_is_not_exposed_in_repr_or_logs(tmp_path, caplog) -> None:
    key = "top-secret-ohsome-key"
    settings = _settings(tmp_path, ohsome_api_key=SecretStr(key))
    provider = OhsomeProvider(settings, RequestCache(settings), offline=True)
    session = Mock()
    session.get.side_effect = requests.HTTPError(key)
    provider.session = session

    with caplog.at_level(logging.DEBUG), pytest.raises(OhsomeError):
        provider.latest_snapshot()

    assert key not in repr(settings)
    assert key not in str(settings)
    assert key not in caplog.text


def test_ohsome_v2_uses_key_and_decodes_mocked_geoparquet(tmp_path) -> None:
    parquet = tmp_path / "features.parquet"
    frame = gpd.GeoDataFrame(
        {"osm_id": [19], "highway": ["path"], "geometry": [LineString([(1, 1), (2, 2)])]},
        crs="EPSG:4326",
    )
    frame.to_parquet(parquet, index=False)
    payload = parquet.read_bytes()
    response = SimpleNamespace(
        content=payload,
        raise_for_status=Mock(),
    )
    session = Mock(post=Mock(return_value=response))
    settings = _settings(
        tmp_path,
        ohsome_api_key=SecretStr("test-key"),
        ohsome_url="https://api.heigit.org/ohsome-api/v2",
        ohsome_api_version="v2",
    )
    provider = OhsomeProvider(
        settings,
        RequestCache(settings),
        session=session,
        offline=False,
    )

    result = provider.extract_tile((1, 1, 2, 2), date(2026, 1, 1))

    assert len(result.layers["roads"]) == 1
    assert result.layers["roads"][0]["road_class"] == "path"
    assert session.post.call_count == len(LAYER_NAMES)
    assert session.post.call_args.kwargs["headers"]["Authorization"] == "test-key"
    assert session.post.call_args.kwargs["json"]["time"] == "2026-01-01"


def test_geopackage_contains_each_layer_and_snapshot_metadata(tmp_path) -> None:
    settings = _settings(tmp_path)
    output_path = tmp_path / "run" / "osm_pre_event.gpkg"
    manifest = RunManifest(
        run_id="layers",
        created_utc=datetime.now(UTC),
        bbox=(85.3, 28.1, 85.42, 28.2),
        event_date=date(2026, 8, 26),
    )
    provider = SimpleNamespace(
        provider_name="overpass",
        source="OpenStreetMap via Overpass API (historical, date-pinned)",
    )
    features = {
        layer: [
            {
                "osm_type": "node",
                "osm_id": index,
                "osm_timestamp": "2026-08-24T12:00:00+00:00",
                "osm_version": 1,
                "tags": {"name": layer},
                "geometry": gpd.points_from_xy([85.31], [28.11])[0],
            }
        ]
        for index, layer in enumerate(LAYER_NAMES, start=1)
    }
    checkpoints: list[tuple[str, int]] = []

    _write_osm_layers(
        features=features,
        snapshot_date=date(2026, 8, 25),
        grid=GridSpec.from_bbox((85.3, 28.1, 85.42, 28.2)),
        output_path=output_path,
        provider=provider,
        manifest=manifest,
        settings=settings,
        checkpoint=lambda layer, count: checkpoints.append((layer, count)),
    )

    written_layers = {layer_name for layer_name, _geometry in pyogrio.list_layers(output_path)}
    layer_info = pyogrio.read_info(output_path, layer="buildings")

    assert written_layers == set(LAYER_NAMES)
    assert layer_info["layer_metadata"]["osm_snapshot_date"] == "2026-08-25"
    osm_records = [item for item in manifest.inputs if item.name.startswith("osm:")]
    assert len(osm_records) == len(LAYER_NAMES)
    assert checkpoints == [(layer, 1) for layer in LAYER_NAMES]


def test_gpkg_tags_are_json_and_raw_tag_keys_never_become_columns(tmp_path) -> None:
    settings = _settings(tmp_path)
    output_path = tmp_path / "run" / "osm_pre_event.gpkg"
    manifest = RunManifest(
        run_id="safe-columns",
        created_utc=datetime.now(UTC),
        bbox=(85.3, 28.1, 85.42, 28.2),
        event_date=date(2026, 8, 26),
    )
    provider = SimpleNamespace(
        provider_name="overpass",
        source="OpenStreetMap via Overpass API (historical, date-pinned)",
    )
    weird_tags = {
        "building": "yes",
        "FIXME": "uppercase",
        "fixme": "lowercase",
        "name:en": "Unicode Café",
        "odd key": "value with spaces",
        "☃": "snow",
    }
    features = {
        layer: [] for layer in LAYER_NAMES
    }
    features["buildings"] = [
        {
            "osm_type": "way",
            "osm_id": osm_id,
            "osm_timestamp": "2026-08-24T12:00:00+00:00",
            "osm_version": 1,
            "tags": {**weird_tags, "name": name},
            "geometry": gpd.points_from_xy([85.31], [28.11])[0],
        }
        for osm_id, name in ((101, "Feature A"), (102, "Feature B"))
    ]

    _write_osm_layers(
        features=features,
        snapshot_date=date(2026, 8, 25),
        grid=GridSpec.from_bbox((85.3, 28.1, 85.42, 28.2)),
        output_path=output_path,
        provider=provider,
        manifest=manifest,
        settings=settings,
        checkpoint=None,
    )

    buildings = pyogrio.read_dataframe(output_path, layer="buildings")
    actual_columns = list(buildings.columns)
    tags_by_name = {
        row["name"]: json.loads(row["tags_json"])
        for _, row in buildings.iterrows()
    }
    expected_columns = {
        "osm_type",
        "osm_id",
        "osm_timestamp",
        "osm_version",
        "building",
        "name",
        "amenity",
        "height",
        "building_levels",
        "tags_json",
        "geometry",
    }

    assert set(actual_columns) == expected_columns
    assert all(re.fullmatch(r"[a-z0-9_]+", column) for column in actual_columns)
    assert tags_by_name["Feature A"] == {**weird_tags, "name": "Feature A"}
    assert tags_by_name["Feature B"] == {**weird_tags, "name": "Feature B"}


def test_atomic_gpkg_failure_leaves_no_partial_package(tmp_path, monkeypatch) -> None:
    import sahayak.ingest.osm as osm_module

    settings = _settings(tmp_path)
    output_path = tmp_path / "run" / "osm_pre_event.gpkg"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"stale package")
    provider = SimpleNamespace(
        provider_name="overpass",
        source="OpenStreetMap via Overpass API (historical, date-pinned)",
    )
    features = {layer: [] for layer in LAYER_NAMES}
    original_writer = osm_module._write_gpkg_layer
    write_calls = 0

    def fail_second_write(*args, **kwargs) -> None:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 2:
            raise OSError("synthetic GeoPackage write error")
        original_writer(*args, **kwargs)

    monkeypatch.setattr(osm_module, "_write_gpkg_layer", fail_second_write)
    manifest = RunManifest(
        run_id="atomic",
        created_utc=datetime.now(UTC),
        bbox=(85.3, 28.1, 85.42, 28.2),
        event_date=date(2026, 8, 26),
    )

    with pytest.raises(OSError, match="synthetic GeoPackage write error"):
        _write_osm_layers(
            features=features,
            snapshot_date=date(2026, 8, 25),
            grid=GridSpec.from_bbox((85.3, 28.1, 85.42, 28.2)),
            output_path=output_path,
            provider=provider,
            manifest=manifest,
            settings=settings,
            checkpoint=None,
        )

    assert not output_path.exists()
    assert list(output_path.parent.glob(".osm_pre_event.*")) == []


def test_timestamp_guard_accepts_before_snapshot_and_rejects_after() -> None:
    snapshot = date(2026, 8, 25)

    verify_osm_timestamp_compliance(
        datetime.fromisoformat("2026-08-24T23:59:59+00:00"),
        snapshot,
    )
    with pytest.raises(RuleViolation, match="later than snapshot instant"):
        verify_osm_timestamp_compliance(
            datetime.fromisoformat("2026-08-25T00:00:01+00:00"),
            snapshot,
        )


def test_parser_fails_when_any_element_timestamp_is_missing() -> None:
    payload = {
        "elements": [
            {
                "type": "node",
                "id": 8,
                "lon": 85.31,
                "lat": 28.11,
                "tags": {"place": "village"},
            }
        ]
    }

    with pytest.raises(RuleViolation, match="omitted an element timestamp"):
        parse_overpass_payload(json.dumps(payload))
