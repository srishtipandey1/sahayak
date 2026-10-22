"""Tests for provenance records and project data-rule guards."""

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from sahayak.config import Settings
from sahayak.provenance import (
    ForbiddenInputError,
    RuleViolation,
    RunManifest,
    SceneTrack,
    assert_allowed_input,
    assert_allowed_source,
    assert_osm_snapshot_valid,
    assert_same_orbit_track,
    sha256_file,
)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", _env_file=None)


def test_reference_paths_rejected_after_parent_resolution(settings: Settings) -> None:
    candidate = settings.data_dir / "processed" / ".." / "reference" / "validation.tif"

    with pytest.raises(ForbiddenInputError, match="Reference-only"):
        assert_allowed_input(candidate, settings)


def test_reference_paths_rejected_with_different_casing(settings: Settings) -> None:
    candidate = Path(str(settings.reference_dir).replace("reference", "REFERENCE")) / "map.tif"

    with pytest.raises(ForbiddenInputError, match="Reference-only"):
        assert_allowed_input(candidate, settings)


def test_non_reference_path_is_allowed(settings: Settings, tmp_path: Path) -> None:
    assert_allowed_input(tmp_path / "data" / "raw" / "scene.tif", settings)


@pytest.mark.parametrize("source", ["Copernicus EMS EMSR927", "UNOSAT product"])
def test_forbidden_source_strings_are_rejected(source: str) -> None:
    with pytest.raises(ForbiddenInputError, match="Forbidden data source"):
        assert_allowed_source(source)


@pytest.mark.parametrize(
    "source",
    ["Copernicus Data Space Sentinel-1 GRD", "OpenStreetMap via ohsome"],
)
def test_allowed_sources_pass(source: str) -> None:
    assert_allowed_source(source)


def test_osm_snapshot_must_precede_event() -> None:
    assert_osm_snapshot_valid("2026-08-25", "2026-08-26")
    with pytest.raises(RuleViolation, match="earlier"):
        assert_osm_snapshot_valid("2026-08-26", "2026-08-26")
    with pytest.raises(RuleViolation, match="earlier"):
        assert_osm_snapshot_valid("2026-08-27", "2026-08-26")


def test_osm_snapshot_rejects_invalid_dates() -> None:
    with pytest.raises(RuleViolation, match="Invalid date"):
        assert_osm_snapshot_valid("not-a-date", "2026-08-26")


def test_radar_pair_requires_same_relative_orbit() -> None:
    scene_a = SceneTrack(relative_orbit=42, pass_direction="ASCENDING")
    scene_b = SceneTrack(relative_orbit=43, pass_direction="ASCENDING")

    with pytest.raises(RuleViolation, match="relative orbit"):
        assert_same_orbit_track(scene_a, scene_b)


def test_radar_pair_requires_same_pass_direction() -> None:
    scene_a = SceneTrack(relative_orbit=42, pass_direction="ASCENDING")
    scene_b = SceneTrack(relative_orbit=42, pass_direction="DESCENDING")

    with pytest.raises(RuleViolation, match="pass direction"):
        assert_same_orbit_track(scene_a, scene_b)


def test_manifest_write_read_round_trip_and_sha256(tmp_path: Path, settings: Settings) -> None:
    input_file = tmp_path / "scene.bin"
    input_file.write_bytes(b"synthetic scene data")
    manifest = RunManifest(
        run_id="test-run",
        created_utc=datetime(2026, 8, 26, tzinfo=UTC),
        bbox=(85.0, 28.0, 85.5, 28.5),
        event_date=date(2026, 8, 26),
    )
    expected_digest = sha256_file(input_file)

    record = manifest.add_input(
        name="scene.bin",
        role="input",
        source="Copernicus Data Space Sentinel-1 GRD",
        license="Copernicus",
        retrieved_utc=datetime(2026, 8, 27, tzinfo=UTC),
        path=input_file,
        settings=settings,
    )
    manifest_path = tmp_path / "runs" / "test-run" / "manifest.json"
    manifest.write(manifest_path)
    restored = RunManifest.read(manifest_path)

    assert record.sha256 == expected_digest
    assert restored == manifest
