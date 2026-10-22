"""Tests for manifest registration of ingestion data sources."""

from datetime import UTC, date, datetime

import pytest

from sahayak.config import Settings
from sahayak.provenance import ForbiddenInputError, RunManifest


def test_manifest_records_ingestion_source_and_file_hash(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    data_file = tmp_path / "dem.tif"
    data_file.write_bytes(b"synthetic raster bytes")
    manifest = RunManifest(
        run_id="test",
        created_utc=datetime(2026, 8, 26, tzinfo=UTC),
        bbox=(85.3, 28.1, 85.42, 28.2),
        event_date=date(2026, 8, 26),
    )

    record = manifest.add_input(
        name="dem.tif",
        role="input",
        source="Copernicus DEM GLO-30 via AWS Open Data",
        license="Copernicus DEM Public licence",
        retrieved_utc=datetime(2026, 8, 25, tzinfo=UTC),
        path=data_file,
        settings=settings,
    )

    assert record.source == "Copernicus DEM GLO-30 via AWS Open Data"
    assert record.license == "Copernicus DEM Public licence"
    assert record.sha256 is not None
    assert manifest.inputs == [record]


def test_manifest_does_not_record_a_forbidden_source_name(tmp_path) -> None:
    manifest = RunManifest(
        run_id="test",
        created_utc=datetime(2026, 8, 26, tzinfo=UTC),
        bbox=(85.3, 28.1, 85.42, 28.2),
        event_date=date(2026, 8, 26),
    )

    with pytest.raises(ForbiddenInputError, match="Forbidden data source"):
        manifest.add_input(
            name="UNOSAT validation overlay",
            role="input",
            source="OpenStreetMap via ohsome API",
            license="ODbL",
            retrieved_utc=datetime(2026, 8, 25, tzinfo=UTC),
        )
    assert manifest.inputs == []
