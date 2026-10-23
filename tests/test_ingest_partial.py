"""Tests for durable metadata after a failed ingestion stage."""

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

import sahayak.ingest.run as run_module
from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.provenance import RunManifest


def test_manifest_checkpoint_survives_later_stage_failure(tmp_path, monkeypatch) -> None:
    settings = Settings(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", _env_file=None)

    def fake_dem(
        *,
        grid: GridSpec,
        settings: Settings,
        cache: RequestCache,
        offline: bool,
        output_dir: Path,
        manifest: RunManifest,
        checkpoint: Callable[[str], None] | None = None,
    ) -> dict[str, object]:
        del grid, cache, offline
        artifact = output_dir / "dem.tif"
        artifact.write_bytes(b"synthetic dem")
        manifest.add_input(
            name="dem.tif",
            role="input",
            source="Copernicus DEM GLO-30 via AWS Open Data",
            license="Copernicus DEM Public licence",
            retrieved_utc=datetime.now(UTC),
            path=artifact,
            settings=settings,
        )
        if checkpoint is not None:
            checkpoint("dem.tif")
        return {"dem_path": artifact}

    def fail_osm(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise RuntimeError("simulated OSM failure")

    monkeypatch.setattr(run_module, "ingest_dem", fake_dem)
    monkeypatch.setattr(run_module, "ingest_osm", fail_osm)

    with pytest.raises(RuntimeError, match="simulated OSM failure"):
        run_module.run_ingest(
            bbox=(85.3, 28.1, 85.42, 28.2),
            event_date=date(2026, 8, 26),
            run_id="partial",
            skip={"s1", "s2"},
            settings=settings,
        )

    run_dir = settings.resolved_runs_dir / "partial"
    manifest = RunManifest.read(run_dir / "manifest.json")
    selection = (run_dir / "selection.json").read_text(encoding="utf-8")

    assert (run_dir / "dem.tif").is_file()
    assert any(item.name == "dem.tif" for item in manifest.inputs)
    assert '"dem.tif"' in selection
