"""Wide-area, historical OSM network extraction for access analysis."""

from __future__ import annotations

import json
import tempfile
import time
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pyogrio
import requests

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.ingest.osm import (
    LAYER_NAMES,
    OSM_ATTRIBUTION,
    OSM_LICENSE,
    OSMProviderError,
    OverpassProvider,
    _frame_from_records,
    _overpass_result_layers,
    tile_bboxes,
    verify_osm_timestamp_compliance,
)
from sahayak.ingest.run import validate_run_id
from sahayak.provenance import RuleViolation, RunManifest, assert_osm_snapshot_valid

NETWORK_LAYERS = ("roads", "bridges", "health_facilities", "settlements")
NETWORK_TILE_DEGREES = 0.1


def estimate_network_requests(
    bbox: tuple[float, float, float, float],
) -> int:
    """Return the number of Overpass tile queries before retry attempts."""
    return len(tile_bboxes(bbox, max_tile_degrees=NETWORK_TILE_DEGREES))


def _read_snapshot_date(run_dir: Path, selection: dict[str, Any]) -> date:
    """Use the run selection date or verified metadata on its existing OSM layers."""
    selected = selection.get("osm_snapshot_date")
    source_path = run_dir / "osm_pre_event.gpkg"
    if not source_path.is_file():
        raise FileNotFoundError(f"Run is missing its original OSM snapshot: {source_path.name}")
    snapshots = set()
    for layer in LAYER_NAMES:
        info = pyogrio.read_info(source_path, layer=layer)
        metadata = info.get("layer_metadata") or {}
        value = metadata.get("osm_snapshot_date")
        if not value:
            raise RuleViolation(f"OSM layer {layer!r} has no snapshot-date metadata.")
        snapshots.add(str(value))
    if len(snapshots) != 1:
        raise RuleViolation("Existing OSM layers do not share one snapshot date.")
    try:
        stored_date = date.fromisoformat(snapshots.pop())
    except ValueError as error:
        raise RuleViolation("Existing OSM layer metadata has an invalid snapshot date.") from error
    if selected:
        try:
            selected_date = date.fromisoformat(str(selected))
        except ValueError as error:
            raise RuleViolation("Run selection has an invalid OSM snapshot date.") from error
        if selected_date != stored_date:
            raise RuleViolation(
                "Run selection OSM snapshot date does not match the original OSM layers."
            )
    return stored_date


def ingest_network(
    *,
    run_id: str,
    bbox: tuple[float, float, float, float],
    settings: Settings | None = None,
    offline: bool = False,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Extract date-pinned OSM roads, bridges, facilities, and settlements."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    selection_path = run_dir / "selection.json"
    manifest_path = run_dir / "manifest.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Run {run_id!r} lacks selection.json or manifest.json.")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError(f"Run selection metadata for {run_id!r} is not an object.")
    manifest = RunManifest.read(manifest_path)
    if manifest.run_id != run_id:
        raise ValueError("Run manifest ID does not match the requested run.")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError(
            "Network bbox coordinates must be ordered valid longitude/latitude bounds."
        )
    snapshot_date = _read_snapshot_date(run_dir, selection)
    assert_osm_snapshot_valid(snapshot_date, manifest.event_date)
    if active_settings.osm_provider.strip().lower() != "overpass":
        raise OSMProviderError(
            "Network extraction requires OSM_PROVIDER=overpass to preserve the existing "
            "date-pinned Overpass provider and cache."
        )

    grid_data = selection.get("grid")
    if not isinstance(grid_data, dict) or "resolution" not in grid_data:
        raise ValueError("Run selection has no valid common-grid metadata.")
    grid = GridSpec.from_bbox(manifest.bbox, resolution=float(grid_data["resolution"]))
    output_path = run_dir / "network_osm_pre_event.gpkg"
    expected_requests = estimate_network_requests(bbox)
    selection.update(
        {
            "network_osm_bbox": list(bbox),
            "network_osm_snapshot_date": snapshot_date.isoformat(),
            "network_overpass_estimated_tile_requests": expected_requests,
            "network_overpass_request_count": 0,
            "network_output": output_path.name,
        }
    )

    def persist_checkpoint() -> None:
        """Persist extraction progress in run selection and provenance."""
        selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        manifest.inputs = [item for item in manifest.inputs if item.name != "selection.json"]
        manifest.add_input(
            name="selection.json",
            role="input",
            source="Sahayak run selection metadata",
            license="Project metadata",
            retrieved_utc=datetime.now(UTC),
            path=selection_path,
            settings=active_settings,
        )
        manifest.write(manifest_path)

    persist_checkpoint()

    provider = OverpassProvider(
        active_settings,
        RequestCache(active_settings),
        offline=offline,
        session=session,
    )
    started = time.monotonic()
    features, _skipped_relations = _overpass_result_layers(
        provider,
        bbox,
        snapshot_date,
        max_tile_degrees=NETWORK_TILE_DEGREES,
    )
    verify_osm_timestamp_compliance(provider.max_element_timestamp, snapshot_date)
    counts = {layer: len(features[layer]) for layer in NETWORK_LAYERS}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".network_osm.", dir=run_dir) as temp_dir:
        temporary_path = Path(temp_dir) / output_path.name
        for index, layer in enumerate(NETWORK_LAYERS):
            frame = _frame_from_records(features[layer], grid.crs, layer)
            pyogrio.write_dataframe(
                frame,
                temporary_path,
                layer=layer,
                driver="GPKG",
                append=index > 0,
                layer_metadata={"osm_snapshot_date": snapshot_date.isoformat()},
            )
        temporary_path.replace(output_path)

    max_timestamp = provider.max_element_timestamp
    elapsed_seconds = time.monotonic() - started
    manifest.inputs = [item for item in manifest.inputs if item.name != output_path.name]
    manifest.add_input(
        name=output_path.name,
        role="output",
        source="Sahayak pre-event network OSM analysis",
        license=OSM_LICENSE,
        retrieved_utc=datetime.now(UTC),
        path=output_path,
        settings=active_settings,
    )
    selection.update(
        {
            "network_osm_counts": counts,
            "network_max_element_timestamp": (
                max_timestamp.isoformat() if max_timestamp is not None else None
            ),
            "network_element_timestamp_verification": (
                "all OSM element timestamps verified against "
                f"{snapshot_date.isoformat()}"
            ),
            "network_overpass_request_count": provider.request_count,
            "network_overpass_elapsed_seconds": elapsed_seconds,
        }
    )
    persist_checkpoint()
    return {
        "path": output_path,
        "counts": counts,
        "estimated_requests": expected_requests,
        "request_count": provider.request_count,
        "elapsed_seconds": elapsed_seconds,
        "max_element_timestamp": max_timestamp,
        "snapshot_date": snapshot_date,
        "bbox": bbox,
        "attribution": OSM_ATTRIBUTION,
    }
