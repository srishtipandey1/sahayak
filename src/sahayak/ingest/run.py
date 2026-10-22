"""Orchestrate the implemented common-grid, DEM, and OSM ingestion stages."""

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.dem import DEM_ATTRIBUTION, DEM_SOURCE, ingest_dem
from sahayak.ingest.grid import GridSpec
from sahayak.ingest.osm import OSM_ATTRIBUTION, ingest_osm
from sahayak.provenance import RunManifest

OSM_ATTRIBUTION_TEXT = (
    f"{OSM_ATTRIBUTION} Any published modified OSM data stays under ODbL."
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUPPORTED_STAGES = {"s1", "s2", "dem", "osm"}


def validate_run_id(run_id: str) -> None:
    """Require a run identifier that cannot escape the configured run directory."""
    if not RUN_ID_PATTERN.fullmatch(run_id) or run_id in {".", ".."}:
        raise ValueError(
            "Run ID may contain only letters, numbers, dots, underscores, and hyphens."
        )


def run_ingest(
    *,
    bbox: tuple[float, float, float, float],
    event_date: date,
    run_id: str,
    osm_snapshot: date | None = None,
    skip: set[str] | None = None,
    offline: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run configured DEM and OSM stages and persist selection and provenance."""
    validate_run_id(run_id)
    selected_skips = skip if skip is not None else {"s1", "s2"}
    unknown_skips = selected_skips - SUPPORTED_STAGES
    if unknown_skips:
        raise ValueError(f"Unsupported skip value(s): {', '.join(sorted(unknown_skips))}.")
    not_skipped_science = {"s1", "s2"} - selected_skips
    if not_skipped_science:
        stages = ", ".join(sorted(not_skipped_science))
        raise ValueError(f"Radar/optical ingestion is not implemented yet; skip: {stages}.")

    active_settings = settings or Settings(_env_file=None)
    grid = GridSpec.from_bbox(bbox)
    cache = RequestCache(active_settings)
    output_dir = active_settings.resolved_runs_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        run_id=run_id,
        created_utc=datetime.now(UTC),
        bbox=bbox,
        event_date=event_date,
    )

    dem_result: dict[str, object] | None = None
    if "dem" not in selected_skips:
        dem_result = ingest_dem(
            grid=grid,
            settings=active_settings,
            cache=cache,
            offline=offline,
            output_dir=output_dir,
            manifest=manifest,
        )

    osm_result: dict[str, object] | None = None
    if "osm" not in selected_skips:
        osm_result = ingest_osm(
            bbox=bbox,
            event_date=event_date,
            requested_snapshot=osm_snapshot,
            grid=grid,
            settings=active_settings,
            cache=cache,
            offline=offline,
            output_dir=output_dir,
            manifest=manifest,
        )

    selection: dict[str, object] = {
        "grid": grid.to_dict(),
        "dem_source": None if "dem" in selected_skips else DEM_SOURCE,
        "osm_snapshot_date": (
            osm_result["snapshot_date"].isoformat() if osm_result is not None else None
        ),
        "ohsome_api_version": osm_result["api_version"] if osm_result is not None else None,
        "attributions": [
            attribution
            for attribution, enabled in (
                (DEM_ATTRIBUTION, dem_result is not None),
                (OSM_ATTRIBUTION_TEXT, osm_result is not None),
            )
            if enabled
        ],
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    manifest.add_input(
        name="selection.json",
        role="input",
        source="Sahayak run selection metadata",
        license="Project metadata",
        retrieved_utc=datetime.now(UTC),
        path=selection_path,
        settings=active_settings,
    )
    manifest_path = output_dir / "manifest.json"
    manifest.write(manifest_path)
    return {
        "run_id": run_id,
        "output_dir": output_dir,
        "grid": grid,
        "dem": dem_result,
        "osm": osm_result,
        "manifest_path": manifest_path,
        "selection_path": selection_path,
    }


def run_file_sizes(directory: Path) -> dict[str, int]:
    """Return byte sizes for all files recursively under a run folder."""
    return {
        path.relative_to(directory).as_posix(): path.stat().st_size
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def format_ingest_summary(result: dict[str, Any]) -> str:
    """Format a concise human summary including the captured raster and vector facts."""
    grid: GridSpec = result["grid"]
    lines = [
        f"Run: {result['run_id']}",
        f"Grid: {grid.crs}, {grid.height} rows x {grid.width} columns at {grid.resolution:g} m",
    ]
    dem_result = result["dem"]
    if dem_result is not None:
        lines.append(
            f"DEM: shape {dem_result['shape']}, elevation range "
            f"{dem_result['minimum_elevation_m']:.2f} to "
            f"{dem_result['maximum_elevation_m']:.2f} m"
        )
    else:
        lines.append("DEM: skipped")

    osm_result = result["osm"]
    if osm_result is not None:
        lines.append(
            f"OSM: ohsome {osm_result['api_version']}, snapshot "
            f"{osm_result['snapshot_date'].isoformat()}"
        )
        lines.extend(
            f"  {layer}: {count}" for layer, count in osm_result["counts"].items()
        )
        lines.append(
            "Syabrubesi place: "
            + (
                "present in downloaded place data"
                if osm_result["syabrubesi_present"]
                else "not found"
            )
        )
    else:
        lines.append("OSM: skipped")

    lines.append("Run files:")
    lines.extend(
        f"  {name}: {size} bytes"
        for name, size in run_file_sizes(result["output_dir"]).items()
    )
    return "\n".join(lines)
