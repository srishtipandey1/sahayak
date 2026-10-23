"""Orchestrate common-grid, DEM, OSM, and Sentinel-1 ingestion stages."""

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
from sahayak.ingest.s1 import S1_ATTRIBUTION, ingest_s1
from sahayak.provenance import RunManifest

OSM_ATTRIBUTION_TEXT = OSM_ATTRIBUTION
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
    s1_baseline_count: int = 3,
    skip: set[str] | None = None,
    offline: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run configured ingestion stages and persist selection and provenance."""
    validate_run_id(run_id)
    selected_skips = skip if skip is not None else {"s2"}
    unknown_skips = selected_skips - SUPPORTED_STAGES
    if unknown_skips:
        raise ValueError(f"Unsupported skip value(s): {', '.join(sorted(unknown_skips))}.")
    if "s2" not in selected_skips:
        raise ValueError("Sentinel-2 ingestion is not implemented yet; add s2 to --skip.")
    if not 1 <= s1_baseline_count <= 3:
        raise ValueError("S1 baseline count must be between 1 and 3, including the chosen scene.")

    active_settings = settings or Settings()
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
    selection: dict[str, object] = {
        "grid": grid.to_dict(),
        "dem_source": None if "dem" in selected_skips else DEM_SOURCE,
        "osm_provider": active_settings.osm_provider if "osm" not in selected_skips else None,
        "osm_endpoint": (
            active_settings.overpass_url
            if active_settings.osm_provider.lower() == "overpass"
            else active_settings.ohsome_url
        )
        if "osm" not in selected_skips
        else None,
        "osm_snapshot_date": None,
        "osm_base_timestamp": None,
        "osm_base_timestamps": [],
        "osm_counts": {},
        "osm_skipped_relations": {},
        "overpass_request_count": 0,
        "overpass_elapsed_seconds": 0.0,
        "s1": None,
        "attributions": [
            attribution
            for attribution, enabled in (
                (DEM_ATTRIBUTION, "dem" not in selected_skips),
                (OSM_ATTRIBUTION_TEXT, "osm" not in selected_skips),
                (S1_ATTRIBUTION, "s1" not in selected_skips),
            )
            if enabled
        ],
    }
    selection_path = output_dir / "selection.json"
    manifest_path = output_dir / "manifest.json"

    def persist_checkpoint(updates: dict[str, object] | None = None) -> None:
        """Persist selection and manifest after each completed pipeline artifact."""
        if updates:
            selection.update(updates)
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

    dem_result: dict[str, object] | None = None
    if "dem" not in selected_skips:
        dem_result = ingest_dem(
            grid=grid,
            settings=active_settings,
            cache=cache,
            offline=offline,
            output_dir=output_dir,
            manifest=manifest,
            checkpoint=lambda artifact: persist_checkpoint(
                {"completed_artifacts": [*selection.get("completed_artifacts", []), artifact]}
            ),
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
            checkpoint=persist_checkpoint,
        )

    if osm_result is not None:
        selection.update(
            {
                "osm_provider": osm_result["provider"],
                "osm_endpoint": osm_result["endpoint"],
                "osm_snapshot_date": osm_result["snapshot_date"].isoformat(),
                "osm_base_timestamp": osm_result["osm_base_timestamp"],
                "osm_base_timestamps": osm_result["osm_base_timestamps"],
                "osm_counts": osm_result["counts"],
                "osm_skipped_relations": osm_result["skipped_relations"],
                "element_geometry_policy": osm_result["element_geometry_policy"],
                "max_element_timestamp": (
                    osm_result["max_element_timestamp"].isoformat()
                    if osm_result["max_element_timestamp"] is not None
                    else None
                ),
                "per_layer_max_timestamps": {
                    layer: timestamp.isoformat() if timestamp is not None else None
                    for layer, timestamp in osm_result["per_layer_max_timestamps"].items()
                },
                "osm_timestamp_verification": osm_result["osm_timestamp_verification"],
                "overpass_request_count": osm_result["request_count"],
                "overpass_elapsed_seconds": osm_result["elapsed_seconds"],
                "named_settlements": osm_result["named_settlements"],
            }
        )

    s1_result: dict[str, object] | None = None
    if "s1" not in selected_skips:
        s1_result = ingest_s1(
            bbox=bbox,
            event_date=event_date,
            baseline_count=s1_baseline_count,
            grid=grid,
            settings=active_settings,
            cache=cache,
            offline=offline,
            output_dir=output_dir,
            manifest=manifest,
            checkpoint=persist_checkpoint,
        )
        selection["s1"] = s1_result["selection"]
    persist_checkpoint()
    return {
        "run_id": run_id,
        "event_date": event_date,
        "output_dir": output_dir,
        "grid": grid,
        "dem": dem_result,
        "osm": osm_result,
        "s1": s1_result,
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
            f"OSM: {osm_result['provider']} via {osm_result['endpoint']}, snapshot "
            f"{osm_result['snapshot_date'].isoformat()}"
        )
        if osm_result["osm_base_timestamp"]:
            lines.append(f"  osm_base: {osm_result['osm_base_timestamp']}")
        lines.append(f"  max element timestamp: {osm_result['max_element_timestamp']}")
        lines.append(f"  max timestamps by layer: {osm_result['per_layer_max_timestamps']}")
        if osm_result["osm_timestamp_verification"]:
            lines.append(f"  {osm_result['osm_timestamp_verification']}")
        if osm_result["provider"] == "overpass":
            lines.append(
                f"  requests: {osm_result['request_count']}; "
                f"elapsed: {osm_result['elapsed_seconds']:.1f} s"
            )
        lines.extend(
            f"  {layer}: {count}" for layer, count in osm_result["counts"].items()
        )
        lines.extend(
            f"  {layer} skipped relations: {count}"
            for layer, count in osm_result["skipped_relations"].items()
        )
        lines.append(f"  named settlements: {osm_result['named_settlements']}")
    else:
        lines.append("OSM: skipped")

    s1_result = result.get("s1")
    if s1_result is None:
        lines.append("Sentinel-1: skipped")
    else:
        s1_selection = s1_result["selection"]
        pair = s1_selection["pair"]
        lines.append(
            "Sentinel-1: "
            f"{pair['pre']['platform']} {pair['pre']['datetime']} -> "
            f"{pair['post']['platform']} {pair['post']['datetime']}; "
            f"relative orbit {pair['pre']['relative_orbit']} {pair['pre']['direction']}; "
            f"gap {pair['gap_days']:.2f} days"
        )
        lines.append(
            "  coverage: "
            f"pre {pair['pre']['coverage_fraction']:.3%}, "
            f"post {pair['post']['coverage_fraction']:.3%}"
        )
        lines.append("  all acquisitions by relative orbit and direction:")
        track_groups = s1_selection["track_groups"]
        if track_groups:
            for group in track_groups:
                lines.append(
                    f"    relative orbit {group['relative_orbit']} {group['direction']}: "
                    f"{group['before_event']} before, "
                    f"{group['on_or_after_event']} on/after"
                )
        else:
            lines.append("    none")
        for acquisition in s1_selection["all_acquisitions"]:
            phase = (
                "before"
                if acquisition["datetime"][:10] < result["event_date"].isoformat()
                else "on/after"
            )
            lines.append(
                f"    {acquisition['platform']} {acquisition['datetime']}; "
                f"orbit {acquisition['relative_orbit']} {acquisition['direction']}; "
                f"{phase}; coverage {acquisition['coverage_fraction']:.3%}"
            )
        exclusions = s1_selection["excluded_acquisitions"]
        lines.append("  excluded acquisitions:")
        if exclusions:
            lines.extend(
                f"    {item['platform']} {item['datetime']} ({item['item_id']}): "
                f"{item['reason']}"
                for item in exclusions
            )
        else:
            lines.append("    none")
        orbit_check = s1_selection["absolute_orbit_cross_check"]
        lines.append(f"  absolute-orbit cross-check: {orbit_check['reason']}")
        alternatives = s1_selection["ranked_alternatives"]
        if alternatives:
            lines.append("  ranked alternatives:")
            for index, alternative in enumerate(alternatives, start=1):
                lines.append(
                    f"    {index}. {alternative['pre']['platform']} "
                    f"{alternative['pre']['datetime']} -> "
                    f"{alternative['post']['platform']} {alternative['post']['datetime']}; "
                    f"orbit {alternative['pre']['relative_orbit']} "
                    f"{alternative['pre']['direction']}; "
                    f"gap {alternative['gap_days']:.2f} days; "
                    f"coverage {alternative['pre']['coverage_fraction']:.3%}/"
                    f"{alternative['post']['coverage_fraction']:.3%}"
                )
        else:
            lines.append("  ranked alternatives: none")
        lines.append(
            f"  requests: Catalog {s1_result['catalog_request_count']}, "
            f"Process {s1_result['process_request_count']}"
        )
        for filename, item in s1_result["files"].items():
            lines.append(f"  {filename}: {item['percent_valid_pixels']:.2f}% valid pixels")
            for polarization, stats in item["statistics_db"].items():
                lines.append(
                    f"    {polarization} dB min/median/max: "
                    f"{stats['minimum_db']:.2f}/{stats['median_db']:.2f}/"
                    f"{stats['maximum_db']:.2f}"
                )
        processing_units = s1_result["processing_units"]
        if processing_units is None:
            lines.append("  processing units: not returned")
        else:
            lines.append(f"  processing units: {processing_units}")

    lines.append("Run files:")
    lines.extend(
        f"  {name}: {size} bytes"
        for name, size in run_file_sizes(result["output_dir"]).items()
    )
    return "\n".join(lines)
