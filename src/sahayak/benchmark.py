"""Cross-event benchmark summaries derived only from run facts and validation artifacts."""

from __future__ import annotations

import json
from typing import Any

from sahayak.config import Settings


def build_benchmark(settings: Settings | None = None) -> dict[str, Any]:
    """Aggregate comparable run metrics; mark unsupported measures explicitly."""
    active_settings = settings or Settings()
    rows: list[dict[str, Any]] = []
    runs_root = active_settings.resolved_runs_dir
    if not runs_root.is_dir():
        return {"status": "no_runs", "rows": rows}

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        facts_path = run_dir / "facts.json"
        manifest_path = run_dir / "manifest.json"
        validation_path = run_dir / "validate" / "validation.json"
        if not facts_path.is_file() or not manifest_path.is_file():
            continue
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.is_file()
            else None
        )
        optical = facts.get("optical", {})
        access = facts.get("access", {})
        flood = facts.get("flood", {})
        extent = (
            validation.get("extent", {}).get("by_grade", {}).get(
                "observed_landslide_extent", {}
            ).get("metrics")
            if isinstance(validation, dict)
            else None
        )
        rows.append(
            {
                "run_id": run_dir.name,
                "event_date": manifest.get("event_date"),
                "event_type": manifest.get("event_type") or "unspecified",
                "state": manifest.get("state"),
                "optical_reliability": optical.get("reliability"),
                "optical_valid_fraction": (
                    optical.get("valid_pixel_fraction", {}).get("value")
                    if isinstance(optical.get("valid_pixel_fraction"), dict)
                    else None
                ),
                "robust_cut_off_count": len(
                    access.get("robust_cut_off_settlement_ids", [])
                    if isinstance(access.get("robust_cut_off_settlement_ids"), list)
                    else []
                ),
                "not_assessable_count": access.get("not_assessable_count"),
                "radar_new_water_ha": (
                    flood.get("areas_ha_by_class", {})
                    .get("new_water", {})
                    .get("value")
                ),
                "radar_disturbed_ha": (
                    flood.get("areas_ha_by_class", {})
                    .get("disturbed_surface", {})
                    .get("value")
                ),
                "validation_iou": extent.get("iou") if isinstance(extent, dict) else None,
                "validation_status": "validated-comparison" if validation else "not-comparable",
            }
        )
    return {"status": "available", "run_count": len(rows), "rows": rows}
