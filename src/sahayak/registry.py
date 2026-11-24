"""Discover named and derived Sahayak event runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object or return an empty object for a missing input."""
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON registry input: {path}") from error
    return value if isinstance(value, dict) else {}


def _has_validation(run_dir: Path) -> bool:
    """Return whether a run has a validation artifact."""
    return (run_dir / "validate" / "validation.json").is_file()


def describe_run(run_dir: Path, *, sample: bool = False) -> dict[str, Any]:
    """Summarize one run without exposing credentials or source rasters."""
    run_id = run_dir.name
    validate_run_id(run_id)
    manifest = _read_json(run_dir / "manifest.json")
    facts = _read_json(run_dir / "facts.json")
    headline = json.loads(
        (run_dir / "headline.json").read_text(encoding="utf-8")
    ) if (run_dir / "headline.json").is_file() else []
    dashboard = _read_json(run_dir / "dashboard" / "dashboard.json")
    validation = _read_json(run_dir / "validate" / "validation.json")
    bbox = manifest.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        bbox = None
    evidence_status = "unvalidated"
    if _has_validation(run_dir):
        evidence_status = "validated-comparison"
    if facts.get("optical", {}).get("reliability") == "unverified":
        evidence_status = "coverage-unverified"
    optical_valid = facts.get("optical", {}).get("valid_pixel_fraction", {}).get("value")
    access = facts.get("access", {})
    not_assessable = access.get("not_assessable_count")
    robust_cut_off = access.get("robust_cut_off_settlements", [])
    scene_dates = facts.get("optical", {}).get("provenance", {}).get("scene_dates", {})
    scene_age_days = None
    if manifest.get("event_date") and scene_dates.get("post"):
        try:
            from datetime import date
            event = date.fromisoformat(str(manifest["event_date"])[:10])
            post = date.fromisoformat(str(scene_dates["post"])[:10])
            scene_age_days = (post - event).days
        except ValueError:
            scene_age_days = None
    return {
        "run_id": run_id,
        "name": manifest.get("name") or run_id.replace("_", " ").title(),
        "event_date": manifest.get("event_date"),
        "bbox": bbox,
        "event_type": manifest.get("event_type") or "unspecified",
        "state": manifest.get("state"),
        "district": manifest.get("district"),
        "evidence_status": evidence_status,
        "has_dashboard": bool(dashboard) or (run_dir / "dashboard" / "index.html").is_file(),
        "has_reports": (run_dir / "report").is_file() if (run_dir / "report").is_dir() else False,
        "sample": sample,
        "validation": {
            "available": bool(validation),
            "scope": validation.get("scope"),
        },
        "evidence": {
            "status": evidence_status,
            "optical_valid_fraction": optical_valid,
            "post_event_scene_age_days": scene_age_days,
            "not_assessable_settlements": (
                not_assessable if isinstance(not_assessable, (int, float)) else None
            ),
            "robust_cut_off_settlement_names": (
                robust_cut_off if isinstance(robust_cut_off, list) else []
            ),
        },
        "headline_count": len(headline) if isinstance(headline, list) else 0,
    }


def build_registry(settings: Settings | None = None) -> dict[str, Any]:
    """Build a bounded registry of available full runs and derived samples."""
    active_settings = settings or Settings()
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    runs_root = active_settings.resolved_runs_dir
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                entry = describe_run(run_dir)
            except (OSError, ValueError):
                continue
            entries.append(entry)
            seen.add(entry["run_id"])

    sample_root = active_settings.resolved_data_dir.parent / "sample_data"
    if sample_root.is_dir():
        for run_dir in sorted(sample_root.iterdir()):
            if not run_dir.is_dir() or run_dir.name in seen:
                continue
            try:
                entries.append(describe_run(run_dir, sample=True))
            except (OSError, ValueError):
                continue

    return {
        "status": "available",
        "run_count": len(entries),
        "runs": entries,
    }


def build_priorities(run_dir: Path) -> dict[str, Any]:
    """Return field-verification priorities derived from access facts."""
    facts = _read_json(run_dir / "facts.json")
    access = facts.get("access", {})
    robust_names = access.get("robust_cut_off_settlements", [])
    routes = access.get("settlement_routes", {})
    priorities: list[dict[str, Any]] = []
    if isinstance(robust_names, list):
        for name in robust_names:
            if not isinstance(name, str):
                continue
            priorities.append(
                {
                    "name": name,
                    "reason": "cut_off_in_all_scenarios",
                    "field_verification": "high",
                }
            )
    not_assessable = access.get("not_assessable_count")
    if isinstance(not_assessable, (int, float)) and not isinstance(not_assessable, bool):
        priorities.append(
            {
                "name": "network coverage gaps",
                "reason": "settlements_not_assessable",
                "count": int(not_assessable),
                "field_verification": "medium",
            }
        )
    return {
        "status": "available",
        "route_record_count": len(routes) if isinstance(routes, dict) else 0,
        "priorities": priorities,
    }
