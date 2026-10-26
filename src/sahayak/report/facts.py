"""Curated, source-linked facts for situation reports and grounded questions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from sahayak.access.hazard import merge_facts
from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id

SCENARIOS = ("strict", "default", "lenient")


def _measurement(value: Any) -> Any:
    """Return a measurement's value, preserving an explicit missing value."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return None


def _headline_item(
    fact_id: str,
    value: Any,
    unit: str,
    label: str,
    path: str,
    scenario: str | None,
    confidence: str,
) -> dict[str, Any]:
    """Build one flat headline item with a traceable facts.json source."""
    if isinstance(value, float) and not math.isfinite(value):
        value = None
    return {
        "id": fact_id,
        "value": value,
        "unit": unit,
        "label_en": label,
        "source_fact_path": path,
        "scenario": scenario,
        "confidence_note": confidence,
    }


def _materialize_settlement_routes(run_dir: Path, facts: dict[str, Any]) -> None:
    """Copy generated route measurements into facts.json for headline-only Q&A."""
    access_path = run_dir / "settlement_access.gpkg"
    if not access_path.is_file():
        return
    try:
        import geopandas as gpd
    except ImportError as error:
        raise RuntimeError("GeoPandas is required to curate settlement route facts.") from error
    settlements = gpd.read_file(access_path, layer="settlements")
    rows: dict[str, Any] = {}
    for _, row in settlements.iterrows():
        osm_id = str(row.get("osm_id", ""))
        name = row.get("name")
        record: dict[str, Any] = {
            "name": str(name) if name is not None and not _is_missing(name) else None,
            "scenarios": {},
        }
        for scenario in SCENARIOS:
            record["scenarios"][scenario] = {}
            for target in ("hospital", "town"):
                for phase in ("before", "after"):
                    for measure in ("name", "km", "minutes"):
                        column = f"{scenario}_{target}_{phase}_{measure}"
                        value = row.get(column)
                        if _is_missing(value):
                            value = None
                        elif measure == "name":
                            value = str(value)
                        else:
                            value = float(value)
                        record["scenarios"][scenario][f"{target}_{phase}_{measure}"] = value
        if osm_id:
            rows[osm_id] = record
    facts.setdefault("access", {})["settlement_routes"] = rows
    merge_facts(run_dir / "facts.json", "access", facts["access"])


def _is_missing(value: Any) -> bool:
    """Check pandas/numpy scalar missing values without accepting array truthiness."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().casefold() in {"", "nan", "none", "<na>"}:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except (ImportError, TypeError, ValueError):
        return False


def build_headline(
    facts: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Curate expected report facts, using null when a source measurement is absent."""
    headline: list[dict[str, Any]] = []

    def add(
        fact_id: str,
        value: Any,
        unit: str,
        label: str,
        path: str,
        scenario: str | None = None,
        confidence: str = "Reported from the run's mapped facts; not independently field-verified.",
    ) -> None:
        headline.append(_headline_item(fact_id, value, unit, label, path, scenario, confidence))

    for scenario in SCENARIOS:
        hazard = facts.get("hazard", {}).get("levels", {}).get(scenario, {})
        damage = facts.get("damage", {}).get("scenarios", {}).get(scenario, {})
        access = facts.get("access", {}).get("scenarios", {}).get(scenario, {})
        prefix = "facts.json"
        add(
            f"hazard_area_{scenario}",
            hazard.get("area_ha"),
            "ha",
            f"Hazard footprint area ({scenario})",
            f"{prefix}.hazard.levels.{scenario}.area_ha",
            scenario,
            "Detection footprint; not ground truth.",
        )
        for metric, label, unit in (
            ("buildings_inside", "Buildings inside detection", "buildings"),
            ("buildings_near_30m", "Buildings near detection", "buildings"),
            ("bridges_hit_within_20m", "Bridges near detection", "bridges"),
            ("settlements_affected", "Settlements affected", "settlements"),
        ):
            measurement = damage.get(metric)
            add(
                f"{metric}_{scenario}",
                _measurement(measurement),
                unit,
                f"{label} ({scenario})",
                f"{prefix}.damage.scenarios.{scenario}.{metric}.value",
                scenario,
            )
        road_classes = damage.get("road_km_hit_by_class")
        road_total = None
        if isinstance(road_classes, dict):
            values = [_measurement(value) for value in road_classes.values()]
            if all(isinstance(value, (int, float)) for value in values):
                road_total = sum(values)
        add(
            f"road_km_hit_{scenario}",
            road_total,
            "km",
            f"Road length intersecting detection ({scenario})",
            f"{prefix}.damage.scenarios.{scenario}.road_km_hit_by_class",
            scenario,
            "Detection intersections; not confirmation of road damage or closure.",
        )
        counts = access.get("counts_by_class", {})
        for access_class in ("cut_off", "degraded", "unaffected", "not_assessable"):
            add(
                f"{access_class}_{scenario}",
                _measurement(counts.get(access_class)),
                "settlements",
                f"Settlements classified {access_class.replace('_', ' ')} ({scenario})",
                f"{prefix}.access.scenarios.{scenario}.counts_by_class.{access_class}.value",
                scenario,
                "Road-network scenario estimate; not field-confirmed.",
            )

    default_access = facts.get("access", {}).get("scenarios", {}).get("default", {})
    add(
        "cut_off_default",
        _measurement(default_access.get("counts_by_class", {}).get("cut_off")),
        "settlements",
        "Settlements cut off (default)",
        "facts.json.access.scenarios.default.counts_by_class.cut_off.value",
        "default",
        "Road-network scenario estimate; not field-confirmed.",
    )
    access_root = facts.get("access", {})
    add(
        "cut_off_robust",
        len(access_root["robust_cut_off_settlement_ids"])
        if isinstance(access_root.get("robust_cut_off_settlement_ids"), list)
        else None,
        "settlements",
        "Settlements cut off in all scenarios",
        "facts.json.access.robust_cut_off_settlement_ids",
        "robust",
        "Common to strict, default, and lenient network scenarios.",
    )
    add(
        "cut_off_names_default",
        default_access.get("cut_off_settlements"),
        "settlement names",
        "Cut-off settlement names (default)",
        "facts.json.access.scenarios.default.cut_off_settlements",
        "default",
        "Names are from the pre-event OSM settlement layer and routing scenario.",
    )
    add(
        "cut_off_names_robust",
        access_root.get("robust_cut_off_settlements"),
        "settlement names",
        "Cut-off settlement names (robust)",
        "facts.json.access.robust_cut_off_settlements",
        "robust",
        "Names are common to strict, default, and lenient routing scenarios.",
    )
    add(
        "not_assessable_default",
        access_root.get("not_assessable_count"),
        "settlements",
        "Settlements not assessable (default)",
        "facts.json.access.not_assessable_count",
        "default",
        "Missing road connectivity is not classified as cut off.",
    )

    optical = facts.get("optical", {})
    add(
        "optical_reliability",
        optical.get("reliability"),
        "category",
        "Optical reliability",
        "facts.json.optical.reliability",
        None,
        "Reliability category recorded by optical control checks.",
    )
    add(
        "optical_valid_fraction",
        _measurement(optical.get("valid_pixel_fraction")),
        "fraction",
        "Valid optical pixel fraction",
        "facts.json.optical.valid_pixel_fraction.value",
        None,
        "Coverage fraction from the optical mapping stage.",
    )

    flood = facts.get("flood", {})
    areas = flood.get("areas_ha_by_class", {})
    water = _measurement(areas.get("new_water"))
    disturbed = _measurement(areas.get("disturbed_surface"))
    if isinstance(water, (int, float)) and isinstance(disturbed, (int, float)):
        radar_finding = (
            "No new-water or disturbed-surface detections in the radar classification."
            if water == 0 and disturbed == 0
            else "Radar classification contains mapped new-water or disturbed-surface detections."
        )
    else:
        radar_finding = None
    add(
        "radar_finding",
        radar_finding,
        "finding",
        "Radar finding",
        "facts.json.flood.areas_ha_by_class",
        None,
        "Derived only from the radar mapped-class area measurements.",
    )

    dates = manifest.get("event_date")
    add(
        "event_date",
        dates,
        "date",
        "Event date",
        "manifest.json.event_date",
        None,
        "Recorded in the run manifest.",
    )
    optical_dates = optical.get("provenance", {}).get("scene_dates", {})
    for role in ("pre", "post"):
        add(
            f"optical_{role}_date",
            optical_dates.get(role),
            "date",
            f"Optical {role}-event acquisition date",
            f"facts.json.optical.provenance.scene_dates.{role}",
            None,
            "Acquisition datetime recorded by the optical stage.",
        )
    bbox = manifest.get("bbox")
    add(
        "focus_bbox",
        bbox if isinstance(bbox, list) else None,
        "min_lon,min_lat,max_lon,max_lat",
        "Focus area bounding box",
        "manifest.json.bbox",
        None,
        "Geographic extent recorded in the run manifest.",
    )
    sources = sorted(
        {
            str(record["source"])
            for record in manifest.get("inputs", [])
            if (
                isinstance(record, dict)
                and record.get("role") == "input"
                and record.get("source")
            )
        }
    )
    add(
        "data_sources",
        sources or None,
        "sources",
        "Recorded data sources",
        "manifest.json.inputs[].source",
        None,
        "Source descriptions are copied from the run manifest.",
    )
    add(
        "report_attributions",
        [
            "Contains modified Copernicus Sentinel data 2026.",
            (
                "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and "
                "© Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by "
                "the European Union and ESA; all rights reserved."
            ),
            (
                "© OpenStreetMap contributors (ODbL). "
                "Any published modified OSM data stays under ODbL."
            ),
            "Educational prototype, not an operational tool.",
        ],
        "attribution",
        "Required data and responsibility attributions",
        "docs/data-rules.md#attribution",
        None,
        "Required project attribution text; it does not identify an analytical input.",
    )

    for osm_id, route in access_root.get("settlement_routes", {}).items():
        settlement_name = route.get("name") or "unnamed settlement"
        hospital_route = route.get("scenarios", {}).get("default", {})
        for measure, unit in (("name", "facility"), ("km", "km"), ("minutes", "minutes")):
            value = hospital_route.get(f"hospital_before_{measure}")
            add(
                f"hospital_route_{osm_id}_{measure}",
                value,
                unit,
                f"Nearest hospital {measure} for {settlement_name}",
                (
                    "facts.json.access.settlement_routes."
                    f"{osm_id}.scenarios.default.hospital_before_{measure}"
                ),
                "default",
                (
                    "Nearest pre-event road route from mapped OSM network; "
                    "not a verified travel route."
                ),
            )
    return headline


def write_headline(run_id: str, settings: Settings | None = None) -> Path:
    """Write a curated headline.json for an existing mapped run."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    facts_path = run_dir / "facts.json"
    manifest_path = run_dir / "manifest.json"
    if not facts_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("Run requires facts.json and manifest.json.")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(facts, dict) or not isinstance(manifest, dict):
        raise ValueError("Run facts and manifest must be JSON objects.")
    _materialize_settlement_routes(run_dir, facts)
    headline = build_headline(facts, manifest)
    output_path = run_dir / "headline.json"
    output_path.write_text(
        json.dumps(headline, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output_path
