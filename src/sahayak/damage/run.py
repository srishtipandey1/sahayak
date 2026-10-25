"""Feature-level exposure estimates for buildings, roads, bridges, and settlements."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box
from shapely.ops import unary_union

from sahayak.access.hazard import HAZARD_LEVELS, build_hazards, merge_facts, read_hazards
from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id
from sahayak.provenance import RunManifest


def _focus_geometry(manifest: RunManifest, crs: Any) -> Any:
    """Transform the run's focus bbox into a source layer CRS."""
    return gpd.GeoSeries([box(*manifest.bbox)], crs="EPSG:4326").to_crs(crs).iloc[0]


def _features_in_focus(frame: gpd.GeoDataFrame, focus: Any) -> gpd.GeoDataFrame:
    """Clip source features to the run focus bbox while preserving their attributes."""
    if frame.crs is None:
        raise ValueError("An OSM exposure layer has no CRS.")
    projected = frame.to_crs(frame.crs)
    selected = projected.loc[projected.geometry.intersects(focus)].copy()
    selected.geometry = selected.geometry.intersection(focus)
    return selected.loc[~selected.geometry.is_empty].copy()


def _write_scenario_layers(
    *,
    path: Path,
    layer_frames: dict[str, gpd.GeoDataFrame],
) -> None:
    """Write scenario-tagged exposure vectors to a GeoPackage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".exposure.", dir=path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / path.name
        for index, (layer, frame) in enumerate(layer_frames.items()):
            pyogrio.write_dataframe(
                frame,
                temporary_path,
                layer=layer,
                driver="GPKG",
                append=index > 0,
            )
        temporary_path.replace(path)


def run_damage(
    *,
    run_id: str,
    sources: tuple[str, ...] = ("optical",),
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Calculate per-scenario exposure without treating footprints as ground truth."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    manifest_path = run_dir / "manifest.json"
    osm_path = run_dir / "osm_pre_event.gpkg"
    if not manifest_path.is_file() or not osm_path.is_file():
        raise FileNotFoundError("Damage analysis requires run manifest and osm_pre_event.gpkg.")
    manifest = RunManifest.read(manifest_path)
    if manifest.run_id != run_id:
        raise ValueError("Run manifest ID does not match the requested run.")
    hazard_result = build_hazards(run_id=run_id, sources=sources, settings=active_settings)
    hazards = read_hazards(hazard_result["path"])
    source_layers = {
        layer: gpd.read_file(osm_path, layer=layer)
        for layer in ("buildings", "roads", "bridges", "settlements")
    }
    focus_layers = {
        layer: _features_in_focus(frame, _focus_geometry(manifest, frame.crs))
        for layer, frame in source_layers.items()
    }
    buildings = focus_layers["buildings"]
    roads = focus_layers["roads"]
    bridges = focus_layers["bridges"]
    settlements = focus_layers["settlements"]
    if roads.crs is None:
        raise ValueError("Road exposure layer has no CRS.")
    hazard_crs = gpd.read_file(hazard_result["path"], layer="hazards").crs
    if hazard_crs is None:
        raise ValueError("Hazard footprints have no CRS.")

    scenario_facts: dict[str, Any] = {}
    layer_frames: dict[str, list[gpd.GeoDataFrame]] = {
        "building_exposure": [],
        "road_exposure": [],
        "bridge_exposure": [],
        "settlement_exposure": [],
    }
    for level in HAZARD_LEVELS:
        hazard = hazards[level]
        if hazard is not None:
            hazard = gpd.GeoSeries([hazard], crs=hazard_crs).to_crs(buildings.crs).iloc[0]
        if hazard is None:
            hazard_union = None
        else:
            hazard_union = unary_union([hazard])

        building_inside: list[bool] = []
        building_near: list[bool] = []
        for geometry in buildings.geometry:
            inside = bool(hazard_union is not None and geometry.intersects(hazard_union))
            near = bool(
                hazard_union is not None
                and not inside
                and geometry.distance(hazard_union) <= 30.0
            )
            building_inside.append(inside)
            building_near.append(near)
        building_exposure = buildings.copy()
        building_exposure["scenario"] = level
        building_exposure["inside"] = building_inside
        building_exposure["near_30m"] = building_near
        layer_frames["building_exposure"].append(
            building_exposure.loc[building_exposure["inside"] | building_exposure["near_30m"]]
        )
        hit_building_geometries = [
            geometry
            for geometry, inside in zip(buildings.geometry, building_inside, strict=True)
            if inside
        ]

        road_km: dict[str, float] = {}
        road_hit_flags: list[bool] = []
        road_lengths_km: list[float] = []
        for _, row in roads.iterrows():
            geometry = row.geometry
            intersection = (
                geometry.intersection(hazard_union)
                if hazard_union is not None and geometry.intersects(hazard_union)
                else None
            )
            hit_length_km = float(intersection.length / 1000) if intersection is not None else 0.0
            road_class = str(row.get("road_class") or row.get("highway") or "unknown")
            road_km[road_class] = road_km.get(road_class, 0.0) + hit_length_km
            road_hit_flags.append(hit_length_km > 0)
            road_lengths_km.append(hit_length_km)
        road_exposure = roads.copy()
        road_exposure["scenario"] = level
        road_exposure["hit_length_km"] = road_lengths_km
        road_exposure["is_hit"] = road_hit_flags
        layer_frames["road_exposure"].append(road_exposure.loc[road_exposure["is_hit"]])

        bridge_hit_flags = [
            bool(hazard_union is not None and geometry.distance(hazard_union) <= 20.0)
            for geometry in bridges.geometry
        ]
        bridge_exposure = bridges.copy()
        bridge_exposure["scenario"] = level
        bridge_exposure["is_hit_20m"] = bridge_hit_flags
        layer_frames["bridge_exposure"].append(
            bridge_exposure.loc[bridge_exposure["is_hit_20m"]]
        )

        settlement_points = settlements.geometry
        settlement_affected: list[bool] = []
        for geometry in settlement_points:
            point = geometry if geometry.geom_type == "Point" else geometry.representative_point()
            hazard_nearby = bool(hazard_union is not None and point.distance(hazard_union) <= 100.0)
            building_nearby = any(
                point.distance(hit_geometry) <= 300.0 for hit_geometry in hit_building_geometries
            )
            settlement_affected.append(hazard_nearby or building_nearby)
        settlement_exposure = settlements.copy()
        settlement_exposure["scenario"] = level
        settlement_exposure["affected"] = settlement_affected
        layer_frames["settlement_exposure"].append(
            settlement_exposure.loc[settlement_exposure["affected"]]
        )

        scenario_facts[level] = {
            "hazard_area_ha": {
                "value": hazard_result["facts"]["levels"][level]["area_ha"],
                "unit": "ha",
                "producer": "hazard",
            },
            "buildings_inside": {"value": sum(building_inside), "unit": "buildings"},
            "buildings_near_30m": {"value": sum(building_near), "unit": "buildings"},
            "road_km_hit_by_class": {
                name: {"value": length, "unit": "km"} for name, length in sorted(road_km.items())
            },
            "bridges_hit_within_20m": {
                "value": sum(bridge_hit_flags),
                "unit": "bridges",
            },
            "settlements_affected": {
                "value": sum(settlement_affected),
                "unit": "settlements",
            },
        }

    output_path = run_dir / "exposure.gpkg"
    flattened_frames = {
        layer: gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True) if frames else [],
            geometry="geometry",
            crs=buildings.crs,
        )
        for layer, frames in layer_frames.items()
    }
    _write_scenario_layers(path=output_path, layer_frames=flattened_frames)
    facts = {
        "scenarios": scenario_facts,
        "sources": list(sources),
        "footprints_are_detections_not_ground_truth": True,
        "population_reported": False,
    }
    facts_path = run_dir / "facts.json"
    merge_facts(facts_path, "damage", facts)
    manifest.inputs = [item for item in manifest.inputs if item.name != output_path.name]
    manifest.add_input(
        name=output_path.name,
        role="output",
        source="Sahayak damage exposure analysis",
        license="Derived from pre-event OpenStreetMap and detection-derived footprints",
        retrieved_utc=datetime.now(UTC),
        path=output_path,
        settings=active_settings,
    )
    manifest.write(manifest_path)
    return {"path": output_path, "facts": facts}
