"""Road-network routing and scenario-based settlement accessibility estimates."""

from __future__ import annotations

import json
import math
import tempfile
import tomllib
from datetime import UTC, datetime
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyogrio
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point, box
from shapely.ops import substring, unary_union
from shapely.strtree import STRtree

from sahayak.access.hazard import HAZARD_LEVELS, build_hazards, merge_facts, read_hazards
from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id
from sahayak.provenance import RunManifest

ACCESS_CLASSES = ("cut_off", "degraded", "unaffected", "not_assessable")
OSM_ATTRIBUTION_TEXT = (
    "© OpenStreetMap contributors (ODbL). "
    "Any published modified OSM data stays under ODbL."
)
ACCESS_MAP_ATTRIBUTION = "\n".join(
    (
        "Contains modified Copernicus Sentinel data 2026.",
        "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and "
        "Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; "
        "all rights reserved.",
        OSM_ATTRIBUTION_TEXT,
    )
)


def _lines(geometry: Any) -> list[LineString]:
    """Flatten line and multiline geometries into usable line segments."""
    if geometry is None or geometry.is_empty:
        return []
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    if hasattr(geometry, "geoms"):
        return [line for part in geometry.geoms for line in _lines(part)]
    return []


def _cluster_coordinates(
    coordinates: list[tuple[float, float]],
    tolerance_m: float,
) -> list[tuple[float, float]]:
    """Snap nearby graph endpoints together within the documented tolerance."""
    if not coordinates:
        return []
    union_find = nx.utils.UnionFind(range(len(coordinates)))
    tree = cKDTree(coordinates)
    for left, right in tree.query_pairs(tolerance_m):
        union_find.union(left, right)
    groups: dict[Any, list[tuple[float, float]]] = {}
    for index, coordinate in enumerate(coordinates):
        groups.setdefault(union_find[index], []).append(coordinate)
    centers = {
        member: (
            float(np.mean([item[0] for item in values])),
            float(np.mean([item[1] for item in values])),
        )
        for member, values in groups.items()
    }
    return [centers[union_find[index]] for index in range(len(coordinates))]


def build_road_graph(
    roads: gpd.GeoDataFrame,
    *,
    speeds_kmh: dict[str, float],
    snap_tolerance_m: float = 3.0,
    snap_points: list[Point] | None = None,
    feature_snap_tolerance_m: float = 500.0,
) -> nx.MultiDiGraph:
    """Node roads at crossings and endpoints, then build a weighted directed graph."""
    if roads.crs is None or not roads.crs.is_projected:
        raise ValueError("Road network must use a projected CRS.")
    if snap_tolerance_m < 0 or not math.isfinite(snap_tolerance_m):
        raise ValueError("Road snap tolerance must be finite and non-negative.")
    if feature_snap_tolerance_m < 0 or not math.isfinite(feature_snap_tolerance_m):
        raise ValueError("Feature snap tolerance must be finite and non-negative.")
    source_geometries: list[LineString] = []
    source_rows: list[dict[str, Any]] = []
    for _, row in roads.iterrows():
        for geometry in _lines(row.geometry):
            if geometry.length > 0:
                source_geometries.append(geometry)
                source_rows.append(row.to_dict())
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = roads.crs
    graph.graph["snap_tolerance_m"] = snap_tolerance_m
    if not source_geometries:
        return graph

    noded_lines = _lines(unary_union(source_geometries))
    feature_tree = STRtree(noded_lines)
    split_positions: dict[int, set[float]] = {}
    for point in snap_points or []:
        line_index = int(feature_tree.nearest(point))
        line = noded_lines[line_index]
        if point.distance(line) <= feature_snap_tolerance_m:
            position = float(line.project(point))
            split_positions.setdefault(line_index, set()).add(position)

    tree = STRtree(source_geometries)
    candidates: list[
        tuple[tuple[float, float], tuple[float, float], LineString, dict[str, Any]]
    ] = []
    for line_index, line in enumerate(noded_lines):
        positions = sorted({0.0, float(line.length), *split_positions.get(line_index, set())})
        pieces = [
            substring(line, start_measure, end_measure)
            for start_measure, end_measure in zip(positions, positions[1:], strict=False)
            if end_measure > start_measure
        ]
        for piece in pieces:
            start = tuple(piece.coords[0])
            end = tuple(piece.coords[-1])
            length = float(piece.length)
            if length <= 0:
                continue
            source_index = int(tree.nearest(piece.interpolate(0.5, normalized=True)))
            row = source_rows[source_index]
            road_class = str(row.get("road_class") or row.get("highway") or "unclassified")
            speed = float(speeds_kmh.get(road_class, speeds_kmh["unclassified"]))
            if speed <= 0 or not math.isfinite(speed):
                raise ValueError(f"Speed for road class {road_class!r} must be positive.")
            candidates.append(
                (
                    start,
                    end,
                    piece,
                    {**row, "road_class": road_class, "speed": speed},
                )
            )

    endpoint_coordinates = [coord for start, end, *_ in candidates for coord in (start, end)]
    snapped_coordinates = _cluster_coordinates(endpoint_coordinates, snap_tolerance_m)
    nodes: dict[tuple[float, float], int] = {}
    for coordinate in snapped_coordinates:
        if coordinate not in nodes:
            node_id = len(nodes)
            nodes[coordinate] = node_id
            graph.add_node(node_id, x=coordinate[0], y=coordinate[1])

    for index, (_start, _end, geometry, attributes) in enumerate(candidates):
        start_node = nodes[snapped_coordinates[index * 2]]
        end_node = nodes[snapped_coordinates[index * 2 + 1]]
        length_m = float(geometry.length)
        minutes = length_m / 1000 / attributes["speed"] * 60
        bridge_value = attributes.get("bridge")
        bridge = bool(
            bridge_value
            and str(bridge_value).strip().casefold() not in {"no", "false", "0"}
        )
        edge_attributes = {
            "edge_id": index,
            "geometry": geometry,
            "length_m": length_m,
            "length_km": length_m / 1000,
            "minutes": minutes,
            "road_class": attributes["road_class"],
            "road_name": attributes.get("name"),
            "bridge": bridge,
            "oneway": str(attributes.get("oneway") or "no").casefold(),
        }
        oneway = edge_attributes["oneway"]
        if oneway in {"-1", "reverse"}:
            graph.add_edge(end_node, start_node, key=index, **edge_attributes)
        else:
            graph.add_edge(start_node, end_node, key=index, **edge_attributes)
            if oneway not in {"yes", "true", "1"}:
                reverse_geometry = LineString(list(geometry.coords)[::-1])
                graph.add_edge(
                    end_node,
                    start_node,
                    key=index + len(candidates),
                    **{**edge_attributes, "geometry": reverse_geometry},
                )
    return graph


class _NodeIndex:
    """Nearest-node index for snapping settlements and service targets."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self.nodes = list(graph.nodes)
        self.coordinates = [(graph.nodes[node]["x"], graph.nodes[node]["y"]) for node in self.nodes]
        self.tree = cKDTree(self.coordinates) if self.coordinates else None

    def nearest(self, point: Point, tolerance_m: float) -> int | None:
        """Return the closest graph node within the configured snap tolerance."""
        if self.tree is None:
            return None
        distance, index = self.tree.query((point.x, point.y), k=1)
        if distance > tolerance_m:
            return None
        return self.nodes[int(index)]


def _point_for_feature(geometry: Any) -> Point:
    """Use a stable representative point for settlement/facility polygons."""
    if geometry.geom_type == "Point":
        return geometry
    return geometry.representative_point()


def _target_name(value: Any, fallback: str) -> str:
    """Return a real target name or a clear label for an unnamed OSM target."""
    if value is None:
        return fallback
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and cleaned.casefold() not in {"nan", "none", "<na>"}:
            return cleaned
        return fallback
    try:
        if bool(np.isnan(value)):
            return fallback
    except (TypeError, ValueError):
        pass
    return str(value)


def _shortest_target(
    graph: nx.MultiDiGraph,
    source: int,
    targets: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the nearest target by road distance with travel time on that route."""
    if not targets:
        return None
    distances = nx.single_source_dijkstra_path_length(graph, source, weight="length_m")
    candidates = [
        (float(distances[target["node"]]), str(target["name"]), target)
        for target in targets
        if target["node"] in distances
    ]
    if not candidates:
        return None
    distance_m, _name, target = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    route = nx.shortest_path(graph, source, target["node"], weight="length_m")
    minutes = 0.0
    for start, end in zip(route, route[1:], strict=False):
        edge_data = graph.get_edge_data(start, end)
        chosen = min(edge_data.values(), key=lambda edge: edge["length_m"])
        minutes += float(chosen["minutes"])
    return {
        "name": str(target["name"]),
        "distance_km": distance_m / 1000,
        "minutes": minutes,
    }


def classify_access(
    before: dict[str, dict[str, Any] | None],
    after: dict[str, dict[str, Any] | None],
    *,
    degraded_increase_fraction: float = 0.5,
    degraded_increase_km: float = 5.0,
) -> str:
    """Classify a settlement from nearest hospital/town reachability changes."""
    reachable_before = [item for item in before.values() if item is not None]
    if not reachable_before:
        return "not_assessable"
    degraded = False
    for target_type, baseline in before.items():
        if baseline is None:
            continue
        current = after.get(target_type)
        if current is None:
            return "cut_off"
        increase_km = current["distance_km"] - baseline["distance_km"]
        if (
            increase_km > 0
            and (
                increase_km >= degraded_increase_km
                or increase_km >= baseline["distance_km"] * degraded_increase_fraction
            )
        ):
            degraded = True
    return "degraded" if degraded else "unaffected"


def _targets(
    frame: gpd.GeoDataFrame,
    *,
    node_index: _NodeIndex,
    snap_m: float,
    name_field: str,
    missing_name: str,
) -> list[dict[str, Any]]:
    """Snap each mapped health/town target to a graph node."""
    targets = []
    for _, row in frame.iterrows():
        point = _point_for_feature(row.geometry)
        node = node_index.nearest(point, snap_m)
        if node is not None:
            targets.append(
                {
                    "node": node,
                    "name": _target_name(row.get(name_field), missing_name),
                    "osm_id": row.get("osm_id"),
                }
            )
    return targets


def _hit_edge_ids(graph: nx.MultiDiGraph, hazard: Any | None) -> set[int]:
    """Identify roads intersecting hazard and bridges within the 20 m buffer."""
    if hazard is None:
        return set()
    hit = set()
    for _start, _end, _key, edge in graph.edges(keys=True, data=True):
        geometry = edge["geometry"]
        if geometry.intersects(hazard) or (edge["bridge"] and geometry.distance(hazard) <= 20.0):
            hit.add(edge["edge_id"])
    return hit


def _remove_edges(graph: nx.MultiDiGraph, hit_ids: set[int]) -> nx.MultiDiGraph:
    """Copy a road graph without any directed edge corresponding to a hit segment."""
    result = graph.copy()
    removed = [
        (start, end, key)
        for start, end, key, data in result.edges(keys=True, data=True)
        if data["edge_id"] in hit_ids
    ]
    result.remove_edges_from(removed)
    return result


def _nearest_routes(
    graph: nx.MultiDiGraph,
    node: int,
    hospitals: list[dict[str, Any]],
    towns: list[dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Calculate nearest reachable hospital and town routes."""
    return {
        "hospital": _shortest_target(graph, node, hospitals),
        "town": _shortest_target(graph, node, towns),
    }


def _routes_from_all_nodes(
    graph: nx.MultiDiGraph,
    targets: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Find the nearest target and route costs for every node in one traversal."""
    target_names: dict[int, str] = {}
    for target in targets:
        node = int(target["node"])
        name = str(target["name"])
        target_names[node] = min(target_names.get(node, name), name)
    best: dict[int, tuple[float, str]] = {
        node: (0.0, name) for node, name in target_names.items()
    }
    minutes: dict[int, float] = {node: 0.0 for node in target_names}
    heap = [(0.0, name, node) for node, name in target_names.items()]
    heapify(heap)
    while heap:
        distance_m, target_name, node = heappop(heap)
        if best.get(node) != (distance_m, target_name):
            continue
        for predecessor in graph.predecessors(node):
            edges = graph.get_edge_data(predecessor, node)
            edge = min(edges.values(), key=lambda value: value["length_m"])
            candidate = (distance_m + float(edge["length_m"]), target_name)
            if candidate < best.get(predecessor, (math.inf, "")):
                best[predecessor] = candidate
                minutes[predecessor] = minutes[node] + float(edge["minutes"])
                heappush(heap, (candidate[0], candidate[1], predecessor))
    return {
        node: {
            "name": target_name,
            "distance_km": distance_m / 1000,
            "minutes": minutes[node],
        }
        for node, (distance_m, target_name) in best.items()
    }


def _blocker_records(
    graph: nx.MultiDiGraph,
    after_graph: nx.MultiDiGraph,
    source: int,
    hit_ids: set[int],
    scenario: str,
) -> list[dict[str, Any]]:
    """Return hazard-hit edges that border the settlement's surviving component."""
    reachable = set(nx.single_source_shortest_path_length(after_graph, source))
    before_reachable = set(nx.single_source_shortest_path_length(graph, source))
    records: dict[int, dict[str, Any]] = {}
    for start, end, _key, edge in graph.edges(keys=True, data=True):
        edge_id = edge["edge_id"]
        if edge_id not in hit_ids:
            continue
        boundary = (
            start in reachable and end in before_reachable - reachable
        ) or (
            end in reachable and start in before_reachable - reachable
        )
        if boundary:
            records[edge_id] = {
                "scenario": scenario,
                "edge_id": edge_id,
                "road_name": edge["road_name"],
                "road_class": edge["road_class"],
                "bridge": edge["bridge"],
                "geometry": edge["geometry"],
            }
    return list(records.values())


def _load_access_config(path: Path) -> dict[str, Any]:
    """Load and validate documented routing assumptions."""
    with path.open("rb") as file:
        data = tomllib.load(file)
    speeds = data.get("speeds_kmh")
    snapping = data.get("snapping")
    classification = data.get("classification")
    if not all(isinstance(value, dict) for value in (speeds, snapping, classification)):
        raise ValueError("Access configuration requires speeds_kmh, snapping, and classification.")
    if "unclassified" not in speeds or any(float(speed) <= 0 for speed in speeds.values()):
        raise ValueError("Access speed table must include positive speeds and unclassified.")
    return data


def _write_access_map(
    *,
    roads: gpd.GeoDataFrame,
    settlements: gpd.GeoDataFrame,
    hospitals: gpd.GeoDataFrame,
    blocked: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """Render roads, default-scenario cut edges, settlements, and health facilities."""
    figure, axis = plt.subplots(figsize=(12, 10))
    if not roads.empty:
        roads.plot(ax=axis, color="#a6a6a6", linewidth=0.55, zorder=1)
    default_blocked = (
        blocked.loc[blocked["scenario"] == "default"] if not blocked.empty else blocked
    )
    if not default_blocked.empty:
        default_blocked.plot(ax=axis, color="#d62728", linewidth=1.5, zorder=3)
    color_by_class = {
        "cut_off": "#d62728",
        "degraded": "#ff7f0e",
        "unaffected": "#2ca02c",
        "not_assessable": "#7f7f7f",
    }
    for class_name, color in color_by_class.items():
        subset = settlements.loc[settlements["default_class"] == class_name]
        if not subset.empty:
            subset.plot(ax=axis, color=color, markersize=24, zorder=4, label=class_name)
    if not hospitals.empty:
        hospitals.plot(
            ax=axis,
            color="#1f77b4",
            marker="*",
            markersize=75,
            zorder=5,
            label="health facility",
        )
    axis.set_title("Pre-event road access scenarios")
    axis.set_axis_off()
    axis.legend(loc="upper right")
    figure.text(
        0.01,
        0.01,
        f"{ACCESS_MAP_ATTRIBUTION}\nDetection footprints are not ground truth.",
        fontsize=6.5,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _write_gpkg_atomic(path: Path, frame: gpd.GeoDataFrame, layer: str) -> None:
    """Write one GeoPackage layer and replace the previous output atomically."""
    with tempfile.TemporaryDirectory(prefix=".access.", dir=path.parent) as temp_dir:
        temporary_path = Path(temp_dir) / path.name
        pyogrio.write_dataframe(
            frame,
            temporary_path,
            layer=layer,
            driver="GPKG",
        )
        temporary_path.replace(path)


def run_access(
    *,
    run_id: str,
    config_path: Path = Path("configs/access.toml"),
    sources: tuple[str, ...] = ("optical",),
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compute pre/post-hazard shortest routes for settlements in the wide area."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    manifest_path = run_dir / "manifest.json"
    selection_path = run_dir / "selection.json"
    network_path = run_dir / "network_osm_pre_event.gpkg"
    if not manifest_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"Run {run_id!r} lacks selection.json or manifest.json.")
    if not network_path.is_file():
        raise FileNotFoundError("Access analysis requires network_osm_pre_event.gpkg.")
    manifest = RunManifest.read(manifest_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    bbox = selection.get("network_osm_bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("Run selection has no wide-area network OSM bbox.")
    config_file = config_path
    if not config_file.is_absolute():
        config_file = active_settings.env_file_path.parent / config_file
    config = _load_access_config(config_file)
    hazard_result = build_hazards(run_id=run_id, sources=sources, settings=active_settings)
    hazards = read_hazards(hazard_result["path"])
    hazard_crs = gpd.read_file(hazard_result["path"], layer="hazards").crs
    if hazard_crs is None:
        raise ValueError("Hazard footprints have no CRS.")

    roads = gpd.read_file(network_path, layer="roads")
    if roads.crs is None or not roads.crs.is_projected:
        raise ValueError("Network roads must use a projected CRS.")
    projected_bbox = (
        gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(roads.crs).iloc[0]
    )
    settlements = gpd.read_file(network_path, layer="settlements").to_crs(roads.crs)
    health = gpd.read_file(network_path, layer="health_facilities").to_crs(roads.crs)
    settlements = settlements.loc[
        settlements.geometry.representative_point().intersects(projected_bbox)
    ].copy()
    health = health.loc[health.geometry.representative_point().intersects(projected_bbox)].copy()
    towns_source = settlements.loc[settlements["place"].isin(("town", "city"))]
    road_subset = roads.loc[roads.geometry.intersects(projected_bbox)].copy()
    road_subset.geometry = road_subset.geometry.intersection(projected_bbox)
    roads_for_graph = road_subset.loc[~road_subset.geometry.is_empty].copy()
    speeds = {name: float(speed) for name, speed in config["speeds_kmh"].items()}
    snap_settlement = float(config["snapping"]["settlement_to_road_m"])
    snap_target = float(config["snapping"].get("target_to_road_m", snap_settlement))
    snap_points = [
        *settlements.geometry.map(_point_for_feature).tolist(),
        *health.geometry.map(_point_for_feature).tolist(),
        *towns_source.geometry.map(_point_for_feature).tolist(),
    ]
    graph = build_road_graph(
        roads_for_graph,
        speeds_kmh=speeds,
        snap_tolerance_m=float(config["snapping"]["road_tolerance_m"]),
        snap_points=snap_points,
        feature_snap_tolerance_m=max(snap_settlement, snap_target),
    )
    node_index = _NodeIndex(graph)
    hospitals = _targets(
        health,
        node_index=node_index,
        snap_m=snap_target,
        name_field="name",
        missing_name="Unnamed health facility",
    )
    towns = _targets(
        towns_source,
        node_index=node_index,
        snap_m=snap_target,
        name_field="name",
        missing_name="Unnamed town",
    )
    settlement_nodes: list[int | None] = [
        node_index.nearest(_point_for_feature(geometry), snap_settlement)
        for geometry in settlements.geometry
    ]
    classifier = config["classification"]
    class_fields: dict[str, list[str]] = {level: [] for level in HAZARD_LEVELS}
    scenario_route_values: dict[str, list[dict[str, Any]]] = {
        level: [] for level in HAZARD_LEVELS
    }
    blocker_rows: list[dict[str, Any]] = []
    before_routes_by_type = {
        "hospital": _routes_from_all_nodes(graph, hospitals),
        "town": _routes_from_all_nodes(graph, towns),
    }

    for level in HAZARD_LEVELS:
        hazard = hazards[level]
        if hazard is not None:
            hazard = gpd.GeoSeries([hazard], crs=hazard_crs).to_crs(roads.crs).iloc[0]
        hit_ids = _hit_edge_ids(graph, hazard)
        after_graph = _remove_edges(graph, hit_ids)
        after_routes_by_type = {
            "hospital": _routes_from_all_nodes(after_graph, hospitals),
            "town": _routes_from_all_nodes(after_graph, towns),
        }
        for row_index, settlement_node in enumerate(settlement_nodes):
            if settlement_node is None:
                before_routes = {"hospital": None, "town": None}
                after_routes = {"hospital": None, "town": None}
                status = "not_assessable"
            else:
                before_routes = {
                    target_type: routes.get(settlement_node)
                    for target_type, routes in before_routes_by_type.items()
                }
                after_routes = {
                    target_type: routes.get(settlement_node)
                    for target_type, routes in after_routes_by_type.items()
                }
                status = classify_access(
                    before_routes,
                    after_routes,
                    degraded_increase_fraction=float(
                        classifier["degraded_increase_fraction"]
                    ),
                    degraded_increase_km=float(classifier["degraded_increase_km"]),
                )
            class_fields[level].append(status)
            scenario_route_values[level].append(
                {
                    "before": before_routes,
                    "after": after_routes,
                    "status": status,
                }
            )
            if status == "cut_off" and settlement_node is not None:
                for record in _blocker_records(
                    graph, after_graph, settlement_node, hit_ids, level
                ):
                    blocker_rows.append(
                        {
                            **record,
                            "settlement": str(
                                settlements.iloc[row_index].get("name")
                                or settlements.iloc[row_index].get("osm_id")
                                or row_index
                            ),
                        }
                    )

    settlement_output = settlements.copy()
    for level in HAZARD_LEVELS:
        settlement_output[f"{level}_class"] = class_fields[level]
        for target_type in ("hospital", "town"):
            for direction in ("before", "after"):
                settlement_output[f"{level}_{target_type}_{direction}_name"] = [
                    item[direction][target_type]["name"]
                    if item[direction][target_type] is not None
                    else None
                    for item in scenario_route_values[level]
                ]
                settlement_output[f"{level}_{target_type}_{direction}_km"] = [
                    item[direction][target_type]["distance_km"]
                    if item[direction][target_type] is not None
                    else None
                    for item in scenario_route_values[level]
                ]
                settlement_output[f"{level}_{target_type}_{direction}_minutes"] = [
                    item[direction][target_type]["minutes"]
                    if item[direction][target_type] is not None
                    else None
                    for item in scenario_route_values[level]
                ]

    blocked_columns = (
        "scenario",
        "edge_id",
        "road_name",
        "road_class",
        "bridge",
        "settlement",
        "geometry",
    )
    blocked = gpd.GeoDataFrame(
        blocker_rows,
        columns=blocked_columns,
        geometry="geometry",
        crs=roads.crs,
    ).drop_duplicates(subset=["scenario", "edge_id", "settlement"])

    scenario_facts: dict[str, Any] = {}
    for level in HAZARD_LEVELS:
        classes = class_fields[level]
        scenario_blockers = (
            blocked.loc[blocked["scenario"] == level] if not blocked.empty else blocked
        )
        cut_off_ids = [
            str(
                settlements.iloc[index].get("osm_id")
                or settlements.iloc[index].get("name")
                or index
            )
            for index, value in enumerate(classes)
            if value == "cut_off"
        ]
        scenario_facts[level] = {
            "counts_by_class": {
                name: {"value": classes.count(name), "unit": "settlements"}
                for name in ACCESS_CLASSES
            },
            "cut_off_settlements": [
                str(
                    settlements.iloc[index].get("name")
                    or settlements.iloc[index].get("osm_id")
                    or index
                )
                for index, value in enumerate(classes)
                if value == "cut_off"
            ],
            "cut_off_settlement_ids": cut_off_ids,
            "cut_off_blockers": {
                str(settlement): [
                    {
                        "road_name": (
                            str(row.road_name)
                            if row.road_name is not None
                            and str(row.road_name).casefold() not in {"nan", "none", ""}
                            else None
                        ),
                        "road_class": str(row.road_class),
                        "bridge": bool(row.bridge),
                    }
                    for row in group.itertuples()
                ]
                for settlement, group in scenario_blockers.groupby("settlement")
            },
            "not_assessable_count": classes.count("not_assessable"),
            "blocked_edge_count": int(scenario_blockers["edge_id"].nunique())
            if not scenario_blockers.empty
            else 0,
        }
    robust_ids = set(scenario_facts["strict"]["cut_off_settlement_ids"])
    robust_ids &= set(scenario_facts["default"]["cut_off_settlement_ids"])
    robust_ids &= set(scenario_facts["lenient"]["cut_off_settlement_ids"])
    all_names: dict[str, str] = {}
    for index, row in settlements.iterrows():
        identifier = str(row.get("osm_id") or row.get("name") or index)
        all_names[identifier] = str(row.get("name") or identifier)
    facts = {
        "scenarios": scenario_facts,
        "robust_cut_off_settlement_ids": sorted(robust_ids),
        "robust_cut_off_settlements": sorted(all_names[item] for item in robust_ids),
        "not_assessable_count": scenario_facts["default"]["not_assessable_count"],
        "assumptions": {
            "access_config": str(config_file),
            "edge_hit_rule": "any intersection; bridge within 20 m",
            "settlement_snap_m": float(config["snapping"]["settlement_to_road_m"]),
            "target_snap_m": snap_target,
            "road_snap_tolerance_m": float(config["snapping"]["road_tolerance_m"]),
            "degraded_rule": "distance increase >=50% or >=5 km",
        },
        "population_reported": False,
        "footprints_are_detections_not_ground_truth": True,
    }

    output_path = run_dir / "settlement_access.gpkg"
    blocked_path = run_dir / "blocked_edges.gpkg"
    _write_gpkg_atomic(output_path, settlement_output, "settlements")
    _write_gpkg_atomic(blocked_path, blocked, "blocked_edges")
    map_path = run_dir / "access_map.png"
    _write_access_map(
        roads=roads_for_graph,
        settlements=settlement_output,
        hospitals=health,
        blocked=blocked,
        path=map_path,
    )
    merge_facts(run_dir / "facts.json", "access", facts)
    manifest = RunManifest.read(manifest_path)
    for output in (output_path, blocked_path, map_path):
        manifest.inputs = [item for item in manifest.inputs if item.name != output.name]
        manifest.add_input(
            name=output.name,
            role="output",
            source="Sahayak settlement access analysis",
            license="Derived from pre-event OpenStreetMap and detection-derived footprints",
            retrieved_utc=datetime.now(UTC),
            path=output,
            settings=active_settings,
        )
    manifest.write(manifest_path)
    return {
        "files": [output_path, blocked_path, map_path],
        "facts": facts,
        "scenario_routes": scenario_route_values,
    }
