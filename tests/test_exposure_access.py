"""Synthetic exposure, hazard, and access-network tests."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyogrio
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from sahayak.access.hazard import (
    _mask_for_level,
    _raster_mask_geometry,
    merge_facts,
)
from sahayak.access.run import (
    _hit_edge_ids,
    _nearest_routes,
    _NodeIndex,
    _remove_edges,
    _routes_from_all_nodes,
    build_road_graph,
    classify_access,
    run_access,
)
from sahayak.config import Settings
from sahayak.damage.run import run_damage
from sahayak.ingest.grid import GridSpec
from sahayak.ingest.network import _read_snapshot_date, estimate_network_requests
from sahayak.provenance import RuleViolation, RunManifest

CRS = "EPSG:32645"


def test_hazard_masks_are_nested_and_lenient_is_buffered() -> None:
    classes = np.array([[1, 2, 4, 3, 255]], dtype=np.uint8)
    confidence = np.array([[0.6, 0.9, 0.3, 1.0, np.nan]], dtype=np.float32)
    strict = _mask_for_level(classes, confidence, "strict")
    default = _mask_for_level(classes, confidence, "default")
    transform = from_origin(0, 10, 10, 10)

    strict_geometry = _raster_mask_geometry(strict, transform)
    default_geometry = _raster_mask_geometry(default, transform)
    lenient_geometry = default_geometry.buffer(30)

    assert strict.sum() == 1
    assert default.sum() == 3
    assert np.all(strict <= default)
    assert strict_geometry.difference(default_geometry).area == 0
    assert default_geometry.difference(lenient_geometry).area == 0
    assert lenient_geometry.area > default_geometry.area


def test_graph_splits_crossings_and_snaps_close_endpoints() -> None:
    roads = gpd.GeoDataFrame(
        {
            "highway": ["primary", "secondary"],
            "road_class": ["primary", "secondary"],
            "name": ["east-west", "north-south"],
            "bridge": ["no", "no"],
            "oneway": ["no", "no"],
        },
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(5, -5), (5, 5)]),
        ],
        crs=CRS,
    )
    graph = build_road_graph(
        roads,
        speeds_kmh={"primary": 40.0, "secondary": 30.0, "unclassified": 20.0},
    )
    center = min(
        graph.nodes,
        key=lambda node: (graph.nodes[node]["x"] - 5) ** 2 + graph.nodes[node]["y"] ** 2,
    )

    assert len(graph.nodes) == 5
    assert graph.degree(center) == 8
    assert graph.nodes[center]["x"] == 5
    assert graph.nodes[center]["y"] == 0


def test_feature_snap_splits_road_at_nearest_projection() -> None:
    roads = gpd.GeoDataFrame(
        {"highway": ["primary"], "road_class": ["primary"], "bridge": ["no"]},
        geometry=[LineString([(0, 0), (1000, 0)])],
        crs=CRS,
    )
    graph = build_road_graph(
        roads,
        speeds_kmh={"primary": 40.0, "unclassified": 20.0},
        snap_points=[Point(500, 100)],
        feature_snap_tolerance_m=150,
    )
    snapped = _NodeIndex(graph).nearest(Point(500, 100), 150)

    assert snapped is not None
    assert graph.nodes[snapped]["x"] == 500
    assert graph.nodes[snapped]["y"] == 0


def test_hit_bridge_detour_cutoff_and_unassessable_classification() -> None:
    roads = gpd.GeoDataFrame(
        {
            "highway": ["primary", "secondary", "secondary", "secondary", "track"],
            "road_class": ["primary", "secondary", "secondary", "secondary", "track"],
            "name": ["bridge road", "detour", "detour", "detour", "spur bridge"],
            "bridge": ["yes", "no", "no", "no", "yes"],
            "oneway": ["no"] * 5,
        },
        geometry=[
            LineString([(0, 0), (20, 0)]),
            LineString([(0, 0), (0, 20)]),
            LineString([(0, 20), (20, 20)]),
            LineString([(20, 20), (20, 0)]),
            LineString([(10, -20), (10, 0)]),
        ],
        crs=CRS,
    )
    graph = build_road_graph(
        roads,
        speeds_kmh={"primary": 40, "secondary": 30, "track": 8, "unclassified": 20},
    )
    hit_ids = _hit_edge_ids(graph, box(9, -1, 11, 1))
    after = _remove_edges(graph, hit_ids)
    index = _NodeIndex(graph)
    start = index.nearest(Point(0, 0), 0.1)
    target = index.nearest(Point(20, 0), 0.1)
    spur = index.nearest(Point(10, -20), 0.1)
    assert start is not None and target is not None and spur is not None

    before_routes = _nearest_routes(graph, start, [{"node": target, "name": "clinic"}], [])
    after_routes = _nearest_routes(after, start, [{"node": target, "name": "clinic"}], [])
    optimized_before = _routes_from_all_nodes(
        graph,
        [{"node": target, "name": "clinic"}],
    )
    optimized_after = _routes_from_all_nodes(
        after,
        [{"node": target, "name": "clinic"}],
    )
    assert before_routes["hospital"]["distance_km"] == 0.02
    assert optimized_before[start] == before_routes["hospital"]
    assert optimized_after[start] == after_routes["hospital"]
    assert classify_access(before_routes, after_routes) == "degraded"

    spur_before = _nearest_routes(graph, spur, [{"node": target, "name": "clinic"}], [])
    spur_after = _nearest_routes(after, spur, [{"node": target, "name": "clinic"}], [])
    assert spur_before["hospital"] is not None
    assert spur_after["hospital"] is None
    assert classify_access(spur_before, spur_after) == "cut_off"
    assert index.nearest(Point(1000, 1000), 3) is None
    assert classify_access({"hospital": None, "town": None}, {"hospital": None, "town": None}) == (
        "not_assessable"
    )
    assert classify_access(
        {"hospital": {"distance_km": 4.0}, "town": None},
        {"hospital": {"distance_km": 4.0}, "town": None},
    ) == "unaffected"


def test_facts_merge_preserves_existing_sections(tmp_path: Path) -> None:
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"optical": {"reliability": "adequate"}}), encoding="utf-8")

    merged = merge_facts(path, "damage", {"buildings": 3})

    assert merged == {
        "optical": {"reliability": "adequate"},
        "damage": {"buildings": 3},
    }
    assert json.loads(path.read_text(encoding="utf-8")) == merged


def test_network_request_estimate_uses_point_one_degree_tiles() -> None:
    assert estimate_network_requests((85.0, 27.8, 85.5, 28.3)) == 25


def test_network_ingest_uses_matching_original_snapshot_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "osm_pre_event.gpkg"
    layers = ("buildings", "roads", "bridges", "settlements", "health_facilities")
    for index, layer in enumerate(layers):
        frame = gpd.GeoDataFrame(
            {"osm_id": [index]},
            geometry=[Point(index, index)],
            crs=CRS,
        )
        pyogrio.write_dataframe(
            frame,
            path,
            layer=layer,
            driver="GPKG",
            append=index > 0,
            layer_metadata={"osm_snapshot_date": "2026-08-25"},
        )

    assert _read_snapshot_date(run_dir, {}) == date(2026, 8, 25)
    with pytest.raises(RuleViolation, match="does not match"):
        _read_snapshot_date(run_dir, {"osm_snapshot_date": "2026-08-24"})


def _write_layer(
    path: Path,
    layer: str,
    frame: gpd.GeoDataFrame,
    *,
    append: bool,
) -> None:
    """Write a synthetic GeoPackage layer."""
    pyogrio.write_dataframe(frame, path, layer=layer, driver="GPKG", append=append)


def _synthetic_run(tmp_path: Path) -> tuple[Settings, str, Point]:
    """Create a small synthetic run with a bridge, detour, settlements, and one detection."""
    run_id = "synthetic"
    settings = Settings(
        sahayak_root=tmp_path,
        data_dir=Path("data"),
        runs_dir=Path("runs"),
        _env_file=None,
    )
    run_dir = settings.resolved_runs_dir / run_id
    run_dir.mkdir(parents=True)
    bbox = (85.295, 28.095, 85.305, 28.105)
    wide_bbox = (85.28, 28.08, 85.32, 28.12)
    grid = GridSpec.from_bbox(bbox, resolution=10)
    center = Point((grid.left + grid.right) / 2, (grid.bottom + grid.top) / 2)
    manifest = RunManifest(
        run_id=run_id,
        created_utc=datetime.now(UTC),
        bbox=bbox,
        event_date=date(2026, 8, 26),
    )
    manifest.write(run_dir / "manifest.json")
    (run_dir / "selection.json").write_text(
        json.dumps({"network_osm_bbox": wide_bbox}),
        encoding="utf-8",
    )

    raster_transform = from_origin(center.x - 500, center.y + 500, 10, 10)
    classes = np.zeros((100, 100), dtype=np.uint8)
    confidence = np.zeros((100, 100), dtype=np.float32)
    classes[50, 50] = 1
    confidence[50, 50] = 0.9
    optical_dir = run_dir / "optical"
    optical_dir.mkdir()
    for name, array, dtype, nodata in (
        ("optical_class.tif", classes, "uint8", 255),
        ("optical_confidence.tif", confidence, "float32", np.nan),
    ):
        with rasterio.open(
            optical_dir / name,
            "w",
            driver="GTiff",
            height=100,
            width=100,
            count=1,
            dtype=dtype,
            crs=CRS,
            transform=raster_transform,
            nodata=nodata,
        ) as raster:
            raster.write(array, 1)
    (run_dir / "facts.json").write_text(
        json.dumps({"optical": {"reliability": "adequate"}}),
        encoding="utf-8",
    )

    roads = gpd.GeoDataFrame(
        {
            "highway": ["primary", "secondary", "secondary", "secondary", "track"],
            "road_class": ["primary", "secondary", "secondary", "secondary", "track"],
            "name": ["main bridge", "detour", "detour", "detour", "spur bridge"],
            "bridge": ["yes", "no", "no", "no", "yes"],
            "oneway": ["no"] * 5,
            "osm_id": [1, 2, 3, 4, 5],
        },
        geometry=[
            LineString([(center.x - 60, center.y), (center.x + 60, center.y)]),
            LineString([(center.x - 60, center.y), (center.x - 60, center.y + 60)]),
            LineString([(center.x - 60, center.y + 60), (center.x + 60, center.y + 60)]),
            LineString([(center.x + 60, center.y + 60), (center.x + 60, center.y)]),
            LineString([(center.x, center.y - 30), (center.x, center.y)]),
        ],
        crs=CRS,
    )
    buildings = gpd.GeoDataFrame(
        {"building": ["yes", "yes"], "osm_id": [10, 11]},
        geometry=[
            box(center.x - 2, center.y - 2, center.x + 2, center.y + 2),
            box(center.x + 15, center.y + 15, center.x + 20, center.y + 20),
        ],
        crs=CRS,
    )
    bridges = gpd.GeoDataFrame(
        {"bridge": ["yes"], "name": ["main bridge"], "osm_id": [20]},
        geometry=[Point(center.x, center.y)],
        crs=CRS,
    )
    settlements = gpd.GeoDataFrame(
        {
            "place": ["hamlet", "village", "town", "hamlet"],
            "name": ["west", "isolated", "target town", "south"],
            "osm_id": [30, 31, 32, 33],
        },
        geometry=[
            Point(center.x - 60, center.y),
            Point(center.x + 700, center.y + 700),
            Point(center.x + 60, center.y),
            Point(center.x, center.y - 30),
        ],
        crs=CRS,
    )
    health = gpd.GeoDataFrame(
        {
            "amenity": ["hospital"],
            "healthcare": [None],
            "name": ["test hospital"],
            "emergency": [None],
            "osm_id": [40],
        },
        geometry=[Point(center.x + 60, center.y)],
        crs=CRS,
    )
    original_path = run_dir / "osm_pre_event.gpkg"
    _write_layer(original_path, "buildings", buildings, append=False)
    _write_layer(original_path, "roads", roads, append=True)
    _write_layer(original_path, "bridges", bridges, append=True)
    _write_layer(original_path, "settlements", settlements, append=True)
    _write_layer(original_path, "health_facilities", health, append=True)
    network_path = run_dir / "network_osm_pre_event.gpkg"
    for index, (layer, frame) in enumerate(
        (
            ("roads", roads),
            ("bridges", bridges),
            ("health_facilities", health),
            ("settlements", settlements),
        )
    ):
        _write_layer(network_path, layer, frame, append=index > 0)
    return settings, run_id, center


def test_damage_counts_and_access_scenarios_on_synthetic_run(tmp_path: Path) -> None:
    settings, run_id, _center = _synthetic_run(tmp_path)
    damage = run_damage(run_id=run_id, settings=settings)
    strict = damage["facts"]["scenarios"]["strict"]

    assert strict["buildings_inside"]["value"] == 1
    assert strict["buildings_near_30m"]["value"] == 1
    assert strict["bridges_hit_within_20m"]["value"] == 1
    assert strict["road_km_hit_by_class"]["primary"]["value"] > 0
    assert strict["settlements_affected"]["value"] >= 2

    result = run_access(
        run_id=run_id,
        config_path=Path(__file__).resolve().parents[1] / "configs" / "access.toml",
        settings=settings,
    )
    scenarios = result["facts"]["scenarios"]
    for scenario in ("strict", "default", "lenient"):
        counts = scenarios[scenario]["counts_by_class"]
        assert counts["degraded"]["value"] >= 1
        assert counts["cut_off"]["value"] >= 1
        assert counts["not_assessable"]["value"] >= 1
    assert "south" in scenarios["strict"]["cut_off_settlements"]
    assert "south" in result["facts"]["robust_cut_off_settlements"]
    assert any(
        edge["bridge"]
        for edge in scenarios["strict"]["cut_off_blockers"]["south"]
    )
    access_frame = gpd.read_file(
        settings.resolved_runs_dir / run_id / "settlement_access.gpkg",
        layer="settlements",
    )
    west = access_frame.loc[access_frame["osm_id"] == 30].iloc[0]
    assert west["strict_class"] == "degraded"
    blocked = gpd.read_file(
        settings.resolved_runs_dir / run_id / "blocked_edges.gpkg",
        layer="blocked_edges",
    )
    assert {"road_name", "bridge", "settlement"} <= set(blocked.columns)
    assert ((blocked["settlement"] == "south") & blocked["bridge"]).any()
    assert (settings.resolved_runs_dir / run_id / "access_map.png").is_file()
    assert (settings.resolved_runs_dir / run_id / "settlement_access.gpkg").is_file()
    facts = json.loads(
        (settings.resolved_runs_dir / run_id / "facts.json").read_text(encoding="utf-8")
    )
    assert {"optical", "hazard", "damage", "access"} <= set(facts)
    areas = [
        facts["hazard"]["levels"][name]["area_ha"]
        for name in ("strict", "default", "lenient")
    ]
    assert areas[0] <= areas[1] <= areas[2]
