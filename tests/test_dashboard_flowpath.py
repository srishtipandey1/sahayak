"""Synthetic tests for dashboard APIs, bundles, and terrain-only flow tracing."""

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from PIL import Image
from pyproj import Transformer
from shapely.geometry import LineString, Point

from sahayak.config import Settings
from sahayak.dashboard.run import (
    GEOJSON_MAX_BYTES,
    SAMPLE_MAX_BYTES,
    _make_georeferenced_quicklook,
    _write_geojson,
    make_sample,
)
from sahayak.flowpath import _feature, trace_downstream

CRS = "EPSG:32645"


def test_trace_downstream_stops_at_distance_grid_edge_invalid_cell_and_cycle() -> None:
    eastward = np.array([[1, 1, 1, 0]], dtype=np.uint8)
    valid = np.ones_like(eastward, dtype=bool)

    assert trace_downstream(
        eastward,
        valid,
        (0, 0),
        pixel_size_x=10,
        pixel_size_y=10,
        max_distance_m=25,
    ) == [(0, 0), (0, 1), (0, 2)]
    assert trace_downstream(
        eastward,
        valid,
        (0, 2),
        pixel_size_x=10,
        pixel_size_y=10,
        max_distance_m=30,
    ) == [(0, 2), (0, 3)]

    invalid = valid.copy()
    invalid[0, 2] = False
    assert trace_downstream(
        eastward,
        invalid,
        (0, 0),
        pixel_size_x=10,
        pixel_size_y=10,
        max_distance_m=100,
    ) == [(0, 0), (0, 1)]

    cycle = np.array([[1, 16]], dtype=np.uint8)
    assert trace_downstream(
        cycle,
        np.ones_like(cycle, dtype=bool),
        (0, 0),
        pixel_size_x=10,
        pixel_size_y=10,
        max_distance_m=100,
    ) == [(0, 0), (0, 1)]


def test_geojson_export_is_valid_simplified_and_omits_numeric_attributes(tmp_path: Path) -> None:
    frame = gpd.GeoDataFrame(
        {
            "name": ["स्याप्रु बेसी"],
            "road_km_hit": [12.5],
            "is_hit": [True],
        },
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=CRS,
    )
    path = tmp_path / "layers" / "roads_hit.geojson"
    _write_geojson(path, frame, max_features=10)

    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    assert result["features"][0]["properties"] == {
        "name": "स्याप्रु बेसी",
        "is_hit": True,
    }
    assert path.stat().st_size < GEOJSON_MAX_BYTES


def test_flowpath_feature_is_transformed_to_wgs84() -> None:
    transformer = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    feature = _feature(
        Point(500_000, 3_100_000),
        {"feature_type": "settlement", "name": "स्याप्रु"},
        transformer,
    )

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Point"
    assert -180 <= feature["geometry"]["coordinates"][0] <= 180
    assert -90 <= feature["geometry"]["coordinates"][1] <= 90
    assert feature["properties"]["name"] == "स्याप्रु"


def test_quicklook_reads_prefixed_sentinel_band_descriptions(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    dashboard_dir = run_dir / "dashboard"
    dashboard_dir.mkdir(parents=True)
    raster_path = run_dir / "s2_post.tif"
    descriptions = (
        "B02 blue surface reflectance",
        "B03 green surface reflectance",
        "B04 red surface reflectance",
        "B08 near-infrared surface reflectance",
        "B11 shortwave infrared 1 surface reflectance",
        "B12 shortwave infrared 2 surface reflectance",
        "Scene classification layer (unitless class code)",
        "Data mask (1 valid data, 0 no data)",
    )
    band_values = (0.6, 0.4, 0.2, 0.1, 0.3, 0.35, 4.0, 1.0)
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=len(descriptions),
        dtype="float32",
        crs="EPSG:32645",
        transform=Affine(10, 0, 500_000, 0, -10, 3_100_000),
    ) as dataset:
        for index, (description, value) in enumerate(
            zip(descriptions, band_values, strict=True),
            start=1,
        ):
            dataset.set_band_description(index, description)
            dataset.write(np.full((3, 4), value, dtype=np.float32), index)

    result = _make_georeferenced_quicklook(run_dir, dashboard_dir)
    with Image.open(dashboard_dir / "post_event_truecolor.png") as image:
        assert image.size == (4, 3)
        assert image.getpixel((0, 0)) == (51, 102, 153)
    assert result["source"] == "s2_post.tif"
    assert result["width"] == 4
    assert result["height"] == 3


def _test_settings(root: Path) -> Settings:
    return Settings(
        _env_file=None,
        sahayak_root=root,
        runs_dir=root / "runs",
        data_dir=root / "data",
    )


def test_sample_bundle_has_derived_outputs_only_and_stays_below_limit(tmp_path: Path) -> None:
    run_id = "sample_test"
    run_dir = tmp_path / "runs" / run_id
    dashboard_dir = run_dir / "dashboard"
    report_dir = run_dir / "report"
    dashboard_dir.mkdir(parents=True)
    report_dir.mkdir()
    (run_dir / "facts.json").write_text('{"damage":{"buildings":1}}', encoding="utf-8")
    (run_dir / "headline.json").write_text('[{"id":"buildings","value":1}]', encoding="utf-8")
    for language in ("en", "hi", "ne"):
        (report_dir / f"situation-{language}.pdf").write_bytes(b"%PDF-test")
    (dashboard_dir / "index.html").write_text("<html></html>", encoding="utf-8")
    (dashboard_dir / "post_event_truecolor.png").write_bytes(b"derived image")
    (dashboard_dir / "layers").mkdir()
    (dashboard_dir / "layers" / "roads.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
    )
    output = tmp_path / "sample_data" / run_id

    result = make_sample(run_id=run_id, output=output, settings=_test_settings(tmp_path))

    assert result["path"] == output
    assert result["size_bytes"] < SAMPLE_MAX_BYTES
    assert result["file_count"] >= 9
    assert not list(output.rglob("*.tif"))
    assert not list(output.rglob("*.gpkg"))
    assert not list(output.rglob("*.pt"))
    assert not list(output.rglob(".env"))
    assert (output / "report" / "situation-en.pdf").is_file()
    text_files = [
        path
        for path in output.rglob("*")
        if path.is_file() and path.suffix.casefold() in {".json", ".js", ".html", ".txt"}
    ]
    assert all(
        re.search(
            r"(client_secret|cdse_password|llm_api_key|ohsome_api_key|"
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
            path.read_text(encoding="utf-8", errors="replace"),
            re.IGNORECASE,
        )
        is None
        for path in text_files
    )


def test_sample_bundle_fails_if_required_report_is_missing(tmp_path: Path) -> None:
    run_id = "sample_missing"
    run_dir = tmp_path / "runs" / run_id
    (run_dir / "dashboard").mkdir(parents=True)
    (run_dir / "report").mkdir()
    (run_dir / "facts.json").write_text("{}", encoding="utf-8")
    (run_dir / "headline.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="situation-en.pdf"):
        make_sample(
            run_id=run_id,
            output=tmp_path / "sample_data" / run_id,
            settings=_test_settings(tmp_path),
        )


def test_api_scenario_layer_selection_and_facts_endpoint(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sahayak.api import create_app

    run_id = "api_test"
    root = tmp_path / "sample_data" / run_id
    dashboard = root / "dashboard"
    layer_dir = dashboard / "layers"
    layer_dir.mkdir(parents=True)
    (root / "facts.json").write_text('{"dashboard":{"visible":true}}', encoding="utf-8")
    (root / "headline.json").write_text("[]", encoding="utf-8")
    (dashboard / "index.html").write_text("<html></html>", encoding="utf-8")
    (root / "report").mkdir()
    (root / "report" / "situation-en.pdf").write_bytes(b"%PDF-test")
    for name in (
        "hazard_strict",
        "hazard_default",
        "hazard_lenient",
        "settlements_strict",
        "settlements_default",
        "settlements_lenient",
        "roads_hit",
    ):
        (layer_dir / f"{name}.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}', encoding="utf-8"
        )

    client = TestClient(
        create_app(
            run_id=run_id,
            settings=_test_settings(tmp_path),
            sample=True,
        )
    )

    facts = client.get("/api/facts")
    strict = client.get("/api/layers", params={"scenario": "strict"})
    layer = client.get("/api/layers/roads_hit.geojson")
    report = client.get("/report/situation-en.pdf")
    index = client.get("/")
    invalid = client.get("/api/layers", params={"scenario": "untrusted"})

    assert facts.status_code == 200
    assert facts.json() == {"dashboard": {"visible": True}}
    assert layer.status_code == 200
    assert layer.json()["type"] == "FeatureCollection"
    assert report.status_code == 200
    assert index.status_code == 200
    assert strict.status_code == 200
    assert strict.json()["layers"] == [
        "hazard_strict",
        "roads_hit",
        "settlements_strict",
    ]
    assert invalid.status_code == 400
