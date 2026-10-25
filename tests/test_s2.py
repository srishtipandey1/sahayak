"""Network-free tests for Sentinel-2 selection and screening."""

from datetime import UTC, date, datetime

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point

from sahayak.ingest.grid import GridSpec
from sahayak.ingest.run import format_ingest_summary
from sahayak.ingest.s2 import (
    S2_BANDS,
    S2_EVALSCRIPT,
    _crop_window,
    _find_syapru_centroid,
    _write_s2_cog,
    choose_s2_placebo_scene,
    choose_s2_post_control_scene,
    choose_s2_scenes,
    deduplicate_s2_scenes,
    scl_fractions,
)


def _scene(
    scene_id: str,
    acquired: str,
    clear: float,
    cloud: float,
) -> dict[str, object]:
    return {
        "id": scene_id,
        "acquired_at": datetime.fromisoformat(acquired).replace(tzinfo=UTC),
        "clear_fraction": clear,
        "eo_cloud_cover": cloud,
    }


def test_scl_fractions_report_requested_categories() -> None:
    result = scl_fractions(np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.float32))

    assert result["pixel_count"] == 12
    assert result["clear_fraction"] == 4 / 12
    assert result["fractions"]["dark_topographic_shadow"]["count"] == 1
    assert result["fractions"]["cloud_shadow"]["count"] == 1
    assert result["fractions"]["cloud_and_cirrus"]["count"] == 3
    assert result["fractions"]["snow_or_ice"]["count"] == 1
    assert result["fractions"]["no_data_or_defective"]["count"] == 2


def test_s2_same_date_duplicates_keep_lowest_cloud_item() -> None:
    scenes = [
        _scene("pre-cloudy", "2026-08-12T05:00:00", 0.4, 30),
        _scene("pre-clear", "2026-08-12T06:00:00", 0.8, 10),
    ]

    kept, duplicates = deduplicate_s2_scenes(scenes)

    assert [scene["id"] for scene in kept] == ["pre-clear"]
    assert duplicates == [
        {
            "date": "2026-08-12",
            "kept_id": "pre-clear",
            "duplicate_id": "pre-cloudy",
            "reason": "Same-date duplicate; item with lower eo:cloud_cover retained.",
        }
    ]


def test_s2_scene_choice_uses_earliest_and_latest_usable_scenes() -> None:
    scenes = [
        _scene("pre-early", "2026-08-01T05:00:00", 0.9, 20),
        _scene("pre-latest-usable", "2026-08-12T05:00:00", 0.6, 30),
        _scene("pre-latest-cloudy", "2026-08-20T05:00:00", 0.4, 5),
        _scene("post-first-usable", "2026-09-01T05:00:00", 0.8, 40),
        _scene("post-second", "2026-09-10T05:00:00", 0.9, 5),
    ]

    pre, post, reasons = choose_s2_scenes(scenes, date(2026, 8, 26))

    assert pre["id"] == "pre-latest-usable"
    assert post["id"] == "post-first-usable"
    assert "Latest pre-event" in reasons["pre"]
    assert "Earliest post-event" in reasons["post"]


def test_s2_scene_choice_fallback_uses_best_clear_fraction_and_ties() -> None:
    scenes = [
        _scene("pre-earlier", "2026-08-01T05:00:00", 0.4, 10),
        _scene("pre-later", "2026-08-12T05:00:00", 0.4, 10),
        _scene("post-later", "2026-09-10T05:00:00", 0.7, 10),
        _scene("post-earlier", "2026-09-01T05:00:00", 0.7, 10),
    ]

    pre, post, reasons = choose_s2_scenes(scenes, date(2026, 8, 26))

    assert pre["id"] == "pre-later"
    assert post["id"] == "post-earlier"
    assert reasons["pre"].startswith("Fallback")
    assert reasons["post"].startswith("Fallback")


def test_s2_evalscript_requests_reflectance_and_expected_bands() -> None:
    assert all(f"sample.{band}" in S2_EVALSCRIPT for band in S2_BANDS)
    assert 'sampleType: "FLOAT32"' in S2_EVALSCRIPT
    assert '"REFLECTANCE"' in S2_EVALSCRIPT


def test_s2_placebo_prefers_clear_scene_ten_days_before_selected_pre() -> None:
    chosen_pre = _scene("pre", "2026-08-20T05:00:00", 0.8, 5)
    candidates = [
        chosen_pre,
        _scene("too-recent-clear", "2026-08-15T05:00:00", 0.95, 2),
        _scene("older-clear", "2026-08-01T05:00:00", 0.7, 15),
        _scene("older-cloudy", "2026-07-20T05:00:00", 0.4, 40),
    ]

    scene, reason = choose_s2_placebo_scene(candidates, chosen_pre)

    assert scene["id"] == "older-clear"
    assert reason["at_least_10_days_before_pre"] is True
    assert reason["fallback"] is False


def test_s2_placebo_falls_back_when_no_clear_ten_day_candidate_exists() -> None:
    chosen_pre = _scene("pre", "2026-08-20T05:00:00", 0.8, 5)
    candidates = [
        chosen_pre,
        _scene("earlier-cloudy", "2026-08-10T05:00:00", 0.3, 50),
    ]

    scene, reason = choose_s2_placebo_scene(candidates, chosen_pre)

    assert scene["id"] == "earlier-cloudy"
    assert reason["fallback"] is True


def test_s2_post_control_chooses_clearest_eligible_scene() -> None:
    selected_post = _scene("post", "2026-10-01T05:00:00", 0.9, 5)
    candidates = [
        selected_post,
        _scene("too-early", "2026-10-14T05:00:00", 1.0, 0),
        _scene("not-clear-enough", "2026-10-20T05:00:00", 0.89, 2),
        _scene("eligible-less-clear", "2026-10-20T05:00:00", 0.92, 5),
        _scene("clearest", "2026-10-25T05:00:00", 0.98, 20),
    ]

    chosen = choose_s2_post_control_scene(candidates, selected_post)

    assert chosen["id"] == "clearest"


def test_syapru_centroid_uses_first_matching_settlement_row() -> None:
    settlements = gpd.GeoDataFrame(
        {"name": ["other", "Syapru Besi old", "स्याप्रु बेसी second"]},
        geometry=[Point(0, 0), Point(12, 34), Point(56, 78)],
        crs="EPSG:32645",
    )

    assert _find_syapru_centroid(settlements) == (12.0, 34.0)


def test_crop_window_uses_scalar_centroid_coordinates() -> None:
    grid = GridSpec(
        bbox=(0, 0, 1, 1),
        epsg=32645,
        resolution=10.0,
        left=0.0,
        bottom=0.0,
        right=100.0,
        top=100.0,
    )

    rows, columns, bounds = _crop_window(grid, 50.0, 50.0, 20.0)

    assert (rows.start, rows.stop) == (3, 7)
    assert (columns.start, columns.stop) == (3, 7)
    assert bounds == (30.0, 30.0, 70.0, 70.0)


def test_s2_cog_preserves_nodata_without_unsupported_creation_option(tmp_path, caplog) -> None:
    grid = GridSpec(
        bbox=(0, 0, 1, 1),
        epsg=32645,
        resolution=10.0,
        left=0.0,
        bottom=0.0,
        right=20.0,
        top=20.0,
    )
    path = tmp_path / "s2.tif"
    data = np.ones((len(S2_BANDS), 2, 2), dtype=np.float32)
    _write_s2_cog(path, data, grid)

    with rasterio.open(path) as source:
        assert source.nodata == -9999.0
        assert source.descriptions[0].startswith("B02")
    assert "does not support the NODATA creation option" not in caplog.text
    assert "photometric" not in caplog.text.casefold()


def test_ingest_summary_formats_scl_fraction_map_from_each_scene(tmp_path) -> None:
    grid = GridSpec(
        bbox=(0, 0, 1, 1),
        epsg=32645,
        resolution=10.0,
        left=0.0,
        bottom=0.0,
        right=20.0,
        top=20.0,
    )
    image_path = tmp_path / "s2_pre.tif"
    image_path.write_bytes(b"test")
    selected_scene = {
        "acquired_at": "2026-08-01T05:00:00+00:00",
        "platform": "sentinel-2a",
        "tile_id": "45RUM",
        "id": "scene",
        "clear_fraction": 0.7,
        "corridor_clear_fraction": None,
        "optical_usable": True,
    }
    result = {
        "run_id": "synthetic",
        "grid": grid,
        "dem": None,
        "osm": None,
        "s1": None,
        "s2": {
            "selection": {
                "selected": {
                    "pre": selected_scene,
                    "pre_2": selected_scene,
                    "post": selected_scene,
                    "post_2": selected_scene,
                }
            },
            "catalog_request_count": 1,
            "process_request_count": 1,
            "processing_units": 1.0,
            "files": {
                "s2_pre.tif": {
                    "acquired_at": selected_scene["acquired_at"],
                    "platform": selected_scene["platform"],
                    "path": image_path,
                    "scl_fractions": {"clear": {"fraction": 0.7}},
                }
            },
            "failed_quicklooks": [],
        },
        "output_dir": tmp_path,
    }

    summary = format_ingest_summary(result)

    assert "Sentinel-2 pre_2:" in summary
    assert "corridor clear fraction not available" in summary
    assert "SCL clear: 0.7" in summary
