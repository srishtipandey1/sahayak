"""Offline tests for Sentinel-1 metadata, pairing, and grid-aligned outputs."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.ingest.s1 import (
    PROCESS_URL,
    Acquisition,
    AcquisitionPair,
    ProcessTile,
    Sentinel1Error,
    SentinelHubClient,
    build_catalog_request,
    build_process_request,
    catalog_acquisitions,
    group_slices,
    ingest_s1,
    mosaic_process_tiles,
    process_tile_windows,
    select_baseline,
    select_pairs,
    write_s1_cog,
)
from sahayak.provenance import RunManifest

BBOX = (85.3, 28.1, 85.42, 28.2)
EVENT_DATE = date(2026, 8, 26)
FOOTPRINT = box(*BBOX)
GRID = GridSpec.from_bbox(BBOX)


def _acquisition(
    day: int,
    *,
    orbit: int = 45,
    direction: str = "ASCENDING",
    platform: str = "sentinel-1a",
    coverage: float = 1.0,
    mode: str = "IW",
    polarizations: tuple[str, ...] = ("VH", "VV"),
    hour: int = 5,
) -> Acquisition:
    """Create one small synthetic acquisition for selection tests."""
    timestamp = datetime(2026, 8, day, hour, tzinfo=UTC)
    return Acquisition(
        platform=platform,
        acquired_at=timestamp,
        relative_orbit=orbit,
        direction=direction,  # type: ignore[arg-type]
        polarizations=polarizations,
        mode=mode,
        footprint=FOOTPRINT,
        coverage_fraction=coverage,
        item_ids=(f"{platform}-{timestamp.date().isoformat()}",),
        start_time=timestamp,
        end_time=timestamp,
    )


def _catalog_item(
    item_id: str,
    timestamp: str,
    *,
    orbit: int = 45,
    platform: str = "sentinel-1a",
    direction: str = "ASCENDING",
    bbox: tuple[float, float, float, float] = BBOX,
) -> dict[str, Any]:
    """Create a synthetic STAC Catalog item with explicit metadata."""
    return {
        "id": item_id,
        "geometry": mapping(box(*bbox)),
        "properties": {
            "datetime": timestamp,
            "platform": platform,
            "sat:relative_orbit": orbit,
            "orbitDirection": direction,
            "s1:polarization": "DV",
            "acquisitionMode": "IW",
        },
    }


def test_catalog_request_searches_requested_event_window() -> None:
    request = build_catalog_request(BBOX, EVENT_DATE)

    assert request["bbox"] == list(BBOX)
    assert request["collections"] == ["sentinel-1-grd"]
    assert request["datetime"].startswith("2026-06-27T00:00:00Z/")
    assert request["datetime"].endswith("2026-10-10T23:59:59Z")
    assert build_catalog_request(BBOX, EVENT_DATE, page_token=5)["next"] == 5
    assert PROCESS_URL == "https://sh.dataspace.copernicus.eu/process/v1"


def test_catalog_pagination_uses_context_next_as_top_level_request_next() -> None:
    class FakeCatalog:
        """Return two pages and record their request payloads."""

        def __init__(self) -> None:
            """Initialize the page counter."""
            self.requests: list[dict[str, object]] = []

        def catalog_search(self, body: dict[str, object]) -> dict[str, Any]:
            """Return one Catalog page with its documented next offset."""
            self.requests.append(body)
            if len(self.requests) == 1:
                return {
                    "features": [_catalog_item("slice-1", "2026-08-20T05:00:00Z")],
                    "context": {"next": 5},
                }
            return {
                "features": [_catalog_item("slice-2", "2026-08-20T05:01:00Z")],
                "context": {},
            }

    client = FakeCatalog()

    acquisitions = catalog_acquisitions(
        bbox=BBOX,
        event_date=EVENT_DATE,
        grid=GRID,
        client=client,  # type: ignore[arg-type]
    )

    assert len(client.requests) == 2
    assert client.requests[1]["next"] == 5
    assert len(acquisitions) == 1
    assert acquisitions[0].item_ids == ("slice-1", "slice-2")


def test_slice_grouping_combines_same_datatake_and_preserves_track() -> None:
    items = [
        _catalog_item("slice-1", "2026-08-20T05:00:00Z"),
        _catalog_item("slice-2", "2026-08-20T05:01:00Z"),
        _catalog_item("slice-3", "2026-08-20T05:02:31Z"),
    ]

    acquisitions = group_slices(items, BBOX, GRID)

    assert len(acquisitions) == 2
    assert acquisitions[0].item_ids == ("slice-1", "slice-2")
    assert acquisitions[0].relative_orbit == 45
    assert acquisitions[0].direction == "ASCENDING"
    assert acquisitions[0].has_dual_polarization


def test_catalog_item_without_relative_orbit_fails_closed() -> None:
    item = _catalog_item("no-orbit", "2026-08-20T05:00:00Z")
    del item["properties"]["sat:relative_orbit"]
    exclusions: list[dict[str, str]] = []

    acquisitions = group_slices([item], BBOX, GRID, exclusions=exclusions)

    assert acquisitions == []
    assert "relative orbit must be at least 1" in exclusions[0]["reason"]


def test_invalid_relative_orbit_is_excluded_with_a_reason() -> None:
    item = _catalog_item("invalid-orbit", "2026-08-20T05:00:00Z", orbit=-1)
    exclusions: list[dict[str, str]] = []

    acquisitions = group_slices([item], BBOX, GRID, exclusions=exclusions)

    assert acquisitions == []
    assert exclusions == [
        {
            "item_id": "invalid-orbit",
            "platform": "sentinel-1a",
            "datetime": "2026-08-20T05:00:00Z",
            "reason": (
                "Catalog item 'invalid-orbit' has missing or invalid "
                "sat:relative_orbit metadata (-1); relative orbit must be at least 1."
            ),
        }
    ]


def test_invalid_orbit_is_recovered_from_cached_main_cdse_stac(tmp_path: Path) -> None:
    class FakeResponse:
        status_code = 200
        content = (
            b'{"features":[{"id":"stac-item","properties":'
            b'{"datetime":"2026-08-20T05:00:00Z","platform":"sentinel-1d",'
            b'"sat:relative_orbit":117,"sat:orbit_state":"DESCENDING"}}]}'
        )

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
            del args, kwargs
            self.calls += 1
            return FakeResponse()

    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    session = FakeSession()
    client = SentinelHubClient(
        settings,
        RequestCache(settings),
        session=session,  # type: ignore[arg-type]
    )
    item = _catalog_item(
        "sentinel-1d-invalid",
        "2026-08-20T05:00:00Z",
        orbit=-1,
        platform="sentinel-1d",
    )

    acquisitions = group_slices(
        [item],
        BBOX,
        GRID,
        orbit_resolver=client.resolve_relative_orbit,
    )
    second_lookup = client.resolve_relative_orbit(
        "sentinel-1d",
        datetime(2026, 8, 20, 5, tzinfo=UTC),
        item["geometry"],
    )

    assert len(acquisitions) == 1
    assert acquisitions[0].relative_orbit == 117
    assert acquisitions[0].direction == "DESCENDING"
    assert acquisitions[0].orbit_metadata_source == "cdse-stac"
    assert second_lookup == (117, "DESCENDING")
    assert session.calls == 1


def test_two_sentinel_1d_invalid_orbits_are_never_grouped_as_same_track() -> None:
    items = [
        _catalog_item(
            f"sentinel-1d-{day}",
            f"2026-08-{day}T05:00:00Z",
            orbit=-1,
            platform="sentinel-1d",
        )
        for day in (20, 26)
    ]
    exclusions: list[dict[str, str]] = []

    acquisitions = group_slices(items, BBOX, GRID, exclusions=exclusions)

    assert acquisitions == []
    assert [item["item_id"] for item in exclusions] == [
        "sentinel-1d-20",
        "sentinel-1d-26",
    ]
    with pytest.raises(Sentinel1Error, match="Track counts"):
        select_pairs(acquisitions, EVENT_DATE)


def test_pair_selection_requires_same_track_direction_and_event_order() -> None:
    pre = _acquisition(20)
    post = _acquisition(27)
    selected, alternatives, _ = select_pairs([pre, post], EVENT_DATE)
    assert selected.pre is pre
    assert selected.post is post
    assert alternatives == []

    for incompatible in (
        _acquisition(27, orbit=46),
        _acquisition(27, direction="DESCENDING"),
    ):
        with pytest.raises(Sentinel1Error, match="relative orbit and direction"):
            select_pairs([pre, incompatible], EVENT_DATE)

    with pytest.raises(Sentinel1Error, match="Track counts"):
        select_pairs([_acquisition(25), _acquisition(25)], EVENT_DATE)


def test_pair_selection_applies_coverage_threshold() -> None:
    with pytest.raises(Sentinel1Error, match="95% bbox coverage"):
        select_pairs(
            [_acquisition(20, coverage=0.949), _acquisition(27)],
            EVENT_DATE,
        )


def test_pair_ranking_uses_time_then_repeat_cadence_then_platform() -> None:
    nearest_post = _acquisition(27)
    later_post = _acquisition(28)
    close_pre_other_platform = _acquisition(20, platform="sentinel-1b")
    close_pre_same_platform = _acquisition(20)
    chosen, ranked, _ = select_pairs(
        [
            close_pre_other_platform,
            close_pre_same_platform,
            _acquisition(19),
            nearest_post,
            later_post,
        ],
        EVENT_DATE,
    )

    assert chosen.post is nearest_post
    assert chosen.pre is close_pre_same_platform
    assert chosen.same_platform
    assert len(ranked) >= 1


def test_pair_ranking_prefers_six_day_repeat_interval_after_time_order() -> None:
    pre_six_day = _acquisition(20)
    post_six_day = _acquisition(26)
    post_other = _acquisition(26, hour=23)
    chosen, _, _ = select_pairs(
        [pre_six_day, post_six_day, post_other],
        EVENT_DATE,
    )

    assert chosen.pre is pre_six_day
    assert chosen.post is post_six_day
    assert chosen.repeat_interval_preferred


def test_pair_selection_prefers_same_platform_when_other_rank_fields_tie() -> None:
    pre_same = _acquisition(20, platform="sentinel-1a")
    pre_different = _acquisition(20, platform="sentinel-1b")
    post = _acquisition(26, platform="sentinel-1a")
    chosen, _, _ = select_pairs([pre_same, pre_different, post], EVENT_DATE)

    assert chosen.pre is pre_same
    assert chosen.same_platform


def test_baseline_stack_is_same_track_and_at_least_six_days_apart() -> None:
    chosen = AcquisitionPair(
        pre=_acquisition(20),
        post=_acquisition(26),
        gap_days=6.0,
        same_platform=True,
        repeat_interval_preferred=True,
    )
    candidates = [
        chosen.pre,
        chosen.post,
        _acquisition(14),
        _acquisition(13),
        _acquisition(6),
        _acquisition(10, orbit=46),
        _acquisition(8, direction="DESCENDING"),
        _acquisition(7, coverage=0.94),
    ]

    baseline = select_baseline(chosen, candidates, count=3)

    assert baseline[0] is chosen.pre
    assert len(baseline) == 3
    assert all(item.relative_orbit == 45 for item in baseline)
    assert all(
        abs((first.acquired_at - second.acquired_at).days) >= 6
        for index, first in enumerate(baseline)
        for second in baseline[index + 1 :]
    )


def test_process_tiles_partition_and_mosaic_exactly() -> None:
    windows = process_tile_windows(5, 4, max_dimension=2)
    arrays = [
        (
            tile,
            np.full(
                (5, tile.height, tile.width),
                tile.row_off * 10 + tile.col_off,
                dtype=np.float32,
            ),
        )
        for tile in windows
    ]

    mosaic = mosaic_process_tiles(5, 4, arrays)

    assert mosaic.shape == (5, 4, 5)
    assert np.all(mosaic[:, :2, :2] == 0)
    assert np.all(mosaic[:, 2:, 4:] == 24)
    with pytest.raises(Sentinel1Error, match="complete common grid"):
        mosaic_process_tiles(2, 2, [])


def test_process_grid_under_limit_uses_one_request_window() -> None:
    windows = process_tile_windows(1194, 1125)

    assert windows == [ProcessTile(row_off=0, col_off=0, height=1125, width=1194)]


def test_process_request_has_verified_processing_and_output_options() -> None:
    acquisition = _acquisition(27)
    tile = ProcessTile(row_off=3, col_off=4, height=5, width=6)

    request = build_process_request(acquisition, GRID, tile)
    data = request["input"]["data"][0]  # type: ignore[index]

    assert data["type"] == "sentinel-1-grd"
    assert data["dataFilter"]["resolution"] == "HIGH"
    assert data["dataFilter"]["acquisitionMode"] == "IW"
    assert data["dataFilter"]["polarization"] == "DV"
    assert data["dataFilter"]["orbitDirection"] == "ASCENDING"
    assert data["processing"] == {
        "backCoeff": "GAMMA0_TERRAIN",
        "orthorectify": "true",
        "demInstance": "COPERNICUS_30",
        "speckleFilter": {"type": "NONE"},
    }
    assert request["output"]["width"] == tile.width  # type: ignore[index]
    assert request["output"]["height"] == tile.height  # type: ignore[index]
    assert request["output"]["responses"][0]["format"]["type"] == "image/tiff"  # type: ignore[index]


def test_s1_cog_has_band_descriptions_and_valid_pixel_fraction(tmp_path: Path) -> None:
    grid = GridSpec(
        bbox=BBOX,
        epsg=32645,
        resolution=10,
        left=0,
        bottom=0,
        right=20,
        top=20,
    )
    data = np.ones((5, 2, 2), dtype=np.float32)
    data[4, 0, 0] = 0
    output = tmp_path / "s1.tif"

    valid_percent = write_s1_cog(output, data, grid)

    assert valid_percent == 75
    with rasterio.open(output) as dataset:
        assert dataset.driver == "GTiff"
        assert dataset.count == 5
        assert dataset.descriptions[0].startswith("VV")
        assert dataset.nodata == -9999


def test_ingest_registers_each_acquisition_and_checkpoints_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rasterio.io import MemoryFile

    import sahayak.ingest.s1 as s1_module

    small_grid = GridSpec(BBOX, 32645, 10, 0, 0, 20, 20)
    before = _acquisition(20)
    after = _acquisition(26)
    response_memory = MemoryFile()
    with response_memory.open(
        driver="GTiff",
        height=2,
        width=2,
        count=5,
        dtype="float32",
        crs="EPSG:32645",
        transform=from_origin(0, 20, 10, 10),
    ) as dataset:
        dataset.write(np.ones((5, 2, 2), dtype=np.float32))
    response_bytes = response_memory.read()

    class FakeClient:
        """Return a tiny fixed Process API response without network access."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """Expose the counters read by ingest_s1."""
            del args, kwargs
            self.processing_units: list[float] = []
            self.catalog_request_count = 1
            self.process_request_count = 0

        def process(self, body: dict[str, object]) -> bytes:
            """Return the same synthetic raster for each acquisition."""
            del body
            self.process_request_count += 1
            return response_bytes

    monkeypatch.setattr(s1_module, "SentinelHubClient", FakeClient)
    monkeypatch.setattr(
        s1_module,
        "catalog_acquisitions",
        lambda **kwargs: [before, after],
    )
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    manifest = RunManifest(
        run_id="synthetic",
        created_utc=datetime(2026, 8, 26, tzinfo=UTC),
        bbox=BBOX,
        event_date=EVENT_DATE,
    )
    checkpoints: list[dict[str, object]] = []

    result = ingest_s1(
        bbox=BBOX,
        event_date=EVENT_DATE,
        baseline_count=1,
        grid=small_grid,
        settings=settings,
        cache=RequestCache(settings),
        offline=False,
        output_dir=tmp_path / "runs",
        manifest=manifest,
        checkpoint=checkpoints.append,
    )
    response_memory.close()

    assert set(result["files"]) == {"s1_post.tif", "s1_pre.tif"}
    assert [record.name for record in manifest.inputs] == ["s1_post.tif", "s1_pre.tif"]
    assert all(record.role == "input" and record.sha256 for record in manifest.inputs)
    assert result["selection"]["pair"]["gap_days"] == 6
    assert len(checkpoints) >= 3
    assert all("s1" in checkpoint for checkpoint in checkpoints)


def test_acquisitions_require_awareness_of_pair_track_metadata() -> None:
    acquisition = _acquisition(20)

    assert acquisition.track.relative_orbit == 45
    assert acquisition.track.pass_direction == "ASCENDING"
