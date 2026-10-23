"""Opt-in live smoke test for the public Overpass endpoint."""

import os
from datetime import date

import pytest

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.ingest.osm import OverpassProvider
from sahayak.ingest.s1 import SentinelHubClient, catalog_acquisitions


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="Set RUN_LIVE=1 to access Overpass.")
def test_live_overpass_returns_historical_timestamp(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    provider = OverpassProvider(settings, RequestCache(settings))
    result = provider.extract_tile((85.3, 28.1, 85.305, 28.105), date(2026, 8, 25))

    assert result.osm_base_timestamp is not None


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("RUN_LIVE") != "1",
    reason="Set RUN_LIVE=1 and configure CDSE OAuth credentials to access Catalog.",
)
def test_live_sentinel1_catalog_returns_grouped_acquisitions(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    client = SentinelHubClient(settings, RequestCache(settings))
    acquisitions = catalog_acquisitions(
        bbox=(85.30, 28.10, 85.42, 28.20),
        event_date=date(2026, 8, 26),
        grid=GridSpec.from_bbox((85.30, 28.10, 85.42, 28.20)),
        client=client,
    )

    assert acquisitions
