"""Opt-in live smoke test for the public Overpass endpoint."""

import os
from datetime import date

import pytest

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.osm import OverpassProvider


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="Set RUN_LIVE=1 to access Overpass.")
def test_live_overpass_returns_historical_timestamp(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    provider = OverpassProvider(settings, RequestCache(settings))
    result = provider.extract_tile((85.3, 28.1, 85.305, 28.105), date(2026, 8, 25))

    assert result.osm_base_timestamp is not None
