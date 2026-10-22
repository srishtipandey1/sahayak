"""Opt-in live smoke test for the configured ohsome metadata service."""

import os

import pytest

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.osm import OhsomeClient


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("RUN_LIVE") != "1", reason="Set RUN_LIVE=1 to access ohsome.")
def test_live_ohsome_metadata_returns_latest_timestamp(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path / "data", _env_file=None)
    client = OhsomeClient(settings, RequestCache(settings))

    assert client.latest_snapshot().year >= 2007
