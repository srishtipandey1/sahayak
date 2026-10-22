"""Tests for stable offline request caching."""

import pytest

from sahayak.config import Settings
from sahayak.ingest.cache import CacheMissError, RequestCache


def test_cache_key_is_stable_and_ignores_secret_values(tmp_path) -> None:
    cache = RequestCache(Settings(data_dir=tmp_path / "data", _env_file=None))
    request_a = {
        "url": "https://example.invalid",
        "auth_token": "secret-a",
        "params": {"bbox": "1,2"},
    }
    request_b = {
        "params": {"bbox": "1,2"},
        "auth_token": "secret-b",
        "url": "https://example.invalid",
    }

    assert cache.key(request_a) == cache.key(request_b)


def test_cache_fetches_then_serves_offline(tmp_path) -> None:
    cache = RequestCache(Settings(data_dir=tmp_path / "data", _env_file=None))
    request = {"endpoint": "metadata", "params": {"time": "2026-01-01"}}
    calls = 0

    def fetch() -> bytes:
        nonlocal calls
        calls += 1
        return b"response"

    online = cache.get_or_fetch(request, fetch, suffix=".json")
    offline = cache.get_or_fetch(request, fetch, suffix=".json", offline=True)

    assert online == offline == b"response"
    assert calls == 1
    assert cache.cached_at(request, ".json").tzinfo is not None


def test_offline_cache_miss_fails_clearly(tmp_path) -> None:
    cache = RequestCache(Settings(data_dir=tmp_path / "data", _env_file=None))

    with pytest.raises(CacheMissError, match="Offline cache miss"):
        cache.get_or_fetch({"request": "missing"}, lambda: b"unused", offline=True)
