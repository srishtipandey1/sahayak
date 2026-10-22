"""Stable request-response cache for public input data."""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from sahayak.config import Settings
from sahayak.provenance import require_input


class CacheMissError(FileNotFoundError):
    """Raised when offline mode requests an item that is not cached."""


class RequestCache:
    """Cache response bytes under data/raw using a stable, secret-free request key."""

    def __init__(self, settings: Settings) -> None:
        """Initialize a cache rooted in the configured data directory."""
        self.settings = settings
        self.root = settings.resolved_data_dir / "raw"

    @staticmethod
    def key(request: dict[str, object]) -> str:
        """Hash canonical request data after removing fields that may hold secrets."""
        safe_request = _remove_secrets(request)
        canonical = json.dumps(safe_request, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def path_for(self, request: dict[str, object], suffix: str = ".cache") -> Path:
        """Return the cache path for a request."""
        safe_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return self.root / f"{self.key(request)}{safe_suffix}"

    def cached_at(self, request: dict[str, object], suffix: str = ".cache") -> datetime:
        """Return the cache file's recorded UTC modification time."""
        path = self.path_for(request, suffix)
        if not path.is_file():
            raise FileNotFoundError(f"Cache entry does not exist: {self.key(request)}.")
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)

    def get_or_fetch(
        self,
        request: dict[str, object],
        fetch: Callable[[], bytes],
        *,
        suffix: str = ".cache",
        offline: bool = False,
    ) -> bytes:
        """Return cached bytes or fetch and cache them, failing clearly offline."""
        path = self.path_for(request, suffix)
        if path.is_file():
            return require_input(path, self.settings).read_bytes()
        if offline:
            raise CacheMissError(f"Offline cache miss for request {self.key(request)}.")

        payload = fetch()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return payload


def _remove_secrets(value: object) -> object:
    """Recursively remove secret-like mapping keys before creating cache keys."""
    if isinstance(value, dict):
        return {
            key: _remove_secrets(item)
            for key, item in value.items()
            if not any(
                sensitive in str(key).lower()
                for sensitive in ("auth", "password", "secret", "token", "credential", "api_key")
            )
        }
    if isinstance(value, (list, tuple)):
        return [_remove_secrets(item) for item in value]
    return value
