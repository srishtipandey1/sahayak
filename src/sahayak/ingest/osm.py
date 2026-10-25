"""Provider-based historical OSM extraction and GeoPackage creation."""

from __future__ import annotations

import json
import logging
import math
import re
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from email.utils import parsedate_to_datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import geopandas as gpd
import pyogrio
import requests
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, polygonize, unary_union

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.provenance import (
    RuleViolation,
    RunManifest,
    assert_osm_snapshot_valid,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)
OSM_ATTRIBUTION = (
    "© OpenStreetMap contributors (ODbL). "
    "Any published modified OSM data stays under ODbL."
)
OVERPASS_SOURCE = "OpenStreetMap via Overpass API (historical, date-pinned)"
OHSOME_SOURCE = "OpenStreetMap via ohsome API"
OSM_LICENSE = f"Open Database License (ODbL); {OSM_ATTRIBUTION}"
OVERPASS_USER_AGENT = "sahayak-hackathon-prototype"
OVERPASS_RETRY_STATUSES = {429, 502, 503, 504}
OVERPASS_RETRY_COUNT = 5
OVERPASS_TIMEOUT_SECONDS = 180
OVERPASS_REQUEST_INTERVAL_SECONDS = 1.0
MAX_TILE_DEGREES = 0.05
LAYER_NAMES = ("buildings", "roads", "bridges", "settlements", "health_facilities")
SETTLEMENT_VALUES = ("village", "hamlet", "town", "city", "suburb", "locality")
AMENITY_VALUES = ("hospital", "clinic", "doctors")
HEALTHCARE_VALUES = ("hospital", "clinic")
LAYER_COLUMNS = {
    "buildings": (
        "building",
        "name",
        "amenity",
        "height",
        "building_levels",
    ),
    "roads": (
        "highway",
        "road_class",
        "name",
        "surface",
        "bridge",
        "tunnel",
        "layer",
        "ref",
        "oneway",
        "is_bridge",
        "is_tunnel",
    ),
    "bridges": ("bridge", "man_made", "highway", "name", "layer"),
    "settlements": ("place", "name", "population"),
    "health_facilities": ("amenity", "healthcare", "name", "emergency"),
}
COMMON_COLUMNS = ("osm_type", "osm_id", "osm_timestamp", "osm_version")
TAG_COLUMN_MAP = {"building:levels": "building_levels"}
LAYER_FILTERS = {
    "buildings": "building=*",
    "roads": "highway=* and geometry:line",
    "bridges": "(bridge=* or man_made=bridge)",
    "settlements": f"place in ({','.join(SETTLEMENT_VALUES)})",
    "health_facilities": (
        f"(amenity in ({','.join(AMENITY_VALUES)}) or "
        f"healthcare in ({','.join(HEALTHCARE_VALUES)}))"
    ),
}


class OSMProviderError(RuntimeError):
    """Raised when an OSM provider cannot return valid extraction data."""


class OhsomeError(OSMProviderError):
    """Raised when a configured ohsome service fails or returns unusable data."""


class Provider(Protocol):
    """Common interface for providers returning historical OSM tile features."""

    provider_name: str
    endpoint: str
    source: str

    def latest_snapshot(self) -> datetime | None:
        """Return the latest source timestamp when the provider exposes one."""

    def extract_tile(
        self,
        bbox: tuple[float, float, float, float],
        snapshot_date: date,
    ) -> TileResult:
        """Extract all required OSM layer features for a single tile."""


class TileResult:
    """Parsed layer features and source metadata for one requested tile."""

    def __init__(
        self,
        layers: dict[str, list[dict[str, Any]]],
        *,
        osm_base_timestamp: str | None = None,
        skipped_relations: dict[str, int] | None = None,
        skipped_relation_ids: dict[str, set[tuple[str, int]]] | None = None,
        element_count: int = 0,
        max_element_timestamp: datetime | None = None,
        per_layer_max_timestamps: dict[str, datetime | None] | None = None,
    ) -> None:
        """Store parsed layers, timestamp and relation-skip counts."""
        self.layers = layers
        self.osm_base_timestamp = osm_base_timestamp
        self.skipped_relations = skipped_relations or {name: 0 for name in LAYER_NAMES}
        self.skipped_relation_ids = skipped_relation_ids or {
            name: set() for name in LAYER_NAMES
        }
        self.element_count = element_count
        self.max_element_timestamp = max_element_timestamp
        self.per_layer_max_timestamps = per_layer_max_timestamps or {
            name: None for name in LAYER_NAMES
        }


def tile_bboxes(
    bbox: tuple[float, float, float, float],
    max_tile_degrees: float = MAX_TILE_DEGREES,
) -> list[tuple[float, float, float, float]]:
    """Split a WGS84 bbox into contiguous tiles no larger than the requested size."""
    min_lon, min_lat, max_lon, max_lat = bbox
    if max_tile_degrees <= 0:
        raise ValueError("Tile size must be positive.")
    lon_steps = max(1, math.ceil((max_lon - min_lon) / max_tile_degrees))
    lat_steps = max(1, math.ceil((max_lat - min_lat) / max_tile_degrees))
    lon_edges = [
        min_lon + (max_lon - min_lon) * index / lon_steps
        for index in range(lon_steps + 1)
    ]
    lat_edges = [
        min_lat + (max_lat - min_lat) * index / lat_steps
        for index in range(lat_steps + 1)
    ]
    return [
        (lon_edges[col], lat_edges[row], lon_edges[col + 1], lat_edges[row + 1])
        for row in range(lat_steps)
        for col in range(lon_steps)
    ]


def build_overpass_query(
    bbox: tuple[float, float, float, float],
    snapshot_date: date,
    *,
    timeout_seconds: int = OVERPASS_TIMEOUT_SECONDS,
) -> str:
    """Build one date-pinned JSON Overpass query for all requested OSM layer classes."""
    west, south, east, north = bbox
    box = f"{south:.8f},{west:.8f},{north:.8f},{east:.8f}"
    settlements = "|".join(SETTLEMENT_VALUES)
    amenities = "|".join(AMENITY_VALUES)
    healthcare = "|".join(HEALTHCARE_VALUES)
    return (
        f'[out:json][timeout:{timeout_seconds}]'
        f'[date:"{snapshot_date.isoformat()}T00:00:00Z"];'
        "("
        f'nwr["building"]({box});'
        f'way["highway"]({box});'
        f'nwr["bridge"]({box});'
        f'nwr["man_made"="bridge"]({box});'
        f'relation["type"="bridge"]({box});'
        f'nwr["place"~"^({settlements})$"]({box});'
        f'nwr["amenity"~"^({amenities})$"]({box});'
        f'nwr["healthcare"~"^({healthcare})$"]({box});'
        ");out meta geom;"
    )


def _coords(element: dict[str, Any]) -> list[tuple[float, float]]:
    """Return longitude/latitude pairs from a node or way element."""
    if element.get("type") == "node":
        if "lon" not in element or "lat" not in element:
            return []
        return [(float(element["lon"]), float(element["lat"]))]
    return [
        (float(point["lon"]), float(point["lat"]))
        for point in element.get("geometry", [])
        if "lon" in point and "lat" in point
    ]


def _closed_polygon(coords: list[tuple[float, float]]) -> Polygon | None:
    """Return a valid closed polygon from coordinates, or None if not polygonal."""
    if len(coords) < 4 or coords[0] != coords[-1]:
        return None
    polygon = Polygon(coords)
    if polygon.is_empty:
        return None
    return polygon if polygon.is_valid else polygon.buffer(0)


def _simple_relation_geometry(
    element: dict[str, Any],
    *,
    polygonal: bool,
) -> Any | None:
    """Assemble simple relation-member lines into a polygon or line geometry."""
    lines: dict[str, list[LineString]] = defaultdict(list)
    for member in element.get("members", []):
        if member.get("type") != "way":
            continue
        coordinates = _coords({**member, "type": "way"})
        if len(coordinates) < 2:
            continue
        lines[str(member.get("role", ""))].append(LineString(coordinates))

    if not polygonal:
        member_lines = [line for role_lines in lines.values() for line in role_lines]
        if not member_lines:
            return None
        merged = unary_union(member_lines)
        return merged if not merged.is_empty else None

    outer_lines = lines.get("outer", [])
    if not outer_lines:
        return None
    outer_polygons = list(polygonize(unary_union(outer_lines)))
    if not outer_polygons:
        merged = linemerge(unary_union(outer_lines))
        merged_lines = [merged] if isinstance(merged, LineString) else list(
            merged.geoms if isinstance(merged, MultiLineString) else []
        )
        outer_polygons = [
            polygon
            for line in merged_lines
            for polygon in polygonize(line)
        ]
    if not outer_polygons:
        return None

    inner_polygons = list(polygonize(unary_union(lines.get("inner", []))))
    assembled = []
    for outer in outer_polygons:
        holes = [
            list(inner.exterior.coords)
            for inner in inner_polygons
            if outer.contains(inner.representative_point())
        ]
        assembled.append(Polygon(outer.exterior.coords, holes))
    result = unary_union(assembled)
    return result if not result.is_empty and result.is_valid else None


def _is_building(tags: dict[str, Any]) -> bool:
    """Return whether the tags represent a mapped building."""
    return bool(tags.get("building"))


def _is_road(tags: dict[str, Any], element_type: str) -> bool:
    """Return whether the element is a road way with a highway tag."""
    return element_type == "way" and bool(tags.get("highway"))


def _is_bridge(tags: dict[str, Any], element_type: str) -> bool:
    """Return whether a way or relation is tagged as a bridge."""
    return element_type in {"node", "way", "relation"} and (
        bool(tags.get("bridge"))
        or tags.get("man_made") == "bridge"
        or (element_type == "relation" and tags.get("type") == "bridge")
    )


def _is_settlement(tags: dict[str, Any]) -> bool:
    """Return whether a feature has one of the selected place values."""
    return tags.get("place") in SETTLEMENT_VALUES


def _is_health_facility(tags: dict[str, Any]) -> bool:
    """Return whether a feature has one of the selected healthcare tags."""
    return tags.get("amenity") in AMENITY_VALUES or tags.get("healthcare") in HEALTHCARE_VALUES


def _metadata_timestamp(element: dict[str, Any]) -> datetime:
    """Parse a required Overpass metadata timestamp as UTC."""
    raw_timestamp = element.get("timestamp")
    if not isinstance(raw_timestamp, str):
        raise RuleViolation(
            "Overpass metadata omitted an element timestamp; cannot verify the OSM snapshot."
        )
    try:
        parsed = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuleViolation(
            f"Overpass returned an invalid element timestamp {raw_timestamp!r}."
        ) from error
    if parsed.tzinfo is None:
        raise RuleViolation(
            f"Overpass returned a timezone-free element timestamp {raw_timestamp!r}."
        )
    return parsed.astimezone(UTC)


def _tag_boolean(value: Any) -> bool:
    """Interpret common affirmative OSM tag values as true."""
    return value is not None and str(value).strip().casefold() not in {
        "",
        "no",
        "false",
        "0",
    }


def _feature_geometry(element: dict[str, Any], layer: str) -> Any | None:
    """Convert an OSM node, way, or relation to the requested Shapely geometry."""
    element_type = element.get("type")
    if element_type == "node":
        coordinates = _coords(element)
        return Point(coordinates[0]) if coordinates else None
    if element_type == "relation":
        polygonal = layer in {"buildings", "settlements", "health_facilities"}
        geometry = _simple_relation_geometry(element, polygonal=polygonal)
        if layer == "health_facilities" and geometry is not None:
            return geometry.centroid
        return geometry

    coordinates = _coords(element)
    if len(coordinates) < 2:
        return None
    polygon = _closed_polygon(coordinates)
    if layer in {"buildings", "settlements", "health_facilities"} and polygon is not None:
        if layer == "health_facilities":
            return polygon.centroid
        return polygon
    if layer == "health_facilities":
        return LineString(coordinates).centroid
    return LineString(coordinates)


def _layer_memberships(element: dict[str, Any]) -> list[str]:
    """Classify an OSM element into all relevant output layers."""
    tags = element.get("tags") or {}
    element_type = str(element.get("type", ""))
    memberships = []
    if _is_building(tags):
        memberships.append("buildings")
    if _is_road(tags, element_type):
        memberships.append("roads")
    if _is_bridge(tags, element_type):
        memberships.append("bridges")
    if _is_settlement(tags):
        memberships.append("settlements")
    if _is_health_facility(tags):
        memberships.append("health_facilities")
    return memberships


def _normalise_road_class(value: Any) -> str | None:
    """Normalize highway values while retaining all valid classes, including track/path."""
    if value is None:
        return None
    return re.sub(r"[\s-]+", "_", str(value).strip().casefold()) or None


def parse_overpass_payload(payload: bytes | str) -> TileResult:
    """Parse Overpass JSON elements into OSM layer records and relation skip counts."""
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise OSMProviderError(f"Overpass returned invalid JSON: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
        raise OSMProviderError("Overpass JSON response must contain an elements array.")

    osm3s = data.get("osm3s") or {}
    osm_base_timestamp = osm3s.get("timestamp_osm_base")
    layers: dict[str, list[dict[str, Any]]] = {name: [] for name in LAYER_NAMES}
    skipped: dict[str, int] = {name: 0 for name in LAYER_NAMES}
    skipped_ids: dict[str, set[tuple[str, int]]] = {name: set() for name in LAYER_NAMES}
    per_layer_max_timestamps: dict[str, datetime | None] = {
        name: None for name in LAYER_NAMES
    }
    max_element_timestamp: datetime | None = None

    for element in data["elements"]:
        if not isinstance(element, dict):
            raise RuleViolation(
                "Overpass response contained a malformed element without verifiable metadata."
            )
        timestamp = _metadata_timestamp(element)
        max_element_timestamp = max(
            max_element_timestamp, timestamp
        ) if max_element_timestamp is not None else timestamp
        memberships = _layer_memberships(element)
        if not memberships:
            continue
        for layer in memberships:
            current_max = per_layer_max_timestamps[layer]
            per_layer_max_timestamps[layer] = (
                max(current_max, timestamp) if current_max is not None else timestamp
            )
            geometry = _feature_geometry(element, layer)
            if geometry is None or geometry.is_empty:
                if element.get("type") == "relation":
                    skipped[layer] += 1
                    skipped_ids[layer].add(
                        (str(element.get("type")), int(element.get("id", -1)))
                    )
                continue
            tags = dict(element.get("tags") or {})
            record: dict[str, Any] = {
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "osm_timestamp": timestamp.isoformat(),
                "osm_version": element.get("version"),
                "tags": tags,
                "geometry": geometry,
            }
            if layer == "roads":
                record["road_class"] = _normalise_road_class(tags.get("highway"))
                record["is_bridge"] = _tag_boolean(tags.get("bridge"))
                record["is_tunnel"] = _tag_boolean(tags.get("tunnel"))
            if layer == "settlements" and geometry.geom_type in {"Polygon", "MultiPolygon"}:
                record["geometry"] = geometry.centroid
            layers[layer].append(record)

    return TileResult(
        layers,
        osm_base_timestamp=str(osm_base_timestamp) if osm_base_timestamp else None,
        skipped_relations=skipped,
        skipped_relation_ids=skipped_ids,
        element_count=len(data["elements"]),
        max_element_timestamp=max_element_timestamp,
        per_layer_max_timestamps=per_layer_max_timestamps,
    )


def select_osm_provider(
    settings: Settings,
    cache: RequestCache,
    *,
    offline: bool,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Provider:
    """Create the provider selected by OSM_PROVIDER, defaulting to Overpass."""
    provider_name = settings.osm_provider.strip().lower()
    if provider_name == "overpass":
        return OverpassProvider(settings, cache, offline=offline, session=session, sleep=sleep)
    if provider_name == "ohsome":
        return OhsomeProvider(settings, cache, offline=offline, session=session)
    raise OSMProviderError(
        f"Unsupported OSM_PROVIDER {settings.osm_provider!r}; choose 'overpass' or 'ohsome'."
    )


class OverpassProvider:
    """Historical OSM provider using polite, date-pinned Overpass API requests."""

    provider_name = "overpass"
    source = OVERPASS_SOURCE

    def __init__(
        self,
        settings: Settings,
        cache: RequestCache,
        *,
        offline: bool = False,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        retry_count: int = OVERPASS_RETRY_COUNT,
        request_interval: float = OVERPASS_REQUEST_INTERVAL_SECONDS,
        timeout: float = OVERPASS_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize provider request, retry, and cache configuration."""
        self.settings = settings
        self.cache = cache
        self.offline = offline
        self.session = session or requests.Session()
        self.sleep = sleep
        self.retry_count = retry_count
        self.request_interval = request_interval
        self.timeout = timeout
        self.endpoint = settings.overpass_url
        self.request_count = 0
        self.elapsed_seconds = 0.0
        self.osm_base_timestamps: list[str] = []
        self.skipped_relation_ids: dict[str, set[tuple[str, int]]] = {
            layer: set() for layer in LAYER_NAMES
        }
        self.element_count = 0
        self.max_element_timestamp: datetime | None = None
        self.per_layer_max_timestamps: dict[str, datetime | None] = {
            layer: None for layer in LAYER_NAMES
        }
        self._last_request_finished: float | None = None

    def latest_snapshot(self) -> datetime | None:
        """Overpass provides its data base timestamp alongside extraction results."""
        return None

    def _wait_for_request_slot(self) -> None:
        """Keep this client sequential and insert a respectful gap between requests."""
        if self._last_request_finished is None:
            return
        wait_seconds = self.request_interval - (time.monotonic() - self._last_request_finished)
        if wait_seconds > 0:
            self.sleep(wait_seconds)

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        """Parse Retry-After as seconds or an HTTP date."""
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
            except (TypeError, ValueError, OverflowError):
                return None
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=UTC)
            return max(0.0, (retry_at - datetime.now(UTC)).total_seconds())

    def _request_bytes(self, query: str) -> bytes:
        """POST one query with retry/backoff while never logging query or headers."""
        self._wait_for_request_slot()
        started = time.monotonic()
        for attempt in range(self.retry_count + 1):
            self.request_count += 1
            try:
                response = self.session.post(
                    self.endpoint,
                    data={"data": query},
                    headers={"User-Agent": OVERPASS_USER_AGENT},
                    timeout=self.timeout,
                )
            except requests.RequestException as error:
                self.elapsed_seconds += time.monotonic() - started
                self._last_request_finished = time.monotonic()
                raise OSMProviderError(f"Overpass request failed: {error}") from error

            if response.status_code == 200:
                self.elapsed_seconds += time.monotonic() - started
                self._last_request_finished = time.monotonic()
                return response.content
            if response.status_code not in OVERPASS_RETRY_STATUSES or attempt >= self.retry_count:
                self.elapsed_seconds += time.monotonic() - started
                self._last_request_finished = time.monotonic()
                raise OSMProviderError(
                    f"Overpass endpoint {self.endpoint} returned HTTP {response.status_code}: "
                    f"{response.text[:300]}"
                )

            backoff = 2**attempt
            retry_after = self._retry_after_seconds(response.headers.get("Retry-After"))
            self.sleep(max(backoff, retry_after or 0.0))

        raise OSMProviderError("Overpass request retries were exhausted.")

    def extract_tile(
        self,
        bbox: tuple[float, float, float, float],
        snapshot_date: date,
    ) -> TileResult:
        """Return the cached or newly requested, parsed results for one tile."""
        query = build_overpass_query(bbox, snapshot_date, timeout_seconds=int(self.timeout))
        request_identity = {
            "provider": self.provider_name,
            "endpoint": self.endpoint,
            "query": query,
        }
        try:
            payload = self.cache.get_or_fetch(
                request_identity,
                lambda: self._request_bytes(query),
                suffix=".json",
                offline=self.offline,
            )
        except OSMProviderError:
            raise
        except OSError as error:
            raise OSMProviderError(f"Overpass cache access failed: {error}") from error

        result = parse_overpass_payload(payload)
        if result.osm_base_timestamp:
            self.osm_base_timestamps.append(result.osm_base_timestamp)
        for layer, identifiers in result.skipped_relation_ids.items():
            self.skipped_relation_ids[layer].update(identifiers)
        return result


class OhsomeProvider:
    """Authenticated ohsome API v2 extraction provider."""

    provider_name = "ohsome"
    source = OHSOME_SOURCE

    def __init__(
        self,
        settings: Settings,
        cache: RequestCache,
        *,
        offline: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the configured ohsome version and authentication."""
        self.settings = settings
        self.cache = cache
        self.offline = offline
        self.session = session or requests.Session()
        self.api_version = settings.ohsome_api_version.strip().lower()
        self.base_url = settings.ohsome_url.rstrip("/")
        self.endpoint = self.base_url
        if self.api_version == "v1":
            raise OhsomeError(
                "ohsome v1 geometry extraction is currently restricted (HTTP 403); "
                "use OSM_PROVIDER=overpass or configure ohsome v2."
            )
        if self.api_version != "v2":
            raise OhsomeError("OHSOME_API_VERSION must be v2; v1 geometry is restricted.")
        if settings.ohsome_api_key is None or not settings.ohsome_api_key.get_secret_value():
            raise OhsomeError(
                "OHSOME_API_KEY is required for ohsome v2. Obtain a key at "
                "https://account.heigit.org/signup."
            )
        self.request_count = 0

    def _headers(self) -> dict[str, str]:
        """Build required authorization header without exposing it in diagnostics."""
        assert self.settings.ohsome_api_key is not None
        return {"Authorization": self.settings.ohsome_api_key.get_secret_value()}

    def _get_bytes(
        self,
        endpoint: str,
        parameters: dict[str, Any] | None,
        *,
        method: str,
        suffix: str,
    ) -> bytes:
        """Fetch a cached ohsome v2 response and redact credentials on errors."""
        identity = {
            "provider": self.provider_name,
            "method": method,
            "endpoint": endpoint,
            "parameters": parameters or {},
        }

        def fetch() -> bytes:
            self.request_count += 1
            try:
                if method == "GET":
                    response = self.session.get(
                        endpoint,
                        params=parameters,
                        headers=self._headers(),
                        timeout=240,
                    )
                else:
                    response = self.session.post(
                        endpoint,
                        json=parameters,
                        headers=self._headers(),
                        timeout=240,
                    )
                response.raise_for_status()
            except requests.RequestException as error:
                raise OhsomeError(
                    f"Configured ohsome v2 endpoint failed ({endpoint}): "
                    f"HTTP {getattr(error.response, 'status_code', 'request error')}."
                ) from None
            return response.content

        try:
            return self.cache.get_or_fetch(identity, fetch, suffix=suffix, offline=self.offline)
        except OhsomeError:
            raise
        except OSError as error:
            message = f"Could not use ohsome v2 request cache: {type(error).__name__}."
            raise OhsomeError(message) from None

    def latest_snapshot(self) -> datetime:
        """Query ohsome metadata for the latest extract timestamp."""
        payload = self._get_bytes(
            f"{self.base_url}/metadata",
            None,
            method="GET",
            suffix=".json",
        )
        try:
            metadata = json.loads(payload)
            timestamp = metadata["extractRegion"]["temporalExtent"]["toTimestamp"]
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as error:
            message = "ohsome v2 metadata omitted a valid latest extract timestamp."
            raise OhsomeError(message) from error
        return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)

    def extract_tile(
        self,
        bbox: tuple[float, float, float, float],
        snapshot_date: date,
    ) -> TileResult:
        """Request v2 GeoParquet layers for the requested bbox and snapshot."""
        payload_layers: dict[str, list[dict[str, Any]]] = {name: [] for name in LAYER_NAMES}
        skipped = {name: 0 for name in LAYER_NAMES}
        filters = {
            "buildings": 'building=*',
            "roads": 'highway=* and geometry:line',
            "bridges": '(bridge=* or man_made=bridge)',
            "settlements": f"place in ({','.join(SETTLEMENT_VALUES)})",
            "health_facilities": (
                f"(amenity in ({','.join(AMENITY_VALUES)}) or "
                f"healthcare in ({','.join(HEALTHCARE_VALUES)}))"
            ),
        }
        for layer, filter_text in filters.items():
            payload = self._get_bytes(
                f"{self.base_url}/extraction/features",
                {
                    "aoi": list(bbox),
                    "time": snapshot_date.isoformat(),
                    "filter": filter_text,
                },
                method="POST",
                suffix=".parquet",
            )
            try:
                frame = gpd.read_parquet(BytesIO(payload))
            except (ValueError, OSError, ImportError) as error:
                raise OhsomeError(
                    f"Could not decode ohsome v2 GeoParquet for layer {layer}: "
                    f"{type(error).__name__}."
                ) from None
            for row in frame.to_dict(orient="records"):
                geometry = row.pop("geometry", None)
                if geometry is None:
                    continue
                row["geometry"] = geometry
                if layer == "roads":
                    row["road_class"] = _normalise_road_class(row.get("highway"))
                if layer == "settlements" and geometry.geom_type in {"Polygon", "MultiPolygon"}:
                    row["geometry"] = geometry.centroid
                payload_layers[layer].append(row)
        return TileResult(payload_layers, skipped_relations=skipped)


# Kept as a compatibility alias for callers of the initial ohsome client.
OhsomeClient = OhsomeProvider


def _frame_from_records(
    records: list[dict[str, Any]],
    crs: str,
    layer_name: str,
) -> gpd.GeoDataFrame:
    """Build a layer frame using only validated fixed columns and JSON-encoded tags."""
    columns = (*COMMON_COLUMNS, *LAYER_COLUMNS[layer_name], "tags_json")
    if any(re.fullmatch(r"[a-z0-9_]+", name) is None for name in columns):
        raise ValueError(f"Unsafe configured GeoPackage column in layer {layer_name}.")

    rows: list[dict[str, Any]] = []
    for record in records:
        tags = record.get("tags", {})
        row: dict[str, Any] = {
            "osm_type": record.get("osm_type"),
            "osm_id": record.get("osm_id"),
            "osm_timestamp": record.get("osm_timestamp"),
            "osm_version": record.get("osm_version"),
            "tags_json": json.dumps(tags, ensure_ascii=False, sort_keys=True),
        }
        for column in LAYER_COLUMNS[layer_name]:
            tag_name = next(
                (tag for tag, mapped_column in TAG_COLUMN_MAP.items() if mapped_column == column),
                column,
            )
            if column == "road_class":
                row[column] = record.get("road_class")
            elif column in {"is_bridge", "is_tunnel"}:
                row[column] = bool(record.get(column, False))
            else:
                row[column] = tags.get(tag_name)
        rows.append({**row, "geometry": record["geometry"]})

    frame = gpd.GeoDataFrame(
        rows,
        columns=(*columns, "geometry"),
        geometry="geometry",
        crs="EPSG:4326",
    )
    return frame.to_crs(crs)


def _write_gpkg_layer(
    frame: gpd.GeoDataFrame,
    path: Path,
    layer_name: str,
    *,
    append: bool,
    snapshot_date: date,
) -> None:
    """Write one layer to the shared GeoPackage with its snapshot metadata."""
    pyogrio.write_dataframe(
        frame,
        path,
        layer=layer_name,
        driver="GPKG",
        append=append,
        layer_metadata={"osm_snapshot_date": snapshot_date.isoformat()},
    )


def _persist_osm_checkpoint(
    *,
    output_path: Path,
    layer_name: str,
    count: int,
    snapshot_date: date,
    provider: Provider,
    manifest: RunManifest,
    settings: Settings,
    checkpoint: Callable[[str, int], None] | None,
) -> None:
    """Record a completed GeoPackage layer before continuing to another layer."""
    current_digest = sha256_file(output_path)
    current_time = datetime.now(UTC)
    manifest.inputs = [
        item.model_copy(
            update={"sha256": current_digest, "retrieved_utc": current_time}
        )
        if item.name.startswith("osm:")
        else item
        for item in manifest.inputs
    ]
    manifest.inputs = [item for item in manifest.inputs if item.name != f"osm:{layer_name}"]
    manifest.add_input(
        name=f"osm:{layer_name}",
        role="input",
        source=provider.source,
        license=OSM_LICENSE,
        retrieved_utc=current_time,
        path=output_path,
        settings=settings,
    )
    if checkpoint is not None:
        checkpoint(layer_name, count)


def _overpass_result_layers(
    provider: OverpassProvider,
    bbox: tuple[float, float, float, float],
    snapshot_date: date,
    *,
    max_tile_degrees: float = MAX_TILE_DEGREES,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Fetch and de-duplicate whole OSM elements from every Overpass tile."""
    features: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        layer: {} for layer in LAYER_NAMES
    }
    skipped: dict[str, set[tuple[str, int]]] = {layer: set() for layer in LAYER_NAMES}
    tiles = tile_bboxes(bbox, max_tile_degrees=max_tile_degrees)
    started = time.monotonic()
    for tile_index, tile in enumerate(tiles, start=1):
        result = provider.extract_tile(tile, snapshot_date)
        provider.element_count += result.element_count
        if result.max_element_timestamp is not None:
            current_max = provider.max_element_timestamp
            provider.max_element_timestamp = (
                max(current_max, result.max_element_timestamp)
                if current_max is not None
                else result.max_element_timestamp
            )
        for layer in LAYER_NAMES:
            for record in result.layers[layer]:
                identifier = (str(record["osm_type"]), int(record["osm_id"]))
                features[layer].setdefault(identifier, record)
            skipped[layer].update(result.skipped_relation_ids.get(layer, set()))
            layer_timestamp = result.per_layer_max_timestamps.get(layer)
            if layer_timestamp is not None:
                current_layer_max = provider.per_layer_max_timestamps[layer]
                provider.per_layer_max_timestamps[layer] = (
                    max(current_layer_max, layer_timestamp)
                    if current_layer_max is not None
                    else layer_timestamp
                )
        LOGGER.info(
            "Overpass tile %d of %d complete: elapsed %.1fs, %d elements",
            tile_index,
            len(tiles),
            time.monotonic() - started,
            result.element_count,
        )
    provider.skipped_relation_ids = skipped
    return (
        {layer: list(records.values()) for layer, records in features.items()},
        {layer: len(identifiers) for layer, identifiers in skipped.items()},
    )


def verify_osm_timestamp_compliance(
    max_element_timestamp: datetime | None,
    snapshot_date: date,
) -> None:
    """Reject missing or post-snapshot element metadata before writing artifacts."""
    if max_element_timestamp is None:
        return
    if max_element_timestamp.tzinfo is None:
        raise RuleViolation("The maximum OSM element timestamp has no timezone.")
    snapshot_instant = datetime.combine(snapshot_date, datetime_time.min, UTC)
    maximum_utc = max_element_timestamp.astimezone(UTC)
    if maximum_utc > snapshot_instant:
        raise RuleViolation(
            f"OSM element timestamp {maximum_utc.isoformat()} is later than snapshot instant "
            f"{snapshot_instant.isoformat()}."
        )


def _extract_ohsome(
    provider: OhsomeProvider,
    bbox: tuple[float, float, float, float],
    snapshot_date: date,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Fetch ohsome layers tile-by-tile and de-duplicate source identifiers."""
    features: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        layer: {} for layer in LAYER_NAMES
    }
    for tile in tile_bboxes(bbox):
        result = provider.extract_tile(tile, snapshot_date)
        for layer, records in result.layers.items():
            for index, record in enumerate(records):
                osm_type = str(record.get("osm_type", record.get("@osmType", "feature")))
                osm_id = str(record.get("osm_id", record.get("@osmId", index)))
                features[layer].setdefault((osm_type, osm_id), record)
    return (
        {layer: list(records.values()) for layer, records in features.items()},
        {layer: 0 for layer in LAYER_NAMES},
    )


def _write_osm_layers(
    *,
    features: dict[str, list[dict[str, Any]]],
    snapshot_date: date,
    grid: GridSpec,
    output_path: Path,
    provider: Provider,
    manifest: RunManifest,
    settings: Settings,
    checkpoint: Callable[[str, int], None] | None,
) -> dict[str, int]:
    """Atomically write the complete GeoPackage and persist layer provenance."""
    counts: dict[str, int] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    with tempfile.TemporaryDirectory(
        prefix=".osm_pre_event.", dir=output_path.parent
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / output_path.name
        for index, layer_name in enumerate(LAYER_NAMES):
            frame = _frame_from_records(features[layer_name], grid.crs, layer_name)
            counts[layer_name] = len(frame)
            _write_gpkg_layer(
                frame,
                temporary_path,
                layer_name,
                append=index > 0,
                snapshot_date=snapshot_date,
            )
        temporary_path.replace(output_path)

    for layer_name in LAYER_NAMES:
        _persist_osm_checkpoint(
            output_path=output_path,
            layer_name=layer_name,
            count=counts[layer_name],
            snapshot_date=snapshot_date,
            provider=provider,
            manifest=manifest,
            settings=settings,
            checkpoint=(
                lambda layer=layer_name, layer_count=counts[layer_name]: checkpoint(
                    layer, layer_count
                )
                if checkpoint is not None
                else None
            ),
        )
    return counts


def ingest_osm(
    *,
    bbox: tuple[float, float, float, float],
    event_date: date,
    requested_snapshot: date | None,
    grid: GridSpec,
    settings: Settings,
    cache: RequestCache,
    offline: bool,
    output_dir: Path,
    manifest: RunManifest,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Extract all pre-event OSM layers using the configured provider."""
    provider = select_osm_provider(
        settings,
        cache,
        offline=offline,
        session=session,
        sleep=sleep,
    )
    latest_available = provider.latest_snapshot()
    if isinstance(provider, OverpassProvider):
        snapshot_date = requested_snapshot or date.fromordinal(event_date.toordinal() - 1)
        assert_osm_snapshot_valid(snapshot_date, event_date)
        if latest_available and snapshot_date > latest_available.date():
            raise OSMProviderError(
                f"Requested snapshot {snapshot_date} is later than the latest available "
                f"Overpass timestamp {latest_available.isoformat()}."
            )
    else:
        if latest_available is None:
            raise OSMProviderError(
                "The selected provider did not report its latest available snapshot."
            )
        snapshot_date = select_ohsome_snapshot(event_date, latest_available, requested_snapshot)

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if checkpoint is not None:
        checkpoint(
            {
                "osm_provider": provider.provider_name,
                "osm_endpoint": provider.endpoint,
                "osm_snapshot_date": snapshot_date.isoformat(),
                "osm_base_timestamp": None,
                "osm_base_timestamps": [],
                "osm_counts": {},
                "osm_skipped_relations": {},
            }
        )

    if isinstance(provider, OverpassProvider):
        features, skipped_relations = _overpass_result_layers(provider, bbox, snapshot_date)
        verify_osm_timestamp_compliance(provider.max_element_timestamp, snapshot_date)
        if checkpoint is not None:
            timestamps = {
                layer: timestamp.isoformat() if timestamp is not None else None
                for layer, timestamp in provider.per_layer_max_timestamps.items()
            }
            checkpoint(
                {
                    "osm_base_timestamps": list(dict.fromkeys(provider.osm_base_timestamps)),
                    "max_element_timestamp": (
                        provider.max_element_timestamp.isoformat()
                        if provider.max_element_timestamp is not None
                        else None
                    ),
                    "per_layer_max_timestamps": timestamps,
                    "osm_timestamp_verification": (
                        "all OSM elements verified edited on or before "
                        f"{snapshot_date.isoformat()}"
                    ),
                    "osm_skipped_relations": skipped_relations,
                    "overpass_request_count": provider.request_count,
                    "overpass_elapsed_seconds": time.monotonic() - started,
                }
            )
    else:
        features, skipped_relations = _extract_ohsome(provider, bbox, snapshot_date)

    output_path = output_dir / "osm_pre_event.gpkg"
    written_counts: dict[str, int] = {}

    def persist_layer(layer_name: str, count: int) -> None:
        """Persist progress and provenance after a GeoPackage layer is written."""
        written_counts[layer_name] = count
        if checkpoint is None:
            return
        timestamps = (
            list(dict.fromkeys(provider.osm_base_timestamps))
            if isinstance(provider, OverpassProvider)
            else []
        )
        update = {
                "osm_counts": dict(written_counts),
                "osm_skipped_relations": {
                    name: skipped_relations.get(name, 0)
                    for name in written_counts
                },
                "osm_base_timestamps": timestamps,
                "osm_base_timestamp": timestamps[0] if timestamps else None,
                "overpass_request_count": provider.request_count,
                "overpass_elapsed_seconds": time.monotonic() - started,
            }
        if isinstance(provider, OverpassProvider):
            update.update(
                {
                    "max_element_timestamp": (
                        provider.max_element_timestamp.isoformat()
                        if provider.max_element_timestamp is not None
                        else None
                    ),
                    "per_layer_max_timestamps": {
                        layer: timestamp.isoformat() if timestamp is not None else None
                        for layer, timestamp in provider.per_layer_max_timestamps.items()
                    },
                    "osm_timestamp_verification": (
                        "all OSM elements verified edited on or before "
                        f"{snapshot_date.isoformat()}"
                    ),
                }
            )
        checkpoint(update)

    counts = _write_osm_layers(
        features=features,
        snapshot_date=snapshot_date,
        grid=grid,
        output_path=output_path,
        provider=provider,
        manifest=manifest,
        settings=settings,
        checkpoint=persist_layer,
    )

    place_records = features["settlements"]
    wanted_names = {"syabrubesi", "syaphrubesi", "dhunche"}
    matches = [
        {
            "name": str(record.get("tags", {}).get("name", "")),
            "coordinates": [record["geometry"].x, record["geometry"].y]
            if record["geometry"].geom_type == "Point"
            else list(record["geometry"].centroid.coords[0]),
        }
        for record in place_records
        if re.sub(r"[^a-z0-9]", "", str(record.get("tags", {}).get("name", "")).casefold())
        in wanted_names
    ]
    elapsed_seconds = time.monotonic() - started
    request_count = provider.request_count
    osm_base_timestamps = (
        list(dict.fromkeys(provider.osm_base_timestamps))
        if isinstance(provider, OverpassProvider)
        else []
    )
    return {
        "path": output_path,
        "snapshot_date": snapshot_date,
        "latest_snapshot": latest_available,
        "api_version": provider.provider_name,
        "provider": provider.provider_name,
        "endpoint": provider.endpoint,
        "osm_base_timestamp": osm_base_timestamps[0] if osm_base_timestamps else None,
        "osm_base_timestamps": osm_base_timestamps,
        "max_element_timestamp": (
            provider.max_element_timestamp
            if isinstance(provider, OverpassProvider)
            else None
        ),
        "per_layer_max_timestamps": (
            provider.per_layer_max_timestamps
            if isinstance(provider, OverpassProvider)
            else {}
        ),
        "osm_timestamp_verification": (
            "all OSM elements verified edited on or before "
            f"{snapshot_date.isoformat()}"
            if isinstance(provider, OverpassProvider)
            else None
        ),
        "request_count": request_count,
        "elapsed_seconds": elapsed_seconds,
        "counts": counts,
        "skipped_relations": skipped_relations,
        "named_settlements": matches,
        "element_geometry_policy": "keep whole elements; deduplicate by OSM type and ID",
    }


def select_ohsome_snapshot(
    event_date: date,
    latest_available: datetime,
    requested_snapshot: date | None = None,
) -> date:
    """Choose a valid ohsome snapshot no later than its latest available data."""
    latest_date = latest_available.date()
    if requested_snapshot is not None:
        assert_osm_snapshot_valid(requested_snapshot, event_date)
        if requested_snapshot > latest_date:
            raise ValueError(
                f"Requested OSM snapshot {requested_snapshot} is later than latest available "
                f"snapshot {latest_date}."
            )
        return requested_snapshot

    selected = min(date.fromordinal(event_date.toordinal() - 1), latest_date)
    assert_osm_snapshot_valid(selected, event_date)
    return selected


def _write_ohsome_compatibility_request(
    bbox: tuple[float, float, float, float],
    snapshot_date: date,
    filter_text: str,
    base_url: str,
) -> tuple[str, str, dict[str, Any]]:
    """Build a documented ohsome v2 extraction request for legacy tests."""
    return (
        "POST",
        f"{base_url.rstrip('/')}/extraction/features",
        {"aoi": list(bbox), "time": snapshot_date.isoformat(), "filter": filter_text},
    )


def select_osm_snapshot(
    event_date: date,
    latest_available: datetime,
    requested_snapshot: date | None = None,
) -> date:
    """Keep the prior public snapshot-selection helper name."""
    return select_ohsome_snapshot(event_date, latest_available, requested_snapshot)


def detect_ohsome_api_version(base_url: str) -> str:
    """Resolve the configured ohsome API version from its documented base URL."""
    endpoint = base_url.rstrip("/").lower()
    if endpoint.endswith("/v1"):
        return "v1"
    if endpoint.endswith(("/v2", "/v2-rc")):
        return "v2"
    raise OhsomeError(
        "OHSOME_URL must end in /v1, /v2, or /v2-rc to select the ohsome API version."
    )


def build_ohsome_request(
    *,
    api_version: str,
    bbox: tuple[float, float, float, float],
    snapshot_date: date,
    filter_text: str,
    base_url: str,
) -> tuple[str, str, dict[str, Any]]:
    """Retain the previous ohsome request-builder API for v1/v2 unit callers."""
    if api_version == "v1":
        endpoint = f"{base_url.rstrip('/')}/elements/geometry"
        params: dict[str, Any] = {
            "bboxes": ",".join(f"{coordinate:.8f}" for coordinate in bbox),
            "time": snapshot_date.isoformat(),
            "filter": filter_text,
            "properties": "tags",
        }
        return "GET", endpoint, params
    if api_version == "v2":
        return _write_ohsome_compatibility_request(bbox, snapshot_date, filter_text, base_url)
    raise OhsomeError(f"Unsupported ohsome API version: {api_version}")


def _compatibility_tile_bboxes(
    bbox: tuple[float, float, float, float],
    max_tile_degrees: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    """Legacy helper for callers needing the former coarse ohsome tiling default."""
    return tile_bboxes(bbox, max_tile_degrees=max_tile_degrees)
