"""Sentinel-1 acquisition search, same-track pairing, and Process API ingestion."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.shutil import copy as raster_copy
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union

from sahayak.config import Settings
from sahayak.ingest.cache import CacheMissError, RequestCache
from sahayak.ingest.cdse_auth import (
    HTTP_MAX_ATTEMPTS,
    HTTP_TIMEOUT,
    HTTP_USER_AGENT,
    AuthenticationError,
    CdseTokenClient,
    RequestFailure,
    post_with_retries,
    scrub_sensitive_text,
)
from sahayak.ingest.grid import GridSpec
from sahayak.provenance import (
    RuleViolation,
    RunManifest,
    SceneTrack,
    assert_same_orbit_track,
)

LOGGER = logging.getLogger(__name__)

CATALOG_URL = "https://sh.dataspace.copernicus.eu/catalog/v1/search"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/process/v1"
CDSE_STAC_SEARCH_URL = "https://stac.dataspace.copernicus.eu/v1/search"
CATALOG_COLLECTION = "sentinel-1-grd"
S1_SOURCE = "Copernicus Data Space Sentinel-1 GRD via Sentinel Hub Process API"
S1_LICENSE = "Copernicus free, full and open data policy"
S1_ATTRIBUTION = "Contains modified Copernicus Sentinel data 2026."
DATA_TAKE_GROUP_SECONDS = 90
PROCESS_MAX_DIMENSION = 2500
PROCESS_TILE_SIZE = PROCESS_MAX_DIMENSION
NODATA = -9999.0
BAND_NAMES = ("VV", "VH", "localIncidenceAngle", "shadowMask", "dataMask")
BAND_DESCRIPTIONS = (
    "VV gamma0 terrain backscatter (linear power)",
    "VH gamma0 terrain backscatter (linear power)",
    "Local incidence angle (degrees)",
    "Shadow mask",
    "Data mask",
)
PROCESSING_EVALSCRIPT = """//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["VV", "VH", "localIncidenceAngle", "shadowMask", "dataMask"],
      units: ["LINEAR_POWER", "LINEAR_POWER", "DN", "DN", "DN"]
    }],
    output: { bands: 5, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [
    sample.VV,
    sample.VH,
    sample.localIncidenceAngle,
    sample.shadowMask,
    sample.dataMask
  ];
}
"""


class Sentinel1Error(RuntimeError):
    """Raised when Sentinel-1 search, selection, or processing fails."""


class RelativeOrbitMetadataError(Sentinel1Error):
    """Raised when Catalog results do not provide explicit relative-orbit metadata."""


@dataclass(frozen=True)
class Acquisition:
    """A grouped Sentinel-1 data-take with its track and footprint metadata."""

    platform: str
    acquired_at: datetime
    relative_orbit: int
    direction: Literal["ASCENDING", "DESCENDING"]
    polarizations: tuple[str, ...]
    mode: str
    footprint: BaseGeometry
    coverage_fraction: float
    item_ids: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    orbit_metadata_source: str = "sentinel-hub-catalog"

    @property
    def track(self) -> SceneTrack:
        """Return the track metadata in the shared provenance model."""
        return SceneTrack(
            relative_orbit=self.relative_orbit,
            pass_direction=self.direction,
        )

    @property
    def has_dual_polarization(self) -> bool:
        """Return whether the acquisition contains both VV and VH."""
        return {"VV", "VH"}.issubset(set(self.polarizations))


@dataclass(frozen=True)
class AcquisitionPair:
    """A valid pre-event and post-event pair with its ranking metadata."""

    pre: Acquisition
    post: Acquisition
    gap_days: float
    same_platform: bool
    repeat_interval_preferred: bool


@dataclass(frozen=True)
class ProcessTile:
    """A row/column window that exactly partitions the common grid."""

    row_off: int
    col_off: int
    height: int
    width: int


def build_catalog_request(
    bbox: tuple[float, float, float, float],
    event_date: date,
    *,
    page_token: int | None = None,
) -> dict[str, object]:
    """Build a Catalog search spanning 60 days before through 45 days after an event."""
    start = datetime.combine(event_date - timedelta(days=60), datetime.min.time(), UTC)
    end = datetime.combine(event_date + timedelta(days=45), datetime.max.time(), UTC)
    body: dict[str, object] = {
        "bbox": list(bbox),
        "datetime": f"{_format_utc(start)}/{_format_utc(end)}",
        "collections": [CATALOG_COLLECTION],
        "limit": 100,
    }
    if page_token is not None:
        body["next"] = page_token
    return body


def build_catalog_probe_request() -> dict[str, object]:
    """Build the requested minimal three-day Sentinel-1 Catalog diagnostic search."""
    return {
        "bbox": [85.30, 28.10, 85.42, 28.20],
        "datetime": "2026-08-23T23:59:59Z/2026-08-26T23:59:59Z",
        "collections": [CATALOG_COLLECTION],
        "limit": 5,
    }


def _format_utc(value: datetime) -> str:
    """Format a timezone-aware datetime in the ISO-8601 UTC form used by the APIs."""
    if value.tzinfo is None:
        raise ValueError("Sentinel-1 API times must be timezone-aware.")
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise Sentinel1Error(f"Catalog returned a timestamp without a timezone: {value!r}.")
    return parsed.astimezone(UTC)


def _relative_orbit(properties: dict[str, Any], item_id: str) -> int:
    """Read an explicit, positive STAC relative orbit; never infer a track."""
    value = properties.get("sat:relative_orbit")
    if isinstance(value, int) and not isinstance(value, bool):
        relative_orbit = value
    elif isinstance(value, float) and value.is_integer():
        relative_orbit = int(value)
    else:
        relative_orbit = 0
    if relative_orbit >= 1:
        return relative_orbit
    raise RelativeOrbitMetadataError(
        "Catalog item "
        f"{item_id!r} has missing or invalid sat:relative_orbit metadata "
        f"({value!r}); relative orbit must be at least 1."
    )


def _polarizations(value: object) -> tuple[str, ...]:
    """Normalize Catalog polarization metadata to uppercase channel names."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(part, str) for part in value):
        values = value
    else:
        return ()
    code_map = {"DV": ("VV", "VH"), "DH": ("HH", "HV"), "SV": ("VV",), "SH": ("HH",)}
    normalized: list[str] = []
    for part in values:
        channels = code_map.get(part.upper(), (part.upper(),))
        normalized.extend(channels)
    return tuple(sorted(set(normalized)))


OrbitResolver = Callable[[str, datetime, dict[str, Any]], tuple[int, str]]


def _item_to_slice(
    item: dict[str, Any],
    orbit_resolver: OrbitResolver | None = None,
) -> dict[str, Any]:
    """Validate and normalize one Catalog item without inventing missing metadata."""
    properties = item.get("properties")
    item_id = str(item.get("id", "<unknown>"))
    if not isinstance(properties, dict):
        raise Sentinel1Error(f"Catalog item {item_id!r} has no properties object.")
    required = ("datetime", "platform", "acquisitionMode")
    missing = [field for field in required if properties.get(field) is None]
    polarization = properties.get("s1:polarization")
    if polarization is None:
        polarization = properties.get("polarization")
    if polarization is None:
        missing.append("s1:polarization")
    if missing:
        raise Sentinel1Error(
            f"Catalog item {item_id!r} lacks required properties: {', '.join(missing)}."
        )
    geometry_data = item.get("geometry")
    if not isinstance(geometry_data, dict):
        raise Sentinel1Error(f"Catalog item {item_id!r} has no usable footprint geometry.")
    try:
        footprint = shape(geometry_data)
    except (TypeError, ValueError) as error:
        raise Sentinel1Error(f"Catalog item {item_id!r} has invalid footprint geometry.") from error
    if footprint.is_empty or not footprint.is_valid:
        raise Sentinel1Error(f"Catalog item {item_id!r} has an empty or invalid footprint.")

    acquired_at = _parse_datetime(str(properties["datetime"]))
    platform = str(properties["platform"])
    metadata_source = "sentinel-hub-catalog"
    try:
        relative_orbit = _relative_orbit(properties, item_id)
    except RelativeOrbitMetadataError:
        if orbit_resolver is None:
            raise
        relative_orbit, direction = orbit_resolver(platform, acquired_at, geometry_data)
        metadata_source = "cdse-stac"
    else:
        direction = str(properties.get("orbitDirection", "")).upper()
        if direction not in {"ASCENDING", "DESCENDING"}:
            raise Sentinel1Error(
                f"Catalog item {item_id!r} has unsupported orbit direction {direction!r}."
            )
    return {
        "id": item_id,
        "datetime": acquired_at,
        "platform": platform,
        "relative_orbit": relative_orbit,
        "direction": direction,
        "orbit_metadata_source": metadata_source,
        "polarizations": _polarizations(polarization),
        "mode": str(properties["acquisitionMode"]).upper(),
        "footprint": footprint,
    }


def _coverage_fraction(
    footprint: BaseGeometry,
    bbox: tuple[float, float, float, float],
    grid: GridSpec,
) -> float:
    """Measure a Catalog footprint's coverage of the requested bbox in the grid CRS."""
    min_lon, min_lat, max_lon, max_lat = bbox
    edge_points = 20
    ring = (
        [(min_lon + (max_lon - min_lon) * i / edge_points, min_lat) for i in range(edge_points)]
        + [
            (max_lon, min_lat + (max_lat - min_lat) * i / edge_points)
            for i in range(edge_points)
        ]
        + [
            (max_lon - (max_lon - min_lon) * i / edge_points, max_lat)
            for i in range(edge_points)
        ]
        + [
            (min_lon, max_lat - (max_lat - min_lat) * i / edge_points)
            for i in range(edge_points)
        ]
    )
    transformer = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True)
    target = transform_geometry(transformer.transform, Polygon(ring))
    projected_footprint = transform_geometry(transformer.transform, footprint)
    if target.area <= 0:
        raise Sentinel1Error("The transformed Sentinel-1 request bbox has no area.")
    covered = projected_footprint.intersection(target).area
    return min(1.0, max(0.0, covered / target.area))


def group_slices(
    items: Iterable[dict[str, Any]],
    bbox: tuple[float, float, float, float],
    grid: GridSpec,
    *,
    group_seconds: int = DATA_TAKE_GROUP_SECONDS,
    orbit_resolver: OrbitResolver | None = None,
    exclusions: list[dict[str, str]] | None = None,
) -> list[Acquisition]:
    """Group slices on the same platform and track when acquisition times are close."""
    slices: list[dict[str, Any]] = []
    rejected = exclusions if exclusions is not None else []
    for item in items:
        try:
            slices.append(_item_to_slice(item, orbit_resolver))
        except RelativeOrbitMetadataError as error:
            properties = item.get("properties")
            properties = properties if isinstance(properties, dict) else {}
            rejected.append(
                {
                    "item_id": str(item.get("id", "<unknown>")),
                    "platform": str(properties.get("platform", "<unknown>")),
                    "datetime": str(properties.get("datetime", "<unknown>")),
                    "reason": str(error),
                }
            )
    slices.sort(
        key=lambda item: (
            item["platform"],
            item["relative_orbit"],
            item["direction"],
            item["datetime"],
        )
    )
    groups: list[list[dict[str, Any]]] = []
    for item in slices:
        if not groups:
            groups.append([item])
            continue
        previous = groups[-1][-1]
        same_track = (
            item["platform"] == previous["platform"]
            and item["relative_orbit"] == previous["relative_orbit"]
            and item["direction"] == previous["direction"]
        )
        within_take = (item["datetime"] - previous["datetime"]).total_seconds() <= group_seconds
        if same_track and within_take:
            groups[-1].append(item)
        else:
            groups.append([item])

    acquisitions: list[Acquisition] = []
    for group in groups:
        group_times = [item["datetime"] for item in group]
        footprints = unary_union([item["footprint"] for item in group])
        coverage = _coverage_fraction(footprints, bbox, grid)
        acquisitions.append(
            Acquisition(
                platform=group[0]["platform"],
                acquired_at=min(group_times),
                relative_orbit=group[0]["relative_orbit"],
                direction=group[0]["direction"],
                polarizations=tuple(
                    sorted({pol for item in group for pol in item["polarizations"]})
                ),
                mode=group[0]["mode"],
                footprint=footprints,
                coverage_fraction=coverage,
                item_ids=tuple(item["id"] for item in group),
                start_time=min(group_times),
                end_time=max(group_times),
                orbit_metadata_source=(
                    "cdse-stac"
                    if any(item["orbit_metadata_source"] == "cdse-stac" for item in group)
                    else "sentinel-hub-catalog"
                ),
            )
        )
    return acquisitions


def _repeat_interval_preferred(gap_days: float) -> bool:
    """Return whether a repeat gap is close to an integer multiple of six days."""
    rounded_days = round(gap_days)
    return abs(gap_days - rounded_days) <= 0.25 and rounded_days > 0 and rounded_days % 6 == 0


def _pair_sort_key(pair: AcquisitionPair, event_date: date) -> tuple[int, int, int, int]:
    """Return the documented lexicographic pair-selection ranking."""
    post_distance = (pair.post.acquired_at.date() - event_date).days
    pre_distance = (event_date - pair.pre.acquired_at.date()).days
    return (
        post_distance,
        pre_distance,
        0 if pair.repeat_interval_preferred else 1,
        0 if pair.same_platform else 1,
    )


def _group_counts(acquisitions: Iterable[Acquisition], event_date: date) -> str:
    """Summarize before and after acquisition counts for every track group."""
    counts: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    for acquisition in acquisitions:
        index = 0 if acquisition.acquired_at.date() < event_date else 1
        counts[(acquisition.relative_orbit, acquisition.direction)][index] += 1
    if not counts:
        return "no track groups found"
    return "; ".join(
        f"relative orbit {orbit} {direction}: {values[0]} before, {values[1]} on/after"
        for (orbit, direction), values in sorted(counts.items())
    )


def _track_inventory(
    acquisitions: Iterable[Acquisition],
    event_date: date,
) -> list[dict[str, object]]:
    """Count valid acquisitions before and on/after the event for every track."""
    counts: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0])
    for acquisition in acquisitions:
        if acquisition.relative_orbit < 1:
            continue
        index = 0 if acquisition.acquired_at.date() < event_date else 1
        counts[(acquisition.relative_orbit, acquisition.direction)][index] += 1
    return [
        {
            "relative_orbit": orbit,
            "direction": direction,
            "before_event": values[0],
            "on_or_after_event": values[1],
        }
        for (orbit, direction), values in sorted(counts.items())
    ]


def select_pairs(
    acquisitions: Iterable[Acquisition],
    event_date: date,
    *,
    minimum_coverage: float = 0.95,
) -> tuple[AcquisitionPair, list[AcquisitionPair], str]:
    """Choose the best strictly pre/post same-track pair and rank alternatives."""
    all_acquisitions = list(acquisitions)
    if not 0 < minimum_coverage <= 1:
        raise ValueError("Minimum coverage must be greater than zero and at most one.")
    eligible = [
        item
        for item in all_acquisitions
        if item.relative_orbit >= 1
        and item.coverage_fraction >= minimum_coverage
        and item.mode == "IW"
        and item.has_dual_polarization
    ]
    before = [item for item in eligible if item.acquired_at.date() < event_date]
    after = [item for item in eligible if item.acquired_at.date() >= event_date]
    pairs: list[AcquisitionPair] = []
    for pre in before:
        for post in after:
            try:
                assert_same_orbit_track(pre.track, post.track)
            except RuleViolation:
                continue
            gap_days = (post.acquired_at - pre.acquired_at).total_seconds() / 86400
            pairs.append(
                AcquisitionPair(
                    pre=pre,
                    post=post,
                    gap_days=gap_days,
                    same_platform=pre.platform == post.platform,
                    repeat_interval_preferred=_repeat_interval_preferred(gap_days),
                )
            )
    ranked = sorted(pairs, key=lambda pair: _pair_sort_key(pair, event_date))
    if not ranked:
        raise Sentinel1Error(
            "No valid Sentinel-1 pair has >=95% bbox coverage, IW dual polarization, "
            "and a matching relative orbit and direction. Track counts: "
            f"{_group_counts(all_acquisitions, event_date)}."
        )
    chosen = ranked[0]
    reason = (
        "Selected the pair with the closest post-event acquisition, then the closest "
        "pre-event acquisition; the ranking next prefers repeat intervals divisible "
        "by six days, then matching platform. Both acquisitions meet the 95% coverage "
        "threshold and share relative orbit and pass direction. "
        + (
            "The platforms match."
            if chosen.same_platform
            else "The platforms differ, but the relative orbit and pass direction match."
        )
    )
    return chosen, ranked[1:], reason


def select_baseline(
    chosen: AcquisitionPair,
    acquisitions: Iterable[Acquisition],
    *,
    count: int = 3,
    minimum_spacing_days: int = 6,
) -> list[Acquisition]:
    """Select up to count pre-event scenes on the chosen track, spaced by six days."""
    if not 1 <= count <= 3:
        raise ValueError("S1 baseline count must be between 1 and 3, including the chosen scene.")
    candidates = [
        item
        for item in acquisitions
        if item.acquired_at < chosen.pre.acquired_at
        and item.relative_orbit >= 1
        and item.relative_orbit == chosen.pre.relative_orbit
        and item.direction == chosen.pre.direction
        and item.coverage_fraction >= 0.95
        and item.mode == "IW"
        and item.has_dual_polarization
    ]
    candidates.sort(
        key=lambda item: (
            abs((chosen.pre.acquired_at - item.acquired_at).total_seconds()),
            item.platform != chosen.pre.platform,
        )
    )
    selected = [chosen.pre]
    for candidate in candidates:
        if all(
            abs((candidate.acquired_at - existing.acquired_at).total_seconds())
            >= minimum_spacing_days * 86400
            for existing in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                break
    return selected


def process_tile_windows(
    width: int,
    height: int,
    *,
    max_dimension: int = PROCESS_TILE_SIZE,
) -> list[ProcessTile]:
    """Partition a raster into non-overlapping windows within the chosen request size."""
    if width <= 0 or height <= 0 or max_dimension <= 0:
        raise ValueError("Grid dimensions and maximum process dimension must be positive.")
    return [
        ProcessTile(row, col, min(max_dimension, height - row), min(max_dimension, width - col))
        for row in range(0, height, max_dimension)
        for col in range(0, width, max_dimension)
    ]


def mosaic_process_tiles(
    width: int,
    height: int,
    tiles: Iterable[tuple[ProcessTile, np.ndarray]],
    *,
    band_count: int = len(BAND_NAMES),
) -> np.ndarray:
    """Mosaic tile arrays into exact grid offsets, rejecting gaps and overlaps."""
    mosaic = np.full((band_count, height, width), np.nan, dtype=np.float32)
    occupied = np.zeros((height, width), dtype=bool)
    for tile, data in tiles:
        expected = (band_count, tile.height, tile.width)
        if data.shape != expected:
            raise Sentinel1Error(f"Process tile has shape {data.shape}; expected {expected}.")
        rows = slice(tile.row_off, tile.row_off + tile.height)
        cols = slice(tile.col_off, tile.col_off + tile.width)
        if occupied[rows, cols].any():
            raise Sentinel1Error("Process tiles overlap; refusing an ambiguous mosaic.")
        mosaic[:, rows, cols] = data
        occupied[rows, cols] = True
    if not occupied.all():
        raise Sentinel1Error("Process tiles do not cover the complete common grid.")
    return mosaic


def build_process_request(
    acquisition: Acquisition,
    grid: GridSpec,
    tile: ProcessTile,
) -> dict[str, object]:
    """Build a Process API request for one exact common-grid window."""
    left = grid.left + tile.col_off * grid.resolution
    top = grid.top - tile.row_off * grid.resolution
    right = left + tile.width * grid.resolution
    bottom = top - tile.height * grid.resolution
    from_time = acquisition.start_time - timedelta(seconds=1)
    to_time = acquisition.end_time + timedelta(seconds=1)
    epsg_code = f"http://www.opengis.net/def/crs/EPSG/0/{grid.epsg}"
    return {
        "input": {
            "bounds": {
                "bbox": [left, bottom, right, top],
                "properties": {"crs": epsg_code},
            },
            "data": [
                {
                    "type": "sentinel-1-grd",
                    "dataFilter": {
                        "timeRange": {
                            "from": _format_utc(from_time),
                            "to": _format_utc(to_time),
                        },
                        "resolution": "HIGH",
                        "acquisitionMode": "IW",
                        "polarization": "DV",
                        "orbitDirection": acquisition.direction,
                    },
                    "processing": {
                        "backCoeff": "GAMMA0_TERRAIN",
                        "orthorectify": "true",
                        "demInstance": "COPERNICUS_30",
                        "speckleFilter": {"type": "NONE"},
                    },
                }
            ],
        },
        "output": {
            "width": tile.width,
            "height": tile.height,
            "responses": [
                {"identifier": "default", "format": {"type": "image/tiff"}}
            ],
        },
        "evalscript": PROCESSING_EVALSCRIPT,
    }


class SentinelHubClient:
    """Authenticated, cached Catalog and Process API client."""

    def __init__(
        self,
        settings: Settings,
        cache: RequestCache,
        *,
        session: requests.Session | Any | None = None,
        token_client: CdseTokenClient | None = None,
        offline: bool = False,
    ) -> None:
        """Initialize a client with a shared session, token cache, and response cache."""
        self.settings = settings
        self.cache = cache
        self.session = session or requests.Session()
        self.token_client = token_client or CdseTokenClient(settings, session=self.session)
        self.offline = offline
        self.processing_units: list[float] = []
        self.catalog_request_count = 0
        self.process_request_count = 0

    def catalog_search(self, body: dict[str, object]) -> dict[str, Any]:
        """Fetch one cached Catalog page with one retry after an unauthorized response."""
        request = {"service": "catalog", "endpoint": CATALOG_URL, "body": body}
        payload = self.cache.get_or_fetch(
            request,
            lambda: self._authorized_post(CATALOG_URL, body, "catalog"),
            suffix=".json",
            offline=self.offline,
        )
        self.catalog_request_count += 1
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as error:
            raise Sentinel1Error("Catalog returned invalid JSON.") from error
        if not isinstance(result, dict):
            raise Sentinel1Error("Catalog returned a non-object response.")
        return result

    def cdse_stac_search(self, body: dict[str, object]) -> dict[str, Any]:
        """Fetch one cached search page from the main Copernicus Data Space STAC."""
        request = {"service": "cdse-stac", "endpoint": CDSE_STAC_SEARCH_URL, "body": body}

        def fetch() -> bytes:
            try:
                response, attempt = post_with_retries(
                    self.session,
                    CDSE_STAC_SEARCH_URL,
                    service_name="CDSE STAC search",
                    headers={"Content-Type": "application/json"},
                    json=body,
                    timeout=HTTP_TIMEOUT,
                )
            except RequestFailure as error:
                raise Sentinel1Error(str(error)) from error
            if response.status_code != 200:
                raise Sentinel1Error(
                    f"CDSE STAC search failed (HTTP {response.status_code}) "
                    f"on attempt {attempt}/{HTTP_MAX_ATTEMPTS}."
                )
            return response.content

        try:
            payload = self.cache.get_or_fetch(
                request,
                fetch,
                suffix=".json",
                offline=self.offline,
            )
        except CacheMissError as error:
            raise RelativeOrbitMetadataError(
                "No cached CDSE STAC item is available for this acquisition in offline mode."
            ) from error
        try:
            result = json.loads(payload)
        except json.JSONDecodeError as error:
            raise Sentinel1Error("CDSE STAC returned invalid JSON.") from error
        if not isinstance(result, dict):
            raise Sentinel1Error("CDSE STAC returned a non-object response.")
        return result

    def resolve_relative_orbit(
        self,
        platform: str,
        acquired_at: datetime,
        geometry: dict[str, Any],
    ) -> tuple[int, str]:
        """Resolve missing Catalog orbit data from a matching main-CDSE-STAC item."""
        start = acquired_at - timedelta(seconds=DATA_TAKE_GROUP_SECONDS)
        end = acquired_at + timedelta(seconds=DATA_TAKE_GROUP_SECONDS)
        result = self.cdse_stac_search(
            {
                "collections": [CATALOG_COLLECTION],
                "datetime": f"{_format_utc(start)}/{_format_utc(end)}",
                "intersects": geometry,
                "limit": 100,
            }
        )
        features = result.get("features")
        if not isinstance(features, list) or not all(
            isinstance(feature, dict) for feature in features
        ):
            raise Sentinel1Error("CDSE STAC search response has no valid features list.")

        matches: list[tuple[float, int, str]] = []
        for feature in features:
            properties = feature.get("properties")
            if not isinstance(properties, dict):
                continue
            if str(properties.get("platform", "")).casefold() != platform.casefold():
                continue
            timestamp = properties.get("datetime")
            if not isinstance(timestamp, str):
                continue
            stac_time = _parse_datetime(timestamp)
            delta_seconds = abs((stac_time - acquired_at).total_seconds())
            if delta_seconds > DATA_TAKE_GROUP_SECONDS:
                continue
            try:
                relative_orbit = _relative_orbit(
                    properties, str(feature.get("id", "<unknown>"))
                )
            except RelativeOrbitMetadataError:
                continue
            direction = str(properties.get("sat:orbit_state", "")).upper()
            if direction not in {"ASCENDING", "DESCENDING"}:
                continue
            matches.append((delta_seconds, relative_orbit, direction))

        if not matches:
            raise RelativeOrbitMetadataError(
                f"CDSE STAC has no item matching platform {platform!r} and acquisition time "
                f"{_format_utc(acquired_at)} with valid sat:relative_orbit and sat:orbit_state."
            )
        nearest_delta = min(match[0] for match in matches)
        nearest = {
            (orbit, direction)
            for delta, orbit, direction in matches
            if delta == nearest_delta
        }
        if len(nearest) != 1:
            raise RelativeOrbitMetadataError(
                f"CDSE STAC returned ambiguous orbit metadata for platform {platform!r} "
                f"at {_format_utc(acquired_at)}."
            )
        return next(iter(nearest))

    def process(self, body: dict[str, object]) -> bytes:
        """Fetch one cached Process API TIFF with one retry after an unauthorized response."""
        request = {"service": "process", "endpoint": PROCESS_URL, "body": body}
        payload = self.cache.get_or_fetch(
            request,
            lambda: self._authorized_post(PROCESS_URL, body, "process"),
            suffix=".tif",
            offline=self.offline,
        )
        self.process_request_count += 1
        return payload

    def _authorized_post(
        self,
        endpoint: str,
        body: dict[str, object],
        service_name: str,
    ) -> bytes:
        """POST JSON with bearer authentication, refreshing once after HTTP 401."""
        for attempt in range(2):
            try:
                token = self.token_client.get_token(force_refresh=attempt > 0)
                secrets = [token]
                for secret in (
                    self.settings.cdse_client_id,
                    self.settings.cdse_client_secret,
                    self.settings.cdse_password,
                    self.settings.ohsome_api_key,
                ):
                    if secret is not None:
                        secrets.append(secret.get_secret_value())
                response, request_attempt = post_with_retries(
                    self.session,
                    endpoint,
                    service_name=f"Sentinel Hub {service_name}",
                    json=body,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "User-Agent": HTTP_USER_AGENT,
                    },
                    timeout=HTTP_TIMEOUT,
                    secrets=secrets,
                )
            except AuthenticationError:
                raise
            except RequestFailure as error:
                safe_message = scrub_sensitive_text(
                    str(error), secrets if "secrets" in locals() else ()
                )
                raise Sentinel1Error(safe_message) from error
            if response.status_code == 401 and attempt == 0:
                self.token_client.invalidate()
                continue
            if response.status_code != 200:
                raise Sentinel1Error(
                    f"Sentinel Hub {service_name} request failed "
                    f"(HTTP {response.status_code}) on request attempt "
                    f"{request_attempt}/{HTTP_MAX_ATTEMPTS}, token attempt {attempt + 1}/2."
                )
            if service_name == "process":
                pu_value = response.headers.get("x-processingunits-spent")
                if pu_value is not None:
                    try:
                        self.processing_units.append(float(pu_value))
                    except ValueError:
                        LOGGER.warning("Ignoring a non-numeric processing-unit response header.")
            return response.content
        raise Sentinel1Error(f"Sentinel Hub {service_name} request remained unauthorized.")


def catalog_acquisitions(
    *,
    bbox: tuple[float, float, float, float],
    event_date: date,
    grid: GridSpec,
    client: SentinelHubClient,
    exclusions: list[dict[str, str]] | None = None,
) -> list[Acquisition]:
    """Search every Catalog page and group the returned slices into data-takes."""
    items: list[dict[str, Any]] = []
    page_token: int | None = None
    while True:
        page = client.catalog_search(build_catalog_request(bbox, event_date, page_token=page_token))
        features = page.get("features")
        if not isinstance(features, list):
            raise Sentinel1Error("Catalog response has no features list.")
        if not all(isinstance(item, dict) for item in features):
            raise Sentinel1Error("Catalog returned a malformed feature.")
        items.extend(features)
        context = page.get("context")
        next_token = context.get("next") if isinstance(context, dict) else None
        if not next_token:
            break
        if not isinstance(next_token, int) or isinstance(next_token, bool):
            raise Sentinel1Error("Catalog returned an invalid pagination token.")
        if next_token == page_token:
            raise Sentinel1Error("Catalog repeated its pagination token.")
        page_token = next_token
    return group_slices(
        items,
        bbox,
        grid,
        orbit_resolver=getattr(client, "resolve_relative_orbit", None),
        exclusions=exclusions,
    )


def _acquisition_record(acquisition: Acquisition, filename: str | None = None) -> dict[str, object]:
    """Serialize acquisition metadata for selection.json."""
    record: dict[str, object] = {
        "platform": acquisition.platform,
        "datetime": acquisition.acquired_at.isoformat(),
        "relative_orbit": acquisition.relative_orbit,
        "direction": acquisition.direction,
        "polarization": list(acquisition.polarizations),
        "acquisition_mode": acquisition.mode,
        "coverage_fraction": acquisition.coverage_fraction,
        "item_ids": list(acquisition.item_ids),
        "orbit_metadata_source": acquisition.orbit_metadata_source,
    }
    if filename is not None:
        record["file"] = filename
    return record


def _pair_record(pair: AcquisitionPair) -> dict[str, object]:
    """Serialize a candidate pair for the ranked alternatives."""
    return {
        "pre": _acquisition_record(pair.pre),
        "post": _acquisition_record(pair.post),
        "gap_days": pair.gap_days,
        "same_platform": pair.same_platform,
        "repeat_interval_preferred": pair.repeat_interval_preferred,
    }


def _load_process_tile(
    payload: bytes,
    tile: ProcessTile,
) -> tuple[ProcessTile, np.ndarray]:
    """Read a response TIFF from memory and validate its dimensions and band count."""
    try:
        with rasterio.MemoryFile(payload) as memory_file:
            with memory_file.open() as dataset:
                if dataset.count != len(BAND_NAMES):
                    raise Sentinel1Error(
                        f"Process response has {dataset.count} bands; expected {len(BAND_NAMES)}."
                    )
                if (dataset.height, dataset.width) != (tile.height, tile.width):
                    raise Sentinel1Error(
                        "Process response dimensions "
                        f"{dataset.width}x{dataset.height} do not match requested "
                        f"{tile.width}x{tile.height}."
                    )
                return tile, dataset.read(out_dtype="float32")
    except rasterio.errors.RasterioError as error:
        message = "Sentinel Hub Process API did not return a readable GeoTIFF."
        raise Sentinel1Error(message) from error


def write_s1_cog(
    path: Path,
    data: np.ndarray,
    grid: GridSpec,
) -> float:
    """Write a float32 five-band COG and return the percentage of data-valid pixels."""
    if data.shape != (len(BAND_NAMES), grid.height, grid.width):
        raise ValueError(
            f"Sentinel-1 mosaic shape {data.shape} does not match the shared grid."
        )
    valid = np.isfinite(data[4]) & (data[4] > 0)
    if not valid.any():
        raise Sentinel1Error(f"Sentinel-1 output {path.name} has no valid data pixels.")
    output = np.where(np.isfinite(data), data, NODATA).astype(np.float32, copy=False)
    output[:, ~valid] = NODATA
    for band_index in range(output.shape[0]):
        if np.all(output[band_index] == NODATA):
            raise Sentinel1Error(
                f"Sentinel-1 output {path.name} has an all-nodata {BAND_NAMES[band_index]} band."
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_", suffix=".tmp.tif", dir=path.parent, delete=False
    ) as temporary_file:
        temporary_source = Path(temporary_file.name)
    temporary_cog = temporary_source.with_name(f"{temporary_source.stem}.cog.tif")
    profile = grid.rasterio_profile(dtype="float32", nodata=NODATA, count=len(BAND_NAMES))
    profile.update(compress="deflate", predictor=3, tiled=True)
    try:
        with rasterio.open(temporary_source, "w", **profile) as dataset:
            dataset.write(output)
            for index, description in enumerate(BAND_DESCRIPTIONS, start=1):
                dataset.set_band_description(index, description)
        raster_copy(
            temporary_source,
            temporary_cog,
            driver="COG",
            compress="DEFLATE",
            predictor="FLOATING_POINT",
            nodata=NODATA,
        )
        with rasterio.open(temporary_cog) as dataset:
            if (dataset.count, dataset.height, dataset.width) != (
                len(BAND_NAMES),
                grid.height,
                grid.width,
            ):
                raise Sentinel1Error("Written Sentinel-1 COG does not match the shared grid.")
            if dataset.crs != rasterio.crs.CRS.from_string(grid.crs):
                raise Sentinel1Error("Written Sentinel-1 COG has the wrong CRS.")
            if dataset.transform != grid.transform:
                raise Sentinel1Error("Written Sentinel-1 COG has the wrong affine transform.")
            written = dataset.read(out_dtype="float32")
            if any(np.all(written[index] == NODATA) for index in range(len(BAND_NAMES))):
                raise Sentinel1Error("Written Sentinel-1 COG contains an all-nodata band.")
            written_valid = np.isfinite(written[4]) & (written[4] > 0)
            if not written_valid.any():
                raise Sentinel1Error("Written Sentinel-1 COG has no valid data pixels.")
        os.replace(temporary_cog, path)
    finally:
        temporary_source.unlink(missing_ok=True)
        temporary_cog.unlink(missing_ok=True)
    return float(written_valid.mean() * 100)


def _db_statistics(data: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute minimum, median, and maximum dB values over positive valid samples."""
    valid = np.isfinite(data[4]) & (data[4] > 0)
    result: dict[str, dict[str, float]] = {}
    for band_index, name in enumerate(("VV", "VH")):
        values = data[band_index][valid & (data[band_index] > 0)]
        if values.size == 0:
            raise Sentinel1Error(f"No positive valid {name} samples are available for dB stats.")
        decibels = 10 * np.log10(values.astype(np.float64))
        result[name] = {
            "minimum_db": float(np.min(decibels)),
            "median_db": float(np.median(decibels)),
            "maximum_db": float(np.max(decibels)),
        }
    return result


def _process_acquisition(
    *,
    acquisition: Acquisition,
    filename: str,
    grid: GridSpec,
    client: SentinelHubClient,
    output_dir: Path,
    settings: Settings,
    manifest: RunManifest,
) -> dict[str, object]:
    """Request, mosaic, validate, and register one acquisition."""
    windows = process_tile_windows(grid.width, grid.height)
    tile_data: list[tuple[ProcessTile, np.ndarray]] = []
    total = len(windows)
    for index, tile in enumerate(windows, start=1):
        request_body = build_process_request(acquisition, grid, tile)
        LOGGER.info(
            "Sentinel-1 %s tile %s of %s: %sx%s",
            filename,
            index,
            total,
            tile.width,
            tile.height,
        )
        payload = client.process(request_body)
        tile_data.append(_load_process_tile(payload, tile))
    data = mosaic_process_tiles(grid.width, grid.height, tile_data)
    output_path = output_dir / filename
    valid_percent = write_s1_cog(output_path, data, grid)
    manifest.add_input(
        name=filename,
        role="input",
        source=S1_SOURCE,
        license=S1_LICENSE,
        retrieved_utc=datetime.now(UTC),
        path=output_path,
        settings=settings,
    )
    result = {
        **_acquisition_record(acquisition, filename),
        "percent_valid_pixels": valid_percent,
        "statistics_db": _db_statistics(data),
    }
    return result


def ingest_s1(
    *,
    bbox: tuple[float, float, float, float],
    event_date: date,
    baseline_count: int,
    grid: GridSpec,
    settings: Settings,
    cache: RequestCache,
    offline: bool,
    output_dir: Path,
    manifest: RunManifest,
    checkpoint: Callable[[dict[str, object]], None] | None = None,
    session: requests.Session | Any | None = None,
) -> dict[str, object]:
    """Ingest a same-track Sentinel-1 pair and an optional temporal baseline stack."""
    client = SentinelHubClient(
        settings,
        cache,
        session=session,
        offline=offline,
    )
    excluded_acquisitions: list[dict[str, str]] = []
    acquisitions = catalog_acquisitions(
        bbox=bbox,
        event_date=event_date,
        grid=grid,
        client=client,
        exclusions=excluded_acquisitions,
    )
    try:
        chosen, alternatives, reason = select_pairs(acquisitions, event_date)
    except Sentinel1Error:
        if checkpoint is not None:
            checkpoint(
                {
                    "s1": {
                        "provider": "sentinel-hub-process-api",
                        "all_acquisitions": [
                            _acquisition_record(item) for item in acquisitions
                        ],
                        "track_groups": _track_inventory(acquisitions, event_date),
                        "excluded_acquisitions": excluded_acquisitions,
                        "absolute_orbit_cross_check": {
                            "performed": False,
                            "reason": (
                                "Skipped because no platform-specific relative/absolute orbit "
                                "formula was verified in the permitted official documentation."
                            ),
                        },
                    }
                }
            )
        raise
    baseline = select_baseline(chosen, acquisitions, count=baseline_count)
    baseline_files = ["s1_pre.tif", "s1_pre_2.tif", "s1_pre_3.tif"]
    files: dict[str, dict[str, object]] = {}
    selection: dict[str, object] = {
        "provider": "sentinel-hub-process-api",
        "catalog_endpoint": CATALOG_URL,
        "orbit_metadata_endpoint": CDSE_STAC_SEARCH_URL,
        "process_endpoint": PROCESS_URL,
        "search_window_days": {"before": 60, "after": 45},
        "pair": {
            "pre": _acquisition_record(chosen.pre, "s1_pre.tif"),
            "post": _acquisition_record(chosen.post, "s1_post.tif"),
            "gap_days": chosen.gap_days,
            "same_platform": chosen.same_platform,
        },
        "baseline": [],
        "ranked_alternatives": [_pair_record(pair) for pair in alternatives],
        "selection_reason": reason,
        "minimum_coverage_fraction": 0.95,
        "process_requests": 0,
        "estimated_processing_units": None,
        "attribution": S1_ATTRIBUTION,
        "all_acquisitions": [_acquisition_record(item) for item in acquisitions],
        "track_groups": _track_inventory(acquisitions, event_date),
        "excluded_acquisitions": excluded_acquisitions,
        "absolute_orbit_cross_check": {
            "performed": False,
            "reason": (
                "Skipped because no platform-specific relative/absolute orbit formula was "
                "verified in the permitted official documentation."
            ),
        },
    }
    if checkpoint is not None:
        checkpoint({"s1": selection})

    post_result = _process_acquisition(
        acquisition=chosen.post,
        filename="s1_post.tif",
        grid=grid,
        client=client,
        output_dir=output_dir,
        settings=settings,
        manifest=manifest,
    )
    files["s1_post.tif"] = post_result
    selection["process_requests"] = client.process_request_count
    selection["estimated_processing_units"] = (
        sum(client.processing_units) if client.processing_units else None
    )
    if checkpoint is not None:
        checkpoint({"s1": {**selection, "files": dict(files)}})

    processed_baselines: list[dict[str, object]] = []
    for acquisition, filename in zip(baseline, baseline_files, strict=False):
        result = _process_acquisition(
            acquisition=acquisition,
            filename=filename,
            grid=grid,
            client=client,
            output_dir=output_dir,
            settings=settings,
            manifest=manifest,
        )
        files[filename] = result
        processed_baselines.append(_acquisition_record(acquisition, filename))
        selection["baseline"] = list(processed_baselines)
        selection["process_requests"] = client.process_request_count
        selection["estimated_processing_units"] = (
            sum(client.processing_units) if client.processing_units else None
        )
        if checkpoint is not None:
            checkpoint({"s1": {**selection, "files": dict(files)}})

    selection["baseline"] = [
        _acquisition_record(item, baseline_files[index])
        for index, item in enumerate(baseline)
    ]
    selection["files"] = files
    selection["process_requests"] = client.process_request_count
    selection["catalog_requests"] = client.catalog_request_count
    selection["estimated_processing_units"] = (
        sum(client.processing_units) if client.processing_units else None
    )
    if checkpoint is not None:
        checkpoint({"s1": selection})
    return {
        "selection": selection,
        "files": files,
        "catalog_request_count": client.catalog_request_count,
        "process_request_count": client.process_request_count,
        "processing_units": sum(client.processing_units) if client.processing_units else None,
    }
