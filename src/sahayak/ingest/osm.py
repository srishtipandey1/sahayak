"""Pre-event OpenStreetMap extraction from configured ohsome API endpoints."""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import requests

from sahayak.config import Settings
from sahayak.ingest.cache import RequestCache
from sahayak.ingest.grid import GridSpec
from sahayak.provenance import RunManifest, assert_osm_snapshot_valid

OSM_ATTRIBUTION = "© OpenStreetMap contributors (ODbL)."
OSM_SOURCE = "OpenStreetMap via ohsome API"
OSM_LICENSE = "Open Database License (ODbL)"
MAX_TILE_DEGREES = 0.25
LAYER_FILTERS = {
    "buildings": "building=*",
    "roads": "highway=* and geometry:line",
    "bridges": "(bridge=* or man_made=bridge)",
    "settlements": "place in (village,hamlet,town,city,suburb,locality)",
    "health_facilities": (
        "(amenity in (hospital,clinic,doctors) or healthcare in (hospital,clinic))"
    ),
}


class OhsomeError(RuntimeError):
    """Raised when the configured ohsome service fails or returns unusable data."""


def detect_ohsome_api_version(base_url: str) -> str:
    """Determine the configured ohsome API generation from its documented URL path."""
    endpoint = base_url.rstrip("/").lower()
    if endpoint.endswith("/v1"):
        return "v1"
    if endpoint.endswith(("/v2", "/v2-rc")):
        return "v2"
    raise OhsomeError(
        "OHSOME_URL must end in /v1, /v2, or /v2-rc to select the ohsome API version."
    )


def tile_bboxes(
    bbox: tuple[float, float, float, float],
    max_tile_degrees: float = MAX_TILE_DEGREES,
) -> list[tuple[float, float, float, float]]:
    """Subdivide a geographic bounding box into bounded, non-overlapping requests."""
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


def build_ohsome_request(
    *,
    api_version: str,
    bbox: tuple[float, float, float, float],
    snapshot_date: date,
    filter_text: str,
    base_url: str,
) -> tuple[str, str, dict[str, Any]]:
    """Build the documented v1 GET or v2 POST extraction request."""
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
        endpoint = f"{base_url.rstrip('/')}/extraction/features"
        payload = {
            "aoi": list(bbox),
            "time": snapshot_date.isoformat(),
            "filter": filter_text,
        }
        return "POST", endpoint, payload
    raise OhsomeError(f"Unsupported ohsome API version: {api_version}")


def _latest_timestamp(metadata: dict[str, Any]) -> datetime:
    """Read the latest available OSM timestamp from ohsome metadata."""
    try:
        timestamp = metadata["extractRegion"]["temporalExtent"]["toTimestamp"]
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError) as error:
        raise OhsomeError(
            "ohsome metadata did not include extractRegion.temporalExtent.toTimestamp."
        ) from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def select_osm_snapshot(
    event_date: date,
    latest_available: datetime,
    requested_snapshot: date | None = None,
) -> date:
    """Select the requested snapshot or the earlier of event-minus-one and latest."""
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


class OhsomeClient:
    """Small cache-aware client for the configured ohsome API."""

    def __init__(
        self,
        settings: Settings,
        cache: RequestCache,
        *,
        offline: bool = False,
        timeout: float = 240.0,
    ) -> None:
        """Initialize with settings and the selected online/offline cache mode."""
        self.settings = settings
        self.cache = cache
        self.offline = offline
        self.timeout = timeout
        self.base_url = settings.ohsome_url.rstrip("/")
        self.api_version = detect_ohsome_api_version(self.base_url)
        self.retrieved_times: list[datetime] = []

    def _headers(self) -> dict[str, str]:
        """Build optional v2 authorization headers without exposing credentials."""
        if self.api_version != "v2" or self.settings.ohsome_api_key is None:
            return {}
        return {"Authorization": self.settings.ohsome_api_key.get_secret_value()}

    def _get_bytes(
        self,
        *,
        method: str,
        endpoint: str,
        parameters: dict[str, Any],
        suffix: str,
    ) -> bytes:
        """Retrieve one API payload through the stable request cache."""
        request_identity = {
            "service": "ohsome",
            "method": method,
            "endpoint": endpoint,
            "parameters": parameters,
        }

        def fetch() -> bytes:
            try:
                if method == "GET":
                    response = requests.get(
                        endpoint,
                        params=parameters,
                        headers=self._headers(),
                        timeout=self.timeout,
                    )
                else:
                    response = requests.post(
                        endpoint,
                        json=parameters,
                        headers=self._headers(),
                        timeout=self.timeout,
                    )
                response.raise_for_status()
            except requests.RequestException as error:
                raise OhsomeError(
                    f"Configured ohsome {self.api_version} endpoint failed "
                    f"({endpoint}): {error}"
                ) from error
            return response.content

        try:
            payload = self.cache.get_or_fetch(
                request_identity,
                fetch,
                suffix=suffix,
                offline=self.offline,
            )
            self.retrieved_times.append(self.cache.cached_at(request_identity, suffix))
            return payload
        except OhsomeError:
            raise
        except OSError as error:
            message = f"Could not use ohsome {self.api_version} request cache: {error}"
            raise OhsomeError(message) from error

    def latest_snapshot(self) -> datetime:
        """Query and parse the latest extract timestamp from the API metadata endpoint."""
        payload = self._get_bytes(
            method="GET",
            endpoint=f"{self.base_url}/metadata",
            parameters={},
            suffix=".json",
        )
        try:
            metadata = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise OhsomeError("ohsome metadata response was not valid JSON.") from error
        return _latest_timestamp(metadata)

    def features(
        self,
        bbox: tuple[float, float, float, float],
        snapshot_date: date,
        filter_text: str,
    ) -> gpd.GeoDataFrame:
        """Request one tile of OSM geometries and return its geospatial table."""
        method, endpoint, parameters = build_ohsome_request(
            api_version=self.api_version,
            bbox=bbox,
            snapshot_date=snapshot_date,
            filter_text=filter_text,
            base_url=self.base_url,
        )
        payload = self._get_bytes(
            method=method,
            endpoint=endpoint,
            parameters=parameters,
            suffix=".parquet" if self.api_version == "v2" else ".geojson",
        )
        try:
            if self.api_version == "v2":
                return gpd.read_parquet(BytesIO(payload))
            return gpd.read_file(BytesIO(payload))
        except (ValueError, OSError, ImportError) as error:
            raise OhsomeError(
                f"Could not decode ohsome {self.api_version} extraction response as "
                f"{'GeoParquet' if self.api_version == 'v2' else 'GeoJSON'}: {error}"
            ) from error


def _combine_tiles(
    client: OhsomeClient,
    bbox: tuple[float, float, float, float],
    snapshot_date: date,
    filter_text: str,
) -> gpd.GeoDataFrame:
    """Fetch each bounded tile and combine duplicate-free feature rows."""
    pieces = [
        client.features(tile, snapshot_date, filter_text)
        for tile in tile_bboxes(bbox)
    ]
    nonempty = [piece for piece in pieces if not piece.empty]
    if not nonempty:
        return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")

    combined = gpd.GeoDataFrame(pd.concat(nonempty, ignore_index=True), geometry="geometry")
    identity_column = next(
        (column for column in ("@osmId", "osm_id", "osmId", "id") if column in combined.columns),
        None,
    )
    if identity_column:
        combined = combined.drop_duplicates(subset=identity_column)
    else:
        combined = combined.drop_duplicates(subset="geometry")
    return combined.set_crs("EPSG:4326", allow_override=True)


def _settlement_centroids(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Replace polygonal settlements with projected centroids and retain points only."""
    if frame.empty:
        return frame
    frame = frame.to_crs("EPSG:4326")
    polygonal = frame.geometry.geom_type.isin(("Polygon", "MultiPolygon"))
    centroids = frame.loc[polygonal].to_crs("EPSG:3857").geometry.centroid.to_crs("EPSG:4326")
    frame.loc[polygonal, "geometry"] = centroids
    point_mask = frame.geometry.geom_type.isin(("Point", "MultiPoint"))
    return frame.loc[point_mask].copy()


def _normalise_road_class(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add a uniform road_class field from the OSM highway tag."""
    if "highway" in frame.columns:
        frame["road_class"] = frame["highway"].astype("string")
    elif "tags" in frame.columns:
        frame["road_class"] = frame["tags"].map(
            lambda tags: tags.get("highway") if isinstance(tags, dict) else None
        ).astype("string")
    else:
        frame["road_class"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame


def _empty_layer() -> gpd.GeoDataFrame:
    """Create an empty WGS84 layer for a class with no matching features."""
    return gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"), crs="EPSG:4326")


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
) -> dict[str, object]:
    """Download, normalize, and write all required OSM classes to one GeoPackage."""
    client = OhsomeClient(settings, cache, offline=offline)
    latest_timestamp = client.latest_snapshot()
    snapshot_date = select_osm_snapshot(event_date, latest_timestamp, requested_snapshot)
    output_path = output_dir / "osm_pre_event.gpkg"
    output_dir.mkdir(parents=True, exist_ok=True)

    layers: dict[str, gpd.GeoDataFrame] = {}
    for layer_name, filter_text in LAYER_FILTERS.items():
        frame = _combine_tiles(client, bbox, snapshot_date, filter_text)
        if layer_name == "roads":
            frame = _normalise_road_class(frame)
        if layer_name == "settlements":
            frame = _settlement_centroids(frame)
        if not frame.empty:
            frame = frame.to_crs(grid.crs)
        layers[layer_name] = frame

    for index, (layer_name, frame) in enumerate(layers.items()):
        if frame.empty:
            frame = _empty_layer().to_crs(grid.crs)
        frame.to_file(
            output_path,
            layer=layer_name,
            driver="GPKG",
            mode="w" if index == 0 else "a",
            metadata={"osm_snapshot_date": snapshot_date.isoformat()},
        )

    manifest.add_input(
        name="osm_pre_event.gpkg",
        role="input",
        source=OSM_SOURCE,
        license=OSM_LICENSE,
        retrieved_utc=min(client.retrieved_times, default=datetime.now(UTC)),
        path=output_path,
        settings=settings,
    )
    settlements = layers["settlements"]
    syabrubesi_present = False
    if not settlements.empty and "name" in settlements.columns:
        syabrubesi_present = (
            settlements["name"].astype("string").str.casefold().eq("syabrubesi").any()
        )

    return {
        "path": output_path,
        "snapshot_date": snapshot_date,
        "latest_snapshot": latest_timestamp,
        "api_version": client.api_version,
        "counts": {name: len(frame) for name, frame in layers.items()},
        "syabrubesi_present": bool(syabrubesi_present),
    }
