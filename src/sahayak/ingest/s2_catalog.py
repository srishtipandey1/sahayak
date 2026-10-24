"""Sentinel-2 Catalog request and metadata parsing for read-only diagnostics."""

import re
from datetime import date, datetime
from typing import Any

SENTINEL_2_L2A_COLLECTION = "sentinel-2-l2a"
S2_TILE_PATTERN = re.compile(r"(?:^|_)T(?P<tile>\d{2}[A-Z]{3})(?:_|$)", re.IGNORECASE)


def build_s2_catalog_request(
    bbox: tuple[float, float, float, float],
    from_date: date,
    to_date: date,
    *,
    next_page: object | None = None,
) -> dict[str, object]:
    """Build an inclusive UTC-date interval search without requesting pixel data."""
    if from_date > to_date:
        raise ValueError("Sentinel-2 catalog start date must not be after end date.")
    body: dict[str, object] = {
        "bbox": list(bbox),
        "datetime": (
            f"{from_date.isoformat()}T00:00:00Z/"
            f"{to_date.isoformat()}T23:59:59Z"
        ),
        "collections": [SENTINEL_2_L2A_COLLECTION],
        "limit": 100,
    }
    if next_page is not None:
        body["next"] = next_page
    return body


def parse_s2_catalog_item(item: dict[str, Any]) -> dict[str, object]:
    """Normalize one STAC feature without assuming optional tile or orbit fields."""
    properties = item.get("properties")
    if not isinstance(properties, dict):
        raise ValueError("Sentinel-2 Catalog item has no properties object.")
    item_id = item.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("Sentinel-2 Catalog item has no id.")
    acquired_at = properties.get("datetime")
    if not isinstance(acquired_at, str):
        raise ValueError(f"Sentinel-2 Catalog item {item_id!r} has no datetime.")
    try:
        parsed_datetime = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(
            f"Sentinel-2 Catalog item {item_id!r} has an invalid datetime."
        ) from error
    if parsed_datetime.tzinfo is None:
        raise ValueError(f"Sentinel-2 Catalog item {item_id!r} datetime has no timezone.")

    tile_id = properties.get("s2:tile_id")
    if not isinstance(tile_id, str) or not tile_id:
        match = S2_TILE_PATTERN.search(item_id)
        tile_id = match.group("tile").upper() if match else None
    orbit = properties.get("sat:relative_orbit")
    if (
        isinstance(orbit, bool)
        or not isinstance(orbit, (int, float))
        or orbit < 1
        or not float(orbit).is_integer()
    ):
        orbit = None
    cloud_cover = properties.get("eo:cloud_cover")
    if isinstance(cloud_cover, bool) or not isinstance(cloud_cover, (int, float)):
        cloud_cover = None
    return {
        "id": item_id,
        "acquired_at": parsed_datetime,
        "platform": properties.get("platform"),
        "tile_id": tile_id,
        "relative_orbit": int(orbit) if orbit is not None else None,
        "eo_cloud_cover": float(cloud_cover) if cloud_cover is not None else None,
    }
