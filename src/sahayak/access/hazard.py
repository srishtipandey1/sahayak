"""Detection-derived hazard footprints used by exposure and access analysis."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyogrio
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id

HAZARD_LEVELS = ("strict", "default", "lenient")
OPTICAL_CLASSES = {
    "scour_debris": 1,
    "new_water": 2,
    "existing_channel_or_bare": 3,
    "bright_recovered": 4,
}


def merge_facts(path: Path, section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Merge a generated facts section without replacing other map measurements."""
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(current, dict):
            raise ValueError("facts.json must contain a JSON object.")
    else:
        current = {}
    current[section] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary:
        temporary.write(json.dumps(current, indent=2, allow_nan=False))
        temporary_path = Path(temporary.name)
    temporary_path.replace(path)
    return current


def _mask_for_level(classes: np.ndarray, confidence: np.ndarray, level: str) -> np.ndarray:
    """Build one of the three documented optical detection masks."""
    if level == "strict":
        return (classes == 1) & np.isfinite(confidence) & (confidence >= 0.6)
    if level in {"default", "lenient"}:
        return np.isin(classes, (1, 2, 4))
    raise ValueError(f"Unknown hazard level: {level}")


def _raster_mask_geometry(mask: np.ndarray, transform: Any) -> Any | None:
    """Polygonize a boolean mask and dissolve adjacent detected pixels."""
    if not mask.any():
        return None
    polygons = [
        shape(geometry)
        for geometry, value in shapes(
            mask.astype("uint8"), mask=mask, transform=transform, connectivity=8
        )
        if value == 1
    ]
    geometry = unary_union(polygons)
    return geometry if not geometry.is_empty else None


def _read_extra_source(
    run_dir: Path,
    source: str,
    reference_shape: tuple[int, int],
    reference_crs: Any,
    reference_transform: Any,
) -> np.ndarray:
    """Load an explicitly requested auxiliary classification on the optical grid."""
    candidates = {
        "radar": run_dir / "flood" / "flood_class.tif",
        "ml": run_dir / "ml" / "ml_class.tif",
    }
    path = candidates[source]
    if not path.is_file():
        raise FileNotFoundError(f"Requested {source} hazard source is missing: {path.name}")
    with rasterio.open(path) as raster:
        values = raster.read(1)
        if (
            values.shape != reference_shape
            or raster.crs != reference_crs
            or raster.transform != reference_transform
        ):
            raise ValueError(f"Requested {source} hazard source is not aligned to optical grid.")
        if source == "radar":
            return np.isin(values, (1, 2))
        return values == 1


def build_hazards(
    *,
    run_id: str,
    sources: tuple[str, ...] = ("optical",),
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Create strict, default, and buffered lenient detection footprints."""
    validate_run_id(run_id)
    allowed = {"optical", "radar", "ml"}
    if not sources or len(set(sources)) != len(sources) or set(sources) - allowed:
        raise ValueError("Hazard sources must be unique values from optical, radar, and ml.")
    if "optical" not in sources:
        raise ValueError("Optical hazard classes are required as the base source.")
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    optical_path = run_dir / "optical" / "optical_class.tif"
    confidence_path = run_dir / "optical" / "optical_confidence.tif"
    if not optical_path.is_file():
        raise FileNotFoundError(f"Optical hazard input is missing: {optical_path.name}")
    if not confidence_path.is_file():
        raise FileNotFoundError(f"Optical confidence input is missing: {confidence_path.name}")
    facts_path = run_dir / "facts.json"
    if not facts_path.is_file():
        raise FileNotFoundError("Run is missing facts.json needed for optical reliability.")
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    optical_facts = facts.get("optical")
    if not isinstance(optical_facts, dict) or not optical_facts.get("reliability"):
        raise ValueError("facts.json has no optical reliability measurement.")
    reliability = str(optical_facts["reliability"])

    with rasterio.open(optical_path) as classes_raster, rasterio.open(
        confidence_path
    ) as confidence_raster:
        classes = classes_raster.read(1)
        confidence = confidence_raster.read(1).astype(np.float32)
        if classes.shape != confidence.shape:
            raise ValueError("Optical class and confidence rasters are not aligned.")
        if classes_raster.crs is None or not classes_raster.crs.is_projected:
            raise ValueError("Optical hazard raster must use a projected CRS.")
        if classes_raster.crs.linear_units.casefold() not in {
            "metre",
            "meter",
            "metres",
            "meters",
        }:
            raise ValueError("Optical hazard raster must use metre units for metric buffers.")
        transform = classes_raster.transform
        crs = classes_raster.crs
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
    extra_mask = np.zeros(classes.shape, dtype=bool)
    for source in sources:
        if source != "optical":
            extra_mask |= _read_extra_source(
                run_dir,
                source,
                classes.shape,
                crs,
                transform,
            )

    geometries: dict[str, Any | None] = {}
    areas: dict[str, float] = {}
    strict_mask = _mask_for_level(classes, confidence, "strict") | extra_mask
    default_mask = _mask_for_level(classes, confidence, "default") | extra_mask
    for level, mask in (
        ("strict", strict_mask),
        ("default", default_mask),
        ("lenient", default_mask),
    ):
        geometry = _raster_mask_geometry(mask, transform)
        if level == "lenient" and geometry is not None:
            geometry = geometry.buffer(30.0)
        geometries[level] = geometry
        areas[level] = (
            float(geometry.area / 10_000) if geometry is not None else 0.0
        )
    if not np.all(strict_mask <= default_mask):
        raise RuntimeError("Strict hazard detections must be a subset of default detections.")

    output_path = run_dir / "hazard.gpkg"
    rows = [
        {
            "scenario": level,
            "area_ha": areas[level],
            "optical_reliability": reliability,
            "sources": ",".join(sources),
            "is_detection_not_ground_truth": True,
            "geometry": geometries[level],
        }
        for level in HAZARD_LEVELS
        if geometries[level] is not None
    ]
    frame = gpd.GeoDataFrame(
        rows,
        columns=(
            "scenario",
            "area_ha",
            "optical_reliability",
            "sources",
            "is_detection_not_ground_truth",
            "geometry",
        ),
        geometry="geometry",
        crs=crs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hazard.", dir=run_dir) as temp_dir:
        temporary_path = Path(temp_dir) / output_path.name
        pyogrio.write_dataframe(
            frame,
            temporary_path,
            layer="hazards",
            driver="GPKG",
            layer_metadata={
                "footprints_are_detections_not_ground_truth": "true",
                "optical_reliability": reliability,
            },
        )
        temporary_path.replace(output_path)
    summary = {
        "levels": {
            level: {
                "area_ha": areas[level],
                "optical_reliability": reliability,
                "sources": list(sources),
                "is_detection_not_ground_truth": True,
            }
            for level in HAZARD_LEVELS
        },
        "source_class_codes": OPTICAL_CLASSES,
        "output": output_path.name,
        "pixel_area_m2": pixel_area_m2,
        "footprints_are_detections_not_ground_truth": True,
    }
    merge_facts(facts_path, "hazard", summary)
    return {"path": output_path, "facts": summary, "geometries": geometries}


def read_hazards(path: Path) -> dict[str, Any]:
    """Read and validate dissolved hazard geometries from the GeoPackage."""
    if not path.is_file():
        raise FileNotFoundError(f"Hazard GeoPackage is missing: {path.name}")
    frame = gpd.read_file(path, layer="hazards")
    if frame.crs is None:
        raise ValueError("Hazard footprints have no CRS.")
    result = {str(row.scenario): row.geometry for row in frame.itertuples()}
    missing = set(HAZARD_LEVELS) - result.keys()
    return {**{name: None for name in missing}, **result}
