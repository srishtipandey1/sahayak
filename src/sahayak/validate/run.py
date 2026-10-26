"""Run a frozen, validation-only comparison against an EMSR927 reference product."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyogrio import list_layers
from shapely.geometry import box
from shapely.ops import unary_union

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id
from sahayak.provenance import open_reference, sha256_file
from sahayak.validate.metrics import area_metrics, not_comparable, point_match_metrics

REFERENCE_RELATIVE_PATH = Path(
    "data/reference/emsr927/EMSR927_AOI01_GRA_PRODUCT_v1.gpkg"
)
ROAD_COMPARABLE_GRADES = ("Destroyed", "Possibly damaged")


def _repository_root() -> Path:
    """Find the repository root containing pyproject.toml."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not find the repository pyproject.toml.")


def _config_hashes(root: Path) -> dict[str, str]:
    """Hash the fixed validation-sensitive optical and access configs."""
    return {
        "configs/optical.toml": sha256_file(root / "configs" / "optical.toml"),
        "configs/access.toml": sha256_file(root / "configs" / "access.toml"),
    }


def freeze_configs(
    root: Path,
    frozen_path: Path,
    *,
    refreeze: bool = False,
    commit_hash: str | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Create or verify a config freeze, requiring explicit refreeze on drift."""
    config_hashes = _config_hashes(root)
    already_frozen = frozen_path.is_file()
    if already_frozen:
        existing = json.loads(frozen_path.read_text(encoding="utf-8"))
        frozen_hashes = existing.get("config_sha256")
        changed = sorted(
            name
            for name, digest in config_hashes.items()
            if not isinstance(frozen_hashes, dict) or frozen_hashes.get(name) != digest
        )
        if changed and not refreeze:
            raise RuntimeError(
                "Frozen validation config hash mismatch for "
                f"{', '.join(changed)}; rerun with --refreeze to record new hashes."
            )
        if not changed:
            return {"frozen": existing, "refrozen": False}

    if commit_hash is None:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit_hash = completed.stdout.strip()
    record = {
        "config_sha256": config_hashes,
        "git_commit": commit_hash,
        "timestamp_utc": timestamp or datetime.now(UTC).isoformat(),
    }
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"frozen": record, "refrozen": bool(already_frozen and refreeze)}


def _metric_crs(frame: gpd.GeoDataFrame) -> Any:
    """Require a projected metric CRS for area and length comparisons."""
    if frame.crs is None or not frame.crs.is_projected:
        raise ValueError("Validation run layers must have a projected CRS.")
    units = {axis.unit_name.casefold() for axis in frame.crs.axis_info}
    if not units or not units <= {"metre", "meter", "metres", "meters"}:
        raise ValueError("Validation run layers must use metre units.")
    return frame.crs


def _union(frame: gpd.GeoDataFrame):
    """Dissolve non-empty geometries and fail on empty comparison layers."""
    geometries = [
        geometry
        for geometry in frame.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        raise ValueError("A required validation geometry layer is empty.")
    return unary_union(geometries)


def _read_reference(
    path: Path,
    layer: str,
    settings: Settings,
) -> gpd.GeoDataFrame | Any:
    """Read one reference table through the restricted provenance gate."""
    return open_reference(path, layer=layer, settings=settings)


def _clip_frame(
    frame: gpd.GeoDataFrame,
    clip_geometry: Any,
    target_crs: Any,
) -> gpd.GeoDataFrame:
    """Reproject and retain features intersecting the focus/AOI overlap."""
    if frame.crs is None:
        raise ValueError("Validation source layer has no CRS.")
    projected = frame.to_crs(target_crs)
    return projected.loc[projected.geometry.intersects(clip_geometry)].copy()


def _ratio(numerator: float, denominator: float) -> float | None:
    """Return a ratio or None when the reference denominator is zero."""
    return numerator / denominator if denominator > 0 else None


def _matching_infrastructure(
    *,
    run_dir: Path,
    reference_path: Path,
    clip_geometry: Any,
    metric_crs: Any,
    hazard_geometry: Any,
    settings: Settings,
) -> dict[str, Any]:
    """Summarize bridge and road counts/lengths without inventing zero scores."""
    reference_bridges = _read_reference(
        reference_path,
        "transportationP_v1",
        settings,
    )
    reference_bridges = reference_bridges.loc[
        reference_bridges["damage_gra"].eq("Destroyed")
    ]
    reference_bridges = _clip_frame(reference_bridges, clip_geometry, metric_crs)
    model_bridges = gpd.read_file(
        run_dir / "exposure.gpkg",
        layer="bridge_exposure",
    )
    model_bridges = model_bridges.loc[
        model_bridges["scenario"].eq("default") & model_bridges["is_hit_20m"].fillna(False)
    ]
    model_bridges = _clip_frame(model_bridges, clip_geometry, metric_crs)

    reference_roads = _read_reference(
        reference_path,
        "transportationL_v1",
        settings,
    )
    reference_roads = reference_roads.loc[
        reference_roads["damage_gra"].isin(ROAD_COMPARABLE_GRADES)
    ]
    reference_roads = _clip_frame(reference_roads, clip_geometry, metric_crs)
    reference_road_length_m = float(
        reference_roads.geometry.intersection(clip_geometry).length.sum()
    )

    model_roads = gpd.read_file(run_dir / "exposure.gpkg", layer="road_exposure")
    model_roads = model_roads.loc[
        model_roads["scenario"].eq("default") & model_roads["is_hit"].fillna(False)
    ]
    model_roads = _clip_frame(model_roads, clip_geometry, metric_crs)
    model_road_length_m = float(
        model_roads.geometry.intersection(hazard_geometry).intersection(clip_geometry).length.sum()
    )
    return {
        "bridges": {
            "our_default_hit_count": int(len(model_bridges)),
            "emsr927_destroyed_point_count": int(len(reference_bridges)),
            "classification": (
                "Counts are reported side by side; feature matching is not asserted."
            ),
        },
        "roads": {
            "our_default_hit_length_km": model_road_length_m / 1000,
            "emsr927_destroyed_or_possibly_damaged_length_km": (
                reference_road_length_m / 1000
            ),
            "emsr927_grade_filter": list(ROAD_COMPARABLE_GRADES),
            "length_crs": str(metric_crs),
            "classification": (
                "Lengths are reported side by side; they are not precision/recall scores."
            ),
        },
    }


def _search_osm_named_items(run_dir: Path, overlap_wgs84: Any) -> dict[str, Any]:
    """Check available pre-event OSM tags for the reference facility names/classes."""
    found: dict[str, list[dict[str, Any]]] = {
        "hydroelectric_power_plant": [],
        "helipad": [],
    }
    search_specs = {
        "hydroelectric_power_plant": ("langtang khola", "hydroelectric", "power"),
        "helipad": ("helipad", "heliport", "aeroway=helipad"),
    }
    for filename in ("osm_pre_event.gpkg", "network_osm_pre_event.gpkg"):
        path = run_dir / filename
        if not path.is_file():
            continue
        for layer, _ in list_layers(path):
            frame = gpd.read_file(path, layer=layer)
            if frame.crs is None:
                continue
            frame = frame.to_crs("EPSG:4326")
            frame = frame.loc[frame.geometry.intersects(overlap_wgs84)]
            if frame.empty:
                continue
            searchable = frame.drop(columns="geometry").apply(
                lambda row: " ".join(str(value) for value in row),
                axis=1,
            )
            for item, terms in search_specs.items():
                matches = frame.loc[
                    searchable.str.contains("|".join(terms), case=False, regex=True)
                ]
                for _, row in matches.iterrows():
                    found[item].append(
                        {
                            "source_file": filename,
                            "layer": layer,
                            "osm_id": str(row.get("osm_id", "")),
                            "name": str(row.get("name", "")),
                            "tags_json": str(row.get("tags_json", "")),
                        }
                    )
    return {
        "pipeline_separate_classification": (
            "The damage pipeline does not separately classify these facility types."
        ),
        "osm_pre_event_tag_search": found,
    }


def _acquisition_metadata(reference_path: Path, settings: Settings) -> list[dict[str, str]]:
    """Read the reference product's post-event imagery date from its source table."""
    sources = _read_reference(reference_path, "source_v1", settings)
    rows: list[dict[str, str]] = []
    for _, row in sources.iterrows():
        if str(row.get("eventphase", "")).casefold() != "post-event":
            continue
        source_date = str(row.get("src_date", ""))
        try:
            parsed_date = datetime.strptime(source_date, "%d/%m/%Y").date()
        except ValueError:
            continue
        rows.append(
            {
                "source": str(row.get("source_nam", "")),
                "acquisition_date": parsed_date.isoformat(),
            }
        )
    return rows


def _write_comparison_figure(
    *,
    run_dir: Path,
    output_path: Path,
    metric_crs: Any,
    region_geometry: Any,
    our_hazard: Any,
    reference_event: Any,
    destroyed_points: gpd.GeoDataFrame,
) -> None:
    """Plot detections, reference extent, and destroyed points over Sentinel RGB."""
    image_path = run_dir / "s2_post.tif"
    with rasterio.open(image_path) as source:
        if source.crs is None:
            raise ValueError("Post-event optical raster has no CRS.")
        band_lookup = {
            description.split(maxsplit=1)[0]: index
            for index, description in enumerate(source.descriptions, start=1)
            if description
        }
        band_indexes = [band_lookup.get(name) for name in ("B04", "B03", "B02")]
        if any(index is None for index in band_indexes):
            raise ValueError("Post-event optical raster is missing RGB bands.")
        rgb = np.stack(
            [
                source.read(int(index)).astype(np.float32)
                for index in band_indexes
            ],
            axis=-1,
        )
        valid = np.all(np.isfinite(rgb), axis=-1)
        for band in range(3):
            values = rgb[..., band][valid]
            if values.size:
                low, high = np.percentile(values, (2, 98))
                if high > low:
                    rgb[..., band] = np.clip((rgb[..., band] - low) / (high - low), 0, 1)
        rgb[~valid] = 0
        bounds = source.bounds
        transform = gpd.GeoSeries([region_geometry], crs=metric_crs).to_crs(source.crs).iloc[0]
        if not transform.intersects(box(bounds.left, bounds.bottom, bounds.right, bounds.top)):
            raise ValueError("Validation overlap does not intersect the post-event image.")
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)

    fig, axis = plt.subplots(figsize=(12, 9), constrained_layout=False)
    axis.imshow(rgb, extent=extent, origin="upper")
    gpd.GeoSeries([region_geometry], crs=metric_crs).boundary.plot(
        ax=axis, color="white", linewidth=0.8, label="Focus/AOI overlap"
    )
    gpd.GeoSeries([reference_event], crs=metric_crs).plot(
        ax=axis, facecolor="none", edgecolor="#ef4444", linewidth=1.7,
        label="EMSR927 observed landslide",
    )
    if not our_hazard.is_empty:
        gpd.GeoSeries([our_hazard], crs=metric_crs).plot(
            ax=axis, facecolor="#2563eb", edgecolor="#1d4ed8", alpha=0.40,
            linewidth=1.2, label="Sahayak optical default footprint",
        )
    if not destroyed_points.empty:
        destroyed_points.plot(
            ax=axis, color="#facc15", edgecolor="black", markersize=12,
            label="EMSR927 destroyed buildings",
        )
    axis.set_title("Sahayak optical detections vs EMSR927 validation reference")
    unit_name = metric_crs.axis_info[0].unit_name
    axis.set_xlabel(f"Easting ({unit_name})")
    axis.set_ylabel(f"Northing ({unit_name})")
    axis.legend(loc="best")
    fig.subplots_adjust(bottom=0.12)
    fig.text(
        0.5,
        0.025,
        "European Union, Copernicus Emergency Management Service data (EMSR927)",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_summary(path: Path, result: dict[str, Any]) -> None:
    """Write an honest compact Markdown summary from the recorded measurements."""
    extent = result["extent"]
    buildings = result["buildings"]
    infrastructure = result["infrastructure"]
    date_info = result["timing"]
    reference_facility_items = infrastructure["reference_facilities"]["items"]
    reference_helipad_names = infrastructure["reference_facilities"][
        "reference_transportation_area_names"
    ]
    osm_items = infrastructure["named_items"]["osm_pre_event_tag_search"]
    facility_text = ", ".join(
        f"{item['name']} ({item['grade']})" for item in reference_facility_items
    ) or "none recorded in the overlap"
    helipad_text = ", ".join(reference_helipad_names) or "none recorded in the overlap"
    plant_osm_text = (
        "matching pre-event OSM feature found"
        if osm_items["hydroelectric_power_plant"]
        else "no matching pre-event OSM name/tag found in the searched layers"
    )
    helipad_osm_text = (
        "matching pre-event OSM feature found"
        if osm_items["helipad"]
        else "no matching pre-event OSM name/tag found in the searched layers"
    )
    summary = [
        "# EMSR927 validation comparison",
        "",
        "This is a validation-only comparison. EMSR927 is also not perfect ground truth.",
        "No optical, flood, or access threshold was tuned against this product; the configs",
        "were frozen before comparison.",
        "",
        "## Extent",
        "",
        (
            "| Reference grade set | Sahayak area (ha) | EMSR927 area (ha) | "
            "IoU | Precision | Recall | F1 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in extent["by_grade"].items():
        summary.append(
            f"| {name} | {item['predicted_area_ha']:.3f} | {item['reference_area_ha']:.3f} | "
            f"{_fmt(item['metrics']['iou'])} | {_fmt(item['metrics']['precision'])} | "
            f"{_fmt(item['metrics']['recall'])} | {_fmt(item['metrics']['f1'])} |"
        )
    summary.extend(
        [
            "",
            "## Buildings",
            "",
            (
                "| EMSR927 threshold | Sahayak inside count | Reference count | "
                "Matched | Precision | Recall |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in buildings.items():
        summary.append(
            f"| {name} | {item['prediction_count']} | {item['reference_count']} | "
            f"{item['matched_count']} | {_fmt(item['precision'])} | {_fmt(item['recall'])} |"
        )
    roads = infrastructure["roads"]
    bridges = infrastructure["bridges"]
    summary.extend(
        [
            "",
            "## Infrastructure and scope",
            "",
            f"- Bridges: Sahayak default hit count {bridges['our_default_hit_count']}; "
            f"EMSR927 destroyed points {bridges['emsr927_destroyed_point_count']}. "
            "Counts are shown side by side, not scored.",
            f"- Roads: Sahayak hit length {roads['our_default_hit_length_km']:.3f} km; "
            "EMSR927 destroyed/possibly-damaged line length "
            f"{roads['emsr927_destroyed_or_possibly_damaged_length_km']:.3f} km. "
            "Lengths are descriptive, not precision/recall.",
            "- Reference facility items in the overlap: " + facility_text + ". The damage "
            "pipeline does not separately classify hydroelectric plants; " + plant_osm_text + ".",
            "- Reference transportation-area helipad names: " + helipad_text + ". The damage "
            "pipeline does not separately classify helipads; " + helipad_osm_text + ".",
            "- Cut-off settlements and flowpaths: not comparable; EMSR927 has no equivalent "
            "validation measurement for these outputs.",
            "",
            "## Timing and limitations",
            "",
            "- EMSR927 post-event acquisition date: "
            f"{date_info['emsr927_image_acquisition_date']}.",
            "- Sahayak post-event optical scene date: "
            f"{date_info['sahayak_optical_post_scene_date']}.",
            f"- Difference: {date_info['difference_days']} days; these are not temporally "
            "coincident observations.",
            "- EMSR927 map interpretation, image dates, positional accuracy, and minimum mapping "
            "unit also limit the comparison. Scores are measurements for this overlap only, "
            "not general model performance.",
            "",
            "## Provenance",
            "",
            f"- Reference product SHA-256: `{result['provenance']['reference_sha256']}`",
            f"- Config freeze commit: `{result['provenance']['freeze']['git_commit']}`",
            f"- Refreeze performed for this run: `{result['provenance']['refrozen']}`",
            "",
            "European Union, Copernicus Emergency Management Service data (EMSR927).",
        ]
    )
    path.write_text("\n".join(summary) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    """Format defined metrics while preserving undefined values."""
    return "not available" if value is None else f"{value:.4f}"


def run_validation(
    *,
    run_id: str,
    refreeze: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compare default optical/building outputs with the validation-only EMSR927 product."""
    validate_run_id(run_id)
    active_settings = settings or Settings()
    root = _repository_root()
    run_dir = active_settings.resolved_runs_dir / run_id
    output_dir = run_dir / "validate"
    reference_path = active_settings.reference_dir / REFERENCE_RELATIVE_PATH.relative_to(
        "data/reference"
    )
    frozen_info = freeze_configs(
        root,
        output_dir / "frozen.json",
        refreeze=refreeze,
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"EMSR927 validation reference is missing: {reference_path}")

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    facts = json.loads((run_dir / "facts.json").read_text(encoding="utf-8"))
    hazards = gpd.read_file(run_dir / "hazard.gpkg", layer="hazards")
    metric_crs = _metric_crs(hazards)
    default_rows = hazards.loc[hazards["scenario"].eq("default")]
    if default_rows.empty:
        raise ValueError("hazard.gpkg has no default optical scenario.")
    our_hazard = _union(default_rows)

    reference_aoi = _read_reference(reference_path, "areaOfInterestA_v1", active_settings)
    event_layer = _read_reference(reference_path, "observedEventA_v1", active_settings)
    reference_buildings = _read_reference(reference_path, "builtUpP_v1", active_settings)
    overlap_wgs84 = _union(reference_aoi).intersection(box(*manifest["bbox"]))
    if overlap_wgs84.is_empty:
        raise ValueError("Run focus bbox and EMSR927 AOI do not intersect.")
    region_geometry = (
        gpd.GeoSeries([overlap_wgs84], crs=reference_aoi.crs).to_crs(metric_crs).iloc[0]
    )
    clipped_hazard = our_hazard.intersection(region_geometry)
    clipped_event = _union(event_layer.to_crs(metric_crs)).intersection(region_geometry)

    extent_by_grade: dict[str, Any] = {}
    extent_by_grade["observed_landslide_extent"] = {
        "reference_layer": "observedEventA_v1",
        "reference_event_type": sorted(
            str(value) for value in event_layer["event_type"].dropna().unique()
        ),
        "reference_obj_desc": sorted(
            str(value) for value in event_layer["obj_desc"].dropna().unique()
        ),
        "reference_area_attribute_value_ha": float(event_layer.iloc[0]["area"]),
        "reference_area_attribute_unit": (
            "hectares, verified by projected event geometry area"
        ),
        "predicted_area_ha": float(clipped_hazard.area / 10_000),
        "reference_area_ha": float(clipped_event.area / 10_000),
        "metrics": area_metrics(clipped_hazard, clipped_event),
    }

    building_layer = gpd.read_file(run_dir / "exposure.gpkg", layer="building_exposure")
    predictions = building_layer.loc[
        building_layer["scenario"].eq("default") & building_layer["inside"].fillna(False)
    ]
    predictions = _clip_frame(predictions, region_geometry, metric_crs)
    prediction_geometries = [
        geometry.intersection(region_geometry)
        for geometry in predictions.geometry
        if geometry is not None and not geometry.is_empty
    ]
    building_results: dict[str, Any] = {}
    destroyed_points_for_map: gpd.GeoDataFrame | None = None
    for grade_name, grades in (
        ("destroyed_only", ("Destroyed",)),
        ("all_grades", None),
    ):
        reference_subset = (
            reference_buildings
            if grades is None
            else reference_buildings.loc[reference_buildings["damage_gra"].isin(grades)]
        )
        reference_subset = _clip_frame(reference_subset, region_geometry, metric_crs)
        reference_points = [
            geometry
            for geometry in reference_subset.geometry
            if geometry is not None and not geometry.is_empty
        ]
        result = point_match_metrics(
            prediction_geometries,
            reference_points,
            tolerance_m=25.0,
        )
        result["reference_grades"] = (
            sorted(str(value) for value in reference_buildings["damage_gra"].dropna().unique())
            if grades is None
            else list(grades)
        )
        building_results[grade_name] = result
        if grade_name == "destroyed_only":
            destroyed_points_for_map = reference_subset

    if destroyed_points_for_map is None:
        raise RuntimeError("Destroyed-building reference points were not prepared.")

    infrastructure = _matching_infrastructure(
        run_dir=run_dir,
        reference_path=reference_path,
        clip_geometry=region_geometry,
        metric_crs=metric_crs,
        hazard_geometry=clipped_hazard,
        settings=active_settings,
    )
    reference_facilities = _read_reference(
        reference_path,
        "facilitiesA_v1",
        active_settings,
    )
    reference_transportation_areas = _read_reference(
        reference_path,
        "transportationA_v1",
        active_settings,
    )
    reference_facilities = _clip_frame(
        reference_facilities,
        region_geometry,
        metric_crs,
    )
    reference_transportation_areas = _clip_frame(
        reference_transportation_areas,
        region_geometry,
        metric_crs,
    )
    facility_names = sorted(str(value) for value in reference_facilities["name"].dropna().unique())
    helipad_names = sorted(
        str(value)
        for value in reference_transportation_areas["name"].dropna().unique()
    )
    infrastructure["named_items"] = _search_osm_named_items(run_dir, overlap_wgs84)
    infrastructure["reference_facilities"] = {
        "items": [
            {
                "name": name,
                "grade": str(grade),
                "obj_type": str(obj_type),
            }
            for name, grade, obj_type in zip(
                reference_facilities["name"],
                reference_facilities["damage_gra"],
                reference_facilities["obj_type"],
                strict=True,
            )
        ],
        "reference_transportation_area_names": helipad_names,
        "facility_name_values": facility_names,
    }
    reference_dates = _acquisition_metadata(reference_path, active_settings)
    if not reference_dates:
        raise ValueError("EMSR927 source metadata has no parseable post-event acquisition date.")
    scene_dates = facts.get("optical", {}).get("provenance", {}).get("scene_dates", {})
    post_scene = scene_dates.get("post")
    if not post_scene:
        raise ValueError("facts.json has no post-event optical scene date.")
    post_date = datetime.fromisoformat(post_scene.replace("Z", "+00:00")).date()
    emsr_date = date.fromisoformat(reference_dates[0]["acquisition_date"])

    validation: dict[str, Any] = {
        "run_id": run_id,
        "scope": {
            "focus_bbox_wgs84": manifest["bbox"],
            "emsr927_aoi_id": str(reference_aoi.iloc[0]["area_id"]),
            "overlap_area_ha": float(region_geometry.area / 10_000),
            "metric_crs": str(metric_crs),
        },
        "extent": {
            "by_grade": extent_by_grade,
            "other_damage_grades": not_comparable(
                "observedEventA_v1 is an event extent, not a per-grade damage footprint."
            ),
        },
        "buildings": building_results,
        "infrastructure": infrastructure,
        "timing": {
            "emsr927_image_sources": reference_dates,
            "emsr927_image_acquisition_date": emsr_date.isoformat(),
            "sahayak_optical_post_scene_date": post_date.isoformat(),
            "difference_days": (post_date - emsr_date).days,
        },
        "not_comparable": {
            "cut_off_settlements": not_comparable(
                "EMSR927 does not provide settlement road-access/cut-off ground truth."
            ),
            "flowpath": not_comparable(
                "EMSR927 does not provide a terrain-only downstream flowpath reference."
            ),
        },
        "limitations": [
            "EMSR927 is also not perfect ground truth.",
            "The reference image and Sahayak optical post-scene dates differ.",
            "Comparison is limited to the intersection of the run focus bbox and EMSR927 AOI.",
            "No thresholds were tuned using EMSR927.",
        ],
        "output_files": {
            "validation_json": "runs/<run_id>/validate/validation.json",
            "comparison_png": "runs/<run_id>/validate/comparison.png",
            "summary_markdown": "docs/validation.md",
        },
        "provenance": {
            "reference_product": str(REFERENCE_RELATIVE_PATH),
            "reference_sha256": sha256_file(reference_path),
            "reference_credit": (
                "European Union, Copernicus Emergency Management Service data (EMSR927)"
            ),
            "freeze": frozen_info["frozen"],
            "refrozen": frozen_info["refrozen"],
            "config_sha256_current": _config_hashes(root),
            "generated_utc": datetime.now(UTC).isoformat(),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison_figure(
        run_dir=run_dir,
        output_path=output_dir / "comparison.png",
        metric_crs=metric_crs,
        region_geometry=region_geometry,
        our_hazard=clipped_hazard,
        reference_event=clipped_event,
        destroyed_points=destroyed_points_for_map,
    )
    _write_summary(_repository_root() / "docs" / "validation.md", validation)
    validation_path = output_dir / "validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return validation
