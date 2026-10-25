"""Command-line interface for Sahayak."""

import re
from datetime import UTC, date
from pathlib import Path
from typing import Annotated

import typer
from pydantic import SecretStr

from sahayak import __version__
from sahayak.config import Settings
from sahayak.ingest.s2_catalog import build_s2_catalog_request, parse_s2_catalog_item
from sahayak.logging_setup import configure_logging

app = typer.Typer(
    no_args_is_help=True,
    invoke_without_command=True,
    help="Sahayak flood-response mapping foundation.",
)

_BBOX_PATTERN = re.compile(
    r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*,\s*"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*$"
)
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show the installed version and exit.", is_eager=True),
    ] = False,
) -> None:
    """Initialize the command-line application."""
    configure_logging()
    if version:
        typer.echo(__version__)
        raise typer.Exit()


def _validate_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse and validate a minlon,minlat,maxlon,maxlat bounding box."""
    if not _BBOX_PATTERN.fullmatch(value):
        raise typer.BadParameter("Expected minlon,minlat,maxlon,maxlat.")
    try:
        min_lon, min_lat, max_lon, max_lat = (float(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise typer.BadParameter("Bounding-box coordinates must be numbers.") from error
    if not (-180 <= min_lon < max_lon <= 180):
        raise typer.BadParameter("Longitude bounds must be ordered and within -180 to 180.")
    if not (-90 <= min_lat < max_lat <= 90):
        raise typer.BadParameter("Latitude bounds must be ordered and within -90 to 90.")
    return min_lon, min_lat, max_lon, max_lat


def _validate_event_date(value: str) -> date:
    """Parse a strict YYYY-MM-DD date."""
    if not _DATE_PATTERN.fullmatch(value):
        raise typer.BadParameter("Expected date format YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        message = "Event date must be a valid calendar date in YYYY-MM-DD format."
        raise typer.BadParameter(message) from error


def _run_placeholder(
    command: str,
    bbox: str,
    event_date: str,
    run_id: str | None,
) -> None:
    """Validate common command parameters and report the unimplemented stage."""
    _validate_bbox(bbox)
    _validate_event_date(event_date)
    if run_id is not None and not run_id.strip():
        raise typer.BadParameter("Run ID must not be empty.")
    typer.echo(f"{command}: not implemented yet")


def _science_command(name: str):
    """Create a science-stage command with shared spatial and event options."""

    def command(
        bbox: Annotated[
            str,
            typer.Option("--bbox", help="Bounding box: minlon,minlat,maxlon,maxlat."),
        ],
        event_date: Annotated[str, typer.Option("--event-date", help="Event date (YYYY-MM-DD).")],
        run_id: Annotated[
            str | None,
            typer.Option("--run-id", help="Optional run identifier."),
        ] = None,
    ) -> None:
        _run_placeholder(name, bbox, event_date, run_id)

    command.__name__ = name
    command.__doc__ = f"Run the {name} stage (not implemented yet)."
    return command


for _command_name in ("flowpath", "report", "run"):
    app.command(name=_command_name)(_science_command(_command_name))


@app.command()
def ingest(
    bbox: Annotated[
        str,
        typer.Option("--bbox", help="Bounding box: minlon,minlat,maxlon,maxlat."),
    ],
    event_date: Annotated[str, typer.Option("--event-date", help="Event date (YYYY-MM-DD).")],
    run_id: Annotated[str, typer.Option("--run-id", help="Run identifier.")],
    osm_snapshot: Annotated[
        str | None,
        typer.Option("--osm-snapshot", help="Optional pre-event OSM snapshot date (YYYY-MM-DD)."),
    ] = None,
    skip: Annotated[
        str,
        typer.Option("--skip", help="Comma-separated stages to skip (s1,s2,dem,osm)."),
    ] = "",
    s1_baseline_count: Annotated[
        int,
        typer.Option(
            "--s1-baseline-count",
            min=1,
            max=3,
            help="Number of pre-event Sentinel-1 scenes, including the chosen scene (1-3).",
        ),
    ] = 3,
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Use cached responses only; fail on a cache miss."),
    ] = False,
) -> None:
    """Ingest Sentinel-1, Sentinel-2, the common grid, DEM, and pre-event OSM."""
    parsed_bbox = _validate_bbox(bbox)
    parsed_event_date = _validate_event_date(event_date)
    parsed_snapshot = None
    if osm_snapshot is not None:
        try:
            parsed_snapshot = _validate_event_date(osm_snapshot)
        except typer.BadParameter as error:
            raise typer.BadParameter(f"Invalid --osm-snapshot: {error.message}") from error
    skipped_stages = {stage.strip().lower() for stage in skip.split(",") if stage.strip()}
    try:
        from sahayak.ingest.run import format_ingest_summary, run_ingest

        result = run_ingest(
            bbox=parsed_bbox,
            event_date=parsed_event_date,
            run_id=run_id,
            osm_snapshot=parsed_snapshot,
            s1_baseline_count=s1_baseline_count,
            skip=skipped_stages,
            offline=offline,
            settings=Settings(),
        )
    except (ValueError, OSError, RuntimeError) as error:
        typer.echo(
            f"Ingestion failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error
    typer.echo(format_ingest_summary(result))


@app.command("ingest-network")
def ingest_network_command(
    run_id: Annotated[str, typer.Option("--run-id", help="Existing run identifier.")],
    bbox: Annotated[
        str,
        typer.Option("--bbox", help="Wide-area bbox: minlon,minlat,maxlon,maxlat."),
    ],
    offline: Annotated[
        bool,
        typer.Option("--offline", help="Use cached Overpass responses only."),
    ] = False,
) -> None:
    """Ingest wide-area, date-pinned OSM roads and access targets."""
    from sahayak.ingest.network import estimate_network_requests, ingest_network

    parsed_bbox = _validate_bbox(bbox)
    typer.echo(
        "Estimated Overpass tile requests before retries: "
        f"{estimate_network_requests(parsed_bbox)}"
    )
    try:
        result = ingest_network(
            run_id=run_id,
            bbox=parsed_bbox,
            settings=Settings(),
            offline=offline,
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(
            f"Network OSM ingestion failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Requests: {result['request_count']} | elapsed: {result['elapsed_seconds']:.1f}s"
    )
    typer.echo(f"Maximum element timestamp: {result['max_element_timestamp']}")
    typer.echo("Elements by layer:")
    for layer, count in result["counts"].items():
        typer.echo(f"  {layer}: {count}")
    typer.echo(f"Output: {result['path']}")


def _parse_hazard_sources(value: str) -> tuple[str, ...]:
    """Parse optical plus optional comma-separated radar/ML sources."""
    sources = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    allowed = {"optical", "radar", "ml"}
    if not sources or len(set(sources)) != len(sources) or set(sources) - allowed:
        raise typer.BadParameter(
            "Sources must be a unique comma-separated list of optical, radar, ml."
        )
    if "optical" not in sources:
        raise typer.BadParameter("Optical must be included as the base hazard source.")
    return sources


@app.command()
def damage(
    run_id: Annotated[str, typer.Option("--run-id", help="Existing mapped run identifier.")],
    sources: Annotated[
        str,
        typer.Option("--sources", help="Hazard sources: optical, optionally radar and/or ml."),
    ] = "optical",
) -> None:
    """Build detection-derived hazards and calculate feature exposure."""
    from sahayak.damage.run import run_damage

    selected_sources = _parse_hazard_sources(sources)
    try:
        result = run_damage(
            run_id=run_id,
            sources=selected_sources,
            settings=Settings(),
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(f"Damage analysis failed: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Damage analysis for run {run_id}")
    typer.echo(
        "Scenario | hazard area (ha) | buildings inside | buildings near 30m | "
        "bridges | settlements"
    )
    for level, values in result["facts"]["scenarios"].items():
        typer.echo(
            f"{level} | {values['hazard_area_ha']['value']:.4f} | "
            f"{values['buildings_inside']['value']} | {values['buildings_near_30m']['value']} | "
            f"{values['bridges_hit_within_20m']['value']} | "
            f"{values['settlements_affected']['value']}"
        )
        typer.echo(f"  Road km hit by class: {values['road_km_hit_by_class']}")
    typer.echo("Footprints are detections, not ground truth.")
    typer.echo(f"Output: {result['path']}")


@app.command()
def access(
    run_id: Annotated[str, typer.Option("--run-id", help="Existing mapped run identifier.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Routing assumptions TOML path."),
    ] = Path("configs/access.toml"),
    sources: Annotated[
        str,
        typer.Option("--sources", help="Hazard sources: optical, optionally radar and/or ml."),
    ] = "optical",
) -> None:
    """Compare settlement road access before and after each hazard scenario."""
    from sahayak.access.run import run_access

    selected_sources = _parse_hazard_sources(sources)
    try:
        result = run_access(
            run_id=run_id,
            config_path=config,
            sources=selected_sources,
            settings=Settings(),
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(f"Access analysis failed: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Access analysis for run {run_id}")
    typer.echo("Scenario | cut off | degraded | unaffected | not assessable")
    for level, values in result["facts"]["scenarios"].items():
        counts = values["counts_by_class"]
        typer.echo(
            f"{level} | {counts['cut_off']['value']} | {counts['degraded']['value']} | "
            f"{counts['unaffected']['value']} | {values['not_assessable_count']}"
        )
        typer.echo(f"  Cut-off settlements: {values['cut_off_settlements']}")
        typer.echo(
            f"  Blocking edges: {values['blocked_edge_count']}; "
            f"not assessable: {values['not_assessable_count']}"
        )
        for settlement, blockers in values["cut_off_blockers"].items():
            blocker_summary = ", ".join(
                f"{edge['road_name'] or 'unnamed'} "
                f"(class={edge['road_class']}, bridge={edge['bridge']})"
                for edge in blockers
            )
            typer.echo(f"  Blockers for {settlement}: {blocker_summary}")
    typer.echo(f"Robust cut-off settlements: {result['facts']['robust_cut_off_settlements']}")
    typer.echo("Footprints are detections, not ground truth.")
    typer.echo("Files:")
    for path in result["files"]:
        typer.echo(f"  {path.name}: {path.stat().st_size} bytes")


@app.command()
def flood(
    run_id: Annotated[str, typer.Option("--run-id", help="Existing ingested run identifier.")],
    max_hand: Annotated[
        float | None,
        typer.Option("--max-hand", help="Maximum HAND height in metres."),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", help="Flood configuration TOML path."),
    ] = Path("configs/flood.toml"),
) -> None:
    """Map post-event flood water and disturbed surfaces from an ingested run."""
    from sahayak.flood.run import run_flood
    from sahayak.flood.terrain import FloodTerrainError

    try:
        result = run_flood(
            run_id=run_id,
            max_hand_m=max_hand,
            config_path=config,
            settings=Settings(),
        )
    except (FloodTerrainError, ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(
            f"Flood mapping failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    flood_facts = result["facts"]
    typer.echo(f"Flood mapping for run {run_id}")
    typer.echo(
        f"VV Otsu water threshold: "
        f"{flood_facts['provenance']['thresholds_used']['vv_otsu_water_threshold']['value']:.2f} dB"
    )
    typer.echo(f"Valid pixel fraction: {flood_facts['valid_pixel_fraction']['value']:.4%}")
    typer.echo("Class               Area (ha)")
    for class_name, measurement in flood_facts["areas_ha_by_class"].items():
        typer.echo(f"{class_name:<20}{measurement['value']:>10.2f}")
    typer.echo(
        "Permanent water before restriction: "
        f"{flood_facts['permanent_water_area_before_restriction_ha']['value']:.2f} ha"
    )
    typer.echo(
        "Permanent water after restriction: "
        f"{flood_facts['permanent_water_area_after_restriction_ha']['value']:.2f} ha"
    )
    typer.echo("HAND sensitivity (area ha)")
    for row in flood_facts["sensitivity"]:
        areas = row["area_ha_by_class"]
        typer.echo(
            f"{row['max_hand']['value']:>5.1f} m: "
            f"new water {areas['new_water']['value']:.2f}; "
            f"disturbed {areas['disturbed_surface']['value']:.2f}; "
            f"permanent water {areas['permanent_water']['value']:.2f}; "
            f"dark low confidence {areas['dark_low_confidence']['value']:.2f}"
        )
    failed_quicklooks = result.get("failed_quicklooks", [])
    if failed_quicklooks:
        typer.echo("Failed quicklooks (mapping completed without these images):")
        for filename in failed_quicklooks:
            typer.echo(f"  {filename}")
    typer.echo(f"Outputs: {result['flood_dir']}")


@app.command(name="flood-diagnose")
def flood_diagnose(
    run_id: Annotated[str, typer.Option("--run-id", help="Existing ingested run identifier.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Flood configuration TOML path."),
    ] = Path("configs/flood.toml"),
) -> None:
    """Write candidate-funnel and pre-event null-pair diagnostics."""
    from sahayak.flood.diagnostics import format_flood_diagnostics, run_flood_diagnostics

    try:
        diagnostics = run_flood_diagnostics(
            run_id=run_id,
            config_path=config,
            settings=Settings(),
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(
            f"Flood diagnostics failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error
    typer.echo(format_flood_diagnostics(diagnostics))


@app.command()
def optical(
    run_id: Annotated[str, typer.Option("--run-id", help="Existing ingested run identifier.")],
    max_hand: Annotated[
        float | None,
        typer.Option("--max-hand", help="Maximum optical corridor HAND in metres."),
    ] = None,
    config: Annotated[
        Path,
        typer.Option("--config", help="Optical configuration TOML path."),
    ] = Path("configs/optical.toml"),
) -> None:
    """Run Sentinel-2 optical change detection and placebo analysis."""
    from sahayak.optical.run import run_optical

    try:
        result = run_optical(
            run_id=run_id,
            max_hand_m=max_hand,
            config_path=config,
            settings=Settings(),
        )
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(f"Optical analysis failed: {type(error).__name__}: {error}", err=True)
        raise typer.Exit(code=1) from error

    facts = result["facts"]
    typer.echo(f"Optical analysis for run {run_id}")
    typer.echo(f"Reliability: {facts['reliability']}")
    typer.echo(f"Control quality: {facts['control_quality']}")
    typer.echo("Real class areas (ha)")
    for name, value in facts["class_areas_ha"].items():
        typer.echo(f"  {name}: {value['value']:.4f}")
    typer.echo("Null-pair class areas (ha)")
    for pair_name in ("pre_pre", "post_post"):
        typer.echo(f"  {pair_name}")
        for class_name in ("scour_debris", "new_water", "bright_recovered"):
            area = facts["null_pairs"][pair_name]["class_areas_ha"][class_name]["value"]
            typer.echo(f"    {class_name}: {area:.4f}")
    typer.echo("Normalization coefficients (scale, offset)")
    for band, values in result["normalization"].items():
        typer.echo(f"  {band}: {values['scale']:.6f}, {values['offset']:.6f}")
    typer.echo("False alarms per 1000 valid pixels (whole tile and HAND corridor)")
    for pair_name in ("pre_pre", "post_post"):
        accounting = facts["null_pairs"][pair_name]["false_alarm_accounting"]
        typer.echo(
            f"  {pair_name} whole: "
            f"{accounting['false_alarms_per_1000_valid_pixels']}"
        )
        typer.echo(
            f"  {pair_name} corridor: "
            f"{accounting['corridor_false_alarms_per_1000_valid_pixels']}"
        )
    typer.echo("Funnel counts")
    for pair_name, classes in facts["null_pair"]["funnels"].items():
        typer.echo(f"  {pair_name}")
        for class_name, stages in classes.items():
            typer.echo(f"    {class_name}: {stages}")
    typer.echo("Sensitivity (max HAND, NDVI pre, delta NDVI; class areas ha)")
    for row in result["sensitivity"]:
        typer.echo(
            f"  {row['max_hand_m']:g}, {row['ndvi_pre_min']:.2f}, "
            f"{row['delta_ndvi_max']:.2f}: {row['class_areas_ha']}"
        )
    typer.echo("Bright-recovery sensitivity (B02 max, normalized dB04 minimum)")
    for row in result["bright_sensitivity"]:
        typer.echo(
            f"  {row['b02_max']:.2f}, {row['delta_b04_min']:.2f}: "
            f"class 1={row['class_1_area_ha']:.4f} ha, "
            f"class 4={row['class_4_area_ha']:.4f} ha, "
            f"union={row['class_1_4_union_area_ha']:.4f} ha, "
            f"B02 removed={row['b02_candidates_removed_fraction']:.4f}"
        )
    typer.echo(f"Footprint areas (ha): {facts['footprint_ha']}")
    typer.echo(f"Bright-recovery candidate accounting: {facts['bright_recovery']}")
    typer.echo(f"Radar corroboration: {facts['radar_corroboration']}")
    typer.echo(f"Failed quicklooks: {result['failed_quicklooks']}")
    typer.echo("Files:")
    for path in result["files"]:
        typer.echo(f"  {path.name}: {path.stat().st_size} bytes")


@app.command("optical-diagnose")
def optical_diagnose(
    run_id: Annotated[str, typer.Option("--run-id", help="Existing ingested run identifier.")],
) -> None:
    """Report Sentinel-2 invalid-pixel spectra and spatial distributions."""
    from sahayak.optical.diagnose import diagnose_optical

    try:
        result = diagnose_optical(run_id=run_id, settings=Settings())
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(
            f"Optical diagnosis failed: {type(error).__name__}: {error}",
            err=True,
        )
        raise typer.Exit(code=1) from error

    diagnosis = result["diagnosis"]
    typer.echo(f"Optical diagnosis for run {run_id}")
    for scene_name, zones in diagnosis["scl_histograms"].items():
        typer.echo(f"SCL histogram: {scene_name}")
        for zone_name, histogram in zones.items():
            typer.echo(
                f"  {zone_name}: counts={histogram['class_counts']}; "
                f"fractions={histogram['class_fraction']}"
            )
    typer.echo(
        "Bright corridor pixels (normalized dB04 >= 0.08): "
        f"{diagnosis['bright_change_candidate']['hand_le_40m_scl_histogram']['class_counts']}"
    )
    typer.echo(
        "Post-invalid/pre-valid corridor share: "
        f"{diagnosis['post_invalid_pre_valid_corridor_share']}"
    )
    typer.echo("Post SCL 8-11 reflectance means:")
    for band, means in diagnosis["post_scl_8_11_band_means"].items():
        typer.echo(f"  {band}: {means}")
    typer.echo(f"Post SCL 11 elevation/HAND: {diagnosis['post_scl_11_elevation_hand']}")
    typer.echo(
        "Post SCL 8-11 near-stream components: "
        f"{diagnosis['post_scl_8_11_near_stream_components']}"
    )
    typer.echo(f"Unavailable reflectance bands: {diagnosis['unavailable_reflectance_bands']}")
    typer.echo("Files:")
    for path in result["files"]:
        typer.echo(f"  {path.name}: {path.stat().st_size} bytes")


@app.command()
def serve(
    run_id: Annotated[str | None, typer.Option("--run-id", help="Optional run identifier.")] = None,
) -> None:
    """Start the API service (not implemented yet)."""
    if run_id is not None and not run_id.strip():
        raise typer.BadParameter("Run ID must not be empty.")
    typer.echo("serve: not implemented yet")


@app.command()
def doctor(
    check_auth: Annotated[
        bool,
        typer.Option("--check-auth", help="Request a Copernicus Data Space OAuth token."),
    ] = False,
    check_catalog: Annotated[
        bool,
        typer.Option(
            "--check-catalog",
            help="Compare authenticated and unauthenticated Sentinel Hub Catalog searches.",
        ),
    ] = False,
    show_properties: Annotated[
        bool,
        typer.Option(
            "--show-properties",
            help="Print property keys from the first authenticated Catalog item.",
        ),
    ] = False,
    list_acquisitions: Annotated[
        bool,
        typer.Option("--list-acquisitions", help="List Catalog acquisitions without processing."),
    ] = False,
    list_s2: Annotated[
        bool,
        typer.Option(
            "--list-s2",
            help="List Sentinel-2 L2A Catalog metadata without downloading pixels.",
        ),
    ] = False,
    bbox: Annotated[
        str,
        typer.Option("--bbox", help="Bounding box: minlon,minlat,maxlon,maxlat."),
    ] = "85.30,28.10,85.42,28.20",
    event_date: Annotated[
        str,
        typer.Option("--event-date", help="Event date (YYYY-MM-DD)."),
    ] = "2026-08-26",
    from_date: Annotated[
        str,
        typer.Option("--from", help="Inclusive Sentinel-2 Catalog start date (YYYY-MM-DD)."),
    ] = "2026-05-01",
    to_date: Annotated[
        str,
        typer.Option("--to", help="Inclusive Sentinel-2 Catalog end date (YYYY-MM-DD)."),
    ] = "2026-10-23",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Print the non-secret Catalog request body."),
    ] = False,
) -> None:
    """Check configured service settings without revealing any configured values."""
    settings = Settings()
    statuses = (
        ("CDSE_CLIENT_ID", settings.cdse_client_id),
        ("CDSE_CLIENT_SECRET", settings.cdse_client_secret),
        ("CDSE_USERNAME", settings.cdse_username),
        ("CDSE_PASSWORD", settings.cdse_password),
        ("OSM_PROVIDER", settings.osm_provider),
        ("OHSOME_API_KEY", settings.ohsome_api_key),
    )
    typer.echo(
        f"Environment file: {settings.env_file_path} "
        f"(exists: {'yes' if settings.env_file_path.is_file() else 'no'})"
    )
    for name, value in statuses:
        is_set = (
            bool(value.get_secret_value().strip())
            if isinstance(value, SecretStr)
            else bool(value.strip())
            if isinstance(value, str)
            else False
        )
        typer.echo(f"{name}: {'set' if is_set else 'EMPTY'}")

    if not check_auth and not check_catalog and not list_acquisitions and not list_s2:
        return

    import requests

    from sahayak.ingest.cdse_auth import (
        HTTP_MAX_ATTEMPTS,
        HTTP_TIMEOUT,
        AuthenticationError,
        CdseTokenClient,
        RequestFailure,
        post_with_retries,
        scrub_sensitive_text,
    )

    session = requests.Session()
    token: str | None = None
    authentication_ok = False
    token_client = CdseTokenClient(settings, session=session)
    try:
        token = token_client.get_token()
    except AuthenticationError as error:
        if error.status_code is not None:
            error_code = error.error_code or "not provided"
            auth_message = (
                f"{type(error).__name__}: HTTP {error.status_code}; error code: {error_code}"
            )
        else:
            auth_message = f"{type(error).__name__}: {error}"
        if check_auth or check_catalog or list_s2:
            typer.echo(f"Authentication: {auth_message}", err=True)
    else:
        authentication_ok = True
        if check_auth:
            typer.echo("Authentication: OK")

    catalog_features_found = False
    if check_catalog:
        from sahayak.ingest.s1 import CATALOG_URL, build_catalog_probe_request

        body = build_catalog_probe_request()
        if verbose:
            typer.echo(f"Catalog request body: {body}")
        secret_values = [
            value.get_secret_value()
            for value in (
                settings.cdse_client_id,
                settings.cdse_client_secret,
                settings.cdse_password,
                settings.ohsome_api_key,
            )
            if value is not None
        ]
        for label, authorization_token in (
            ("with Authorization", token),
            ("without Authorization", None),
        ):
            if label == "with Authorization" and authorization_token is None:
                typer.echo("Catalog with Authorization: skipped (token unavailable)")
                continue
            headers = {"Content-Type": "application/json"}
            if authorization_token is not None:
                headers["Authorization"] = "Bearer " + authorization_token
                secret_values.append(authorization_token)
            try:
                response, attempt = post_with_retries(
                    session,
                    CATALOG_URL,
                    service_name=f"Sentinel Hub Catalog probe {label}",
                    secrets=secret_values,
                    json=body,
                    headers=headers,
                    timeout=HTTP_TIMEOUT,
                )
            except RequestFailure as error:
                typer.echo(
                    f"Catalog {label}: {type(error).__name__}: {error}; works: no",
                    err=True,
                )
                continue
            if response.status_code != 200:
                typer.echo(
                    f"Catalog {label}: HTTP {response.status_code}; "
                    f"features 0; attempt {attempt}/{HTTP_MAX_ATTEMPTS}; works: no"
                )
                continue
            try:
                payload = response.json()
            except (ValueError, TypeError) as error:
                message = scrub_sensitive_text(str(error), secret_values)
                typer.echo(
                    f"Catalog {label}: {type(error).__name__}: {message}; works: no",
                    err=True,
                )
                continue
            features = payload.get("features") if isinstance(payload, dict) else None
            if not isinstance(features, list):
                typer.echo(
                    f"Catalog {label}: HTTP 200; InvalidResponse; "
                    f"attempt {attempt}/{HTTP_MAX_ATTEMPTS}; works: no",
                    err=True,
                )
                continue
            valid_features = [feature for feature in features if isinstance(feature, dict)]
            typer.echo(
                f"Catalog {label}: HTTP {response.status_code}; "
                f"features {len(valid_features)}; attempt {attempt}/{HTTP_MAX_ATTEMPTS}; "
                "works: yes"
            )
            if label == "with Authorization":
                catalog_features_found = bool(valid_features)
                if show_properties and valid_features:
                    properties = valid_features[0].get("properties")
                    keys = sorted(properties) if isinstance(properties, dict) else []
                    typer.echo(f"Catalog first item property keys: {keys}")
            for feature in valid_features:
                properties = feature.get("properties")
                if not isinstance(properties, dict):
                    properties = {}
                platform = scrub_sensitive_text(
                    str(properties.get("platform", "unknown")), secret_values
                )
                orbit_state = scrub_sensitive_text(
                    str(properties.get("sat:orbit_state", "unknown")), secret_values
                )
                relative_orbit = scrub_sensitive_text(
                    str(properties.get("sat:relative_orbit", "unknown")), secret_values
                )
                typer.echo(
                    f"  platform={platform}; sat:orbit_state={orbit_state}; "
                    f"sat:relative_orbit={relative_orbit}"
                )

    if check_auth and not authentication_ok:
        raise typer.Exit(code=1)
    if list_s2 and not authentication_ok:
        raise typer.Exit(code=1)
    if check_catalog and not catalog_features_found:
        if not list_acquisitions:
            raise typer.Exit(code=1)

    if list_acquisitions:
        import requests

        from sahayak.ingest.cache import RequestCache
        from sahayak.ingest.grid import GridSpec
        from sahayak.ingest.s1 import Sentinel1Error, SentinelHubClient, catalog_acquisitions

        try:
            parsed_bbox = _validate_bbox(bbox)
            parsed_event_date = _validate_event_date(event_date)
            grid = GridSpec.from_bbox(parsed_bbox)
            catalog_client = SentinelHubClient(
                settings,
                RequestCache(settings),
                session=requests.Session(),
            )
            exclusions: list[dict[str, str]] = []
            acquisitions = catalog_acquisitions(
                bbox=parsed_bbox,
                event_date=parsed_event_date,
                grid=grid,
                client=catalog_client,
                exclusions=exclusions,
            )
        except (Sentinel1Error, ValueError, OSError, RuntimeError) as error:
            typer.echo(
                f"Acquisition listing failed: {type(error).__name__}: {error}",
                err=True,
            )
            raise typer.Exit(code=1) from error

        ordered = sorted(
            acquisitions,
            key=lambda item: (
                item.relative_orbit,
                item.direction,
                item.acquired_at,
                item.platform,
            ),
        )
        typer.echo(
            "Platform | UTC datetime | Relative orbit | Direction | Mode | "
            "Polarisation | Bbox coverage | Event side"
        )
        group_counts: dict[tuple[int, str], list[int]] = {}
        for acquisition in ordered:
            before_event = acquisition.acquired_at.date() < parsed_event_date
            event_side = "before" if before_event else "after"
            counts = group_counts.setdefault(
                (acquisition.relative_orbit, acquisition.direction), [0, 0]
            )
            counts[0 if before_event else 1] += 1
            timestamp = (
                acquisition.acquired_at.astimezone(UTC)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
            typer.echo(
                f"{acquisition.platform} | {timestamp} | {acquisition.relative_orbit} | "
                f"{acquisition.direction} | {acquisition.mode} | "
                f"{'+'.join(acquisition.polarizations)} | "
                f"{acquisition.coverage_fraction * 100:.1f}% | {event_side}"
            )
        typer.echo("Per-track counts (before / after):")
        if group_counts:
            for (relative_orbit, direction), counts in sorted(group_counts.items()):
                typer.echo(f"  orbit {relative_orbit} {direction}: {counts[0]} / {counts[1]}")
        else:
            typer.echo("  no valid track groups found")
        typer.echo("Exclusions:")
        if exclusions:
            for excluded in exclusions:
                typer.echo(f"  {excluded['item_id']}: {excluded['reason']}")
        else:
            typer.echo("  none")

    if list_s2:
        from sahayak.ingest.cache import RequestCache
        from sahayak.ingest.s1 import Sentinel1Error, SentinelHubClient

        try:
            parsed_bbox = _validate_bbox(bbox)
            start = _validate_event_date(from_date)
            end = _validate_event_date(to_date)
            event = _validate_event_date(event_date)
            catalog = SentinelHubClient(
                settings,
                RequestCache(settings),
                session=session,
                token_client=token_client,
            )
            items: list[dict[str, object]] = []
            next_page: object | None = None
            seen_pages: set[str] = set()
            while True:
                request_body = build_s2_catalog_request(
                    parsed_bbox,
                    start,
                    end,
                    next_page=next_page,
                )
                payload = catalog.catalog_search(request_body)
                features = payload.get("features")
                if not isinstance(features, list):
                    raise Sentinel1Error("Sentinel-2 Catalog response has no features list.")
                for feature in features:
                    if isinstance(feature, dict):
                        items.append(parse_s2_catalog_item(feature))
                context = payload.get("context")
                next_value = context.get("next") if isinstance(context, dict) else None
                if next_value is None:
                    break
                page_key = str(next_value)
                if page_key in seen_pages:
                    raise Sentinel1Error("Sentinel-2 Catalog repeated a pagination token.")
                seen_pages.add(page_key)
                next_page = next_value
        except (Sentinel1Error, ValueError, OSError, RuntimeError) as error:
            typer.echo(
                f"Sentinel-2 Catalog listing failed: {type(error).__name__}: {error}",
                err=True,
            )
            raise typer.Exit(code=1) from error

        typer.echo("Acquisition date | Tile id | Relative orbit | eo:cloud_cover (%) | Event side")
        missing_cloud_cover = 0
        for item in sorted(items, key=lambda record: (record["acquired_at"], str(record["id"]))):
            acquired_at = item["acquired_at"]
            side = (
                "before"
                if acquired_at.date() < event
                else ("after" if acquired_at.date() > event else "event day")
            )
            cloud_cover = item["eo_cloud_cover"]
            if cloud_cover is None:
                missing_cloud_cover += 1
                cloud_label = "not provided"
            else:
                cloud_label = f"{cloud_cover:.2f}"
            tile_id = item["tile_id"] or "not provided"
            orbit = item["relative_orbit"]
            orbit_label = str(orbit) if orbit is not None else "not provided"
            typer.echo(
                f"{acquired_at.date().isoformat()} | {tile_id} | {orbit_label} | "
                f"{cloud_label} | {side}"
            )
        if not items:
            typer.echo("  no Sentinel-2 L2A acquisitions found")
        typer.echo("Cloud-cover property: eo:cloud_cover (percentage, as documented).")
        if missing_cloud_cover:
            typer.echo(
                f"eo:cloud_cover absent on {missing_cloud_cover} of {len(items)} returned items."
            )


@app.command()
def rules() -> None:
    """Print the project's hard data rules."""
    typer.echo(
        "\n".join(
            (
                "Allowed inputs: Sentinel-1/2, Copernicus DEM GLO-30, pre-event OSM via ohsome, "
                "Kuro Siwo, and Sen1Floods11.",
                "Forbidden inputs: Copernicus EMS, UNOSAT, published damage maps, and post-event "
                "OSM edits.",
                "data/reference/ is validation-only and must never be read by pipeline code.",
                "OSM snapshots must be strictly earlier than the event.",
                "Radar pairs must share relative orbit and pass direction.",
                "Report numbers must come only from facts.json; never invent or alter them.",
            )
        )
    )


if __name__ == "__main__":
    app()
