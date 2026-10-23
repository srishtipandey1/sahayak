"""Command-line interface for Sahayak."""

import re
from datetime import UTC, date
from typing import Annotated

import typer
from pydantic import SecretStr

from sahayak import __version__
from sahayak.config import Settings
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


for _command_name in ("flood", "damage", "access", "flowpath", "report", "run"):
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
    ] = "s2",
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
    """Ingest Sentinel-1, the common grid, Copernicus DEM, and pre-event OSM."""
    parsed_bbox = _validate_bbox(bbox)
    parsed_event_date = _validate_event_date(event_date)
    parsed_snapshot = None
    if osm_snapshot is not None:
        try:
            parsed_snapshot = _validate_event_date(osm_snapshot)
        except typer.BadParameter as error:
            raise typer.BadParameter(f"Invalid --osm-snapshot: {error.message}") from error
    skipped_stages = {stage.strip().lower() for stage in skip.split(",") if stage.strip()}
    if "s2" in skipped_stages:
        typer.echo("Sentinel-2 optical ingestion is not implemented yet; skipping s2.")
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
    bbox: Annotated[
        str,
        typer.Option("--bbox", help="Bounding box: minlon,minlat,maxlon,maxlat."),
    ] = "85.30,28.10,85.42,28.20",
    event_date: Annotated[
        str,
        typer.Option("--event-date", help="Event date (YYYY-MM-DD)."),
    ] = "2026-08-26",
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

    if not check_auth and not check_catalog and not list_acquisitions:
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
    try:
        token = CdseTokenClient(settings, session=session).get_token()
    except AuthenticationError as error:
        if error.status_code is not None:
            error_code = error.error_code or "not provided"
            auth_message = (
                f"{type(error).__name__}: HTTP {error.status_code}; error code: {error_code}"
            )
        else:
            auth_message = f"{type(error).__name__}: {error}"
        if check_auth or check_catalog:
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
            timestamp = acquisition.acquired_at.astimezone(UTC).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
            typer.echo(
                f"{acquisition.platform} | {timestamp} | {acquisition.relative_orbit} | "
                f"{acquisition.direction} | {acquisition.mode} | "
                f"{'+'.join(acquisition.polarizations)} | "
                f"{acquisition.coverage_fraction * 100:.1f}% | {event_side}"
            )
        typer.echo("Per-track counts (before / after):")
        if group_counts:
            for (relative_orbit, direction), counts in sorted(group_counts.items()):
                typer.echo(
                    f"  orbit {relative_orbit} {direction}: "
                    f"{counts[0]} / {counts[1]}"
                )
        else:
            typer.echo("  no valid track groups found")
        typer.echo("Exclusions:")
        if exclusions:
            for excluded in exclusions:
                typer.echo(f"  {excluded['item_id']}: {excluded['reason']}")
        else:
            typer.echo("  none")


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
