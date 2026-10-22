"""Command-line interface for Sahayak."""

import re
from datetime import date
from typing import Annotated

import typer

from sahayak import __version__
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


for _command_name in ("ingest", "flood", "damage", "access", "flowpath", "report", "run"):
    app.command(name=_command_name)(_science_command(_command_name))


@app.command()
def serve(
    run_id: Annotated[str | None, typer.Option("--run-id", help="Optional run identifier.")] = None,
) -> None:
    """Start the API service (not implemented yet)."""
    if run_id is not None and not run_id.strip():
        raise typer.BadParameter("Run ID must not be empty.")
    typer.echo("serve: not implemented yet")


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
