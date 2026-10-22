"""Tests for command-line argument validation and help."""

from typer.testing import CliRunner

from sahayak import __version__
from sahayak.cli import app

runner = CliRunner()


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "flood" in result.output
    assert "rules" in result.output


def test_version_option() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.strip() == __version__


def test_flood_rejects_bad_bbox() -> None:
    result = runner.invoke(
        app,
        ["flood", "--bbox", "invalid", "--event-date", "2026-08-26"],
    )

    assert result.exit_code != 0
    assert "Expected minlon" in result.output


def test_flood_rejects_bad_date() -> None:
    result = runner.invoke(
        app,
        ["flood", "--bbox", "85.0,28.0,85.5,28.5", "--event-date", "2026-02-30"],
    )

    assert result.exit_code != 0
    assert "Event date" in result.output


def test_flood_placeholder_accepts_valid_options() -> None:
    result = runner.invoke(
        app,
        ["flood", "--bbox", "85.0,28.0,85.5,28.5", "--event-date", "2026-08-26"],
    )

    assert result.exit_code == 0
    assert "not implemented yet" in result.output
