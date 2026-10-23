"""Tests for command-line argument validation and help."""

from typer.testing import CliRunner

import sahayak.cli as cli
from sahayak import __version__
from sahayak.cli import app
from sahayak.config import Settings

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


def test_ingest_defaults_to_s1_and_skips_only_s2(tmp_path, monkeypatch) -> None:
    import sahayak.ingest.run as run_module

    captured = {}

    def fake_run_ingest(**kwargs):
        captured.update(kwargs)
        output_dir = tmp_path / "runs" / "test"
        output_dir.mkdir(parents=True)
        from sahayak.ingest.grid import GridSpec

        return {
            "run_id": "test",
            "output_dir": output_dir,
            "grid": GridSpec.from_bbox((85.0, 28.0, 85.5, 28.5)),
            "dem": None,
            "osm": None,
            "s1": None,
        }

    monkeypatch.setattr(run_module, "run_ingest", fake_run_ingest)
    result = runner.invoke(
        app,
        [
            "ingest",
            "--bbox",
            "85.0,28.0,85.5,28.5",
            "--event-date",
            "2026-08-26",
            "--run-id",
            "test",
        ],
    )

    assert result.exit_code == 0
    assert captured["skip"] == {"s2"}
    assert captured["s1_baseline_count"] == 3
    assert "Sentinel-2 optical ingestion is not implemented yet" in result.output


def test_doctor_reports_status_only_and_hides_secrets(monkeypatch) -> None:
    settings = Settings(
        cdse_client_id="private-client-id",
        cdse_client_secret="private-client-secret",
        cdse_username="private-username",
        cdse_password="private-password",
        osm_provider="ohsome",
        ohsome_api_key="private-ohsome-key",
        _env_file=None,
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert f"Environment file: {settings.env_file_path}" in result.output
    assert "exists:" in result.output
    for name in (
        "CDSE_CLIENT_ID",
        "CDSE_CLIENT_SECRET",
        "CDSE_USERNAME",
        "CDSE_PASSWORD",
        "OSM_PROVIDER",
        "OHSOME_API_KEY",
    ):
        assert f"{name}: set" in result.output
    assert all(
        secret not in result.output
        for secret in (
            "private-client-id",
            "private-client-secret",
            "private-username",
            "private-password",
            "private-ohsome-key",
        )
    )


def test_doctor_check_auth_reports_status_and_error_code_without_body(monkeypatch) -> None:
    import sahayak.ingest.cdse_auth as cdse_auth

    settings = Settings(
        cdse_client_id="private-client-id",
        cdse_client_secret="private-client-secret",
        _env_file=None,
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    class FailingTokenClient:
        def __init__(self, configured_settings: Settings, *, session) -> None:
            assert configured_settings is settings
            assert session is not None

        def get_token(self) -> str:
            raise cdse_auth.AuthenticationError(
                "Copernicus Data Space token request failed (HTTP 401).",
                status_code=401,
                error_code="invalid_client",
            )

    monkeypatch.setattr(cdse_auth, "CdseTokenClient", FailingTokenClient)

    result = runner.invoke(app, ["doctor", "--check-auth"])

    assert result.exit_code == 1
    assert "HTTP 401; error code: invalid_client" in result.output
    assert "private-client-id" not in result.output
    assert "private-client-secret" not in result.output


def test_doctor_check_auth_success_does_not_print_token(monkeypatch) -> None:
    import sahayak.ingest.cdse_auth as cdse_auth

    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    class SuccessfulTokenClient:
        def __init__(self, configured_settings: Settings, *, session) -> None:
            assert configured_settings is settings
            assert session is not None

        def get_token(self) -> str:
            return "private-access-token"

    monkeypatch.setattr(cdse_auth, "CdseTokenClient", SuccessfulTokenClient)

    result = runner.invoke(app, ["doctor", "--check-auth"])

    assert result.exit_code == 0
    assert "Authentication: OK" in result.output
    assert "private-access-token" not in result.output


def test_doctor_check_catalog_reports_both_auth_modes_without_secrets(monkeypatch) -> None:
    import requests

    import sahayak.ingest.cdse_auth as cdse_auth

    settings = Settings(
        cdse_client_id="private-client-id",
        cdse_client_secret="private-client-secret",
        cdse_password="private-password",
        _env_file=None,
    )
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    class CatalogResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.headers = {}
            self._payload = payload

        def json(self):
            return self._payload

    class CatalogSession:
        def __init__(self):
            self.calls = []
            self.responses = iter(
                [
                    CatalogResponse(
                        200,
                        {
                            "features": [
                                {
                                    "properties": {
                                        "platform": "sentinel-1d",
                                        "sat:orbit_state": "ASCENDING",
                                        "sat:relative_orbit": -1,
                                    }
                                }
                            ]
                        },
                    ),
                    CatalogResponse(401, {}),
                ]
            )

        def post(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return next(self.responses)

    session = CatalogSession()
    expected_session = session
    monkeypatch.setattr(requests, "Session", lambda: session)

    class SuccessfulTokenClient:
        def __init__(self, configured_settings, *, session):
            assert configured_settings is settings
            assert session is expected_session

        def get_token(self):
            return "private-access-token"

    monkeypatch.setattr(cdse_auth, "CdseTokenClient", SuccessfulTokenClient)

    result = runner.invoke(app, ["doctor", "--check-catalog", "--verbose"])

    assert result.exit_code == 0
    assert "Catalog request body:" in result.output
    assert "Catalog with Authorization: HTTP 200; features 1;" in result.output
    assert "Catalog without Authorization: HTTP 401; features 0;" in result.output
    assert "platform=sentinel-1d" in result.output
    assert "sat:relative_orbit=-1" in result.output
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer private-access-token"
    assert "Authorization" not in session.calls[1][1]["headers"]
    assert session.calls[0][1]["timeout"] == (15, 120)
    assert session.calls[0][1]["json"] == session.calls[1][1]["json"]
    for secret in (
        "private-client-id",
        "private-client-secret",
        "private-password",
        "private-access-token",
    ):
        assert secret not in result.output
