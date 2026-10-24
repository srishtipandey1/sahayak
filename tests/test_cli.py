"""Tests for command-line argument validation and help."""

from conftest import plain
from typer.testing import CliRunner

import sahayak.cli as cli
from sahayak import __version__
from sahayak.cli import app
from sahayak.config import Settings

runner = CliRunner(env={"NO_COLOR": "1", "COLUMNS": "200"})


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "flood" in plain(result.output)
    assert "rules" in plain(result.output)


def test_version_option() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert plain(result.output).strip() == __version__


def test_flood_requires_run_id() -> None:
    result = runner.invoke(
        app,
        ["flood"],
    )

    assert result.exit_code != 0
    assert "--run-id" in plain(result.output)


def test_flood_command_reports_mapping_facts(tmp_path, monkeypatch) -> None:
    import sahayak.flood.run as flood_module

    monkeypatch.setattr(
        flood_module,
        "run_flood",
        lambda **_kwargs: {
            "flood_dir": tmp_path / "flood",
            "facts": {
                "provenance": {
                    "thresholds_used": {"vv_otsu_water_threshold": {"value": -19.5}}
                },
                "valid_pixel_fraction": {"value": 0.98},
                "permanent_water_area_before_restriction_ha": {"value": 5.0},
                "permanent_water_area_after_restriction_ha": {"value": 3.0},
                "areas_ha_by_class": {
                    "none": {"value": 12.0},
                    "new_water": {"value": 1.0},
                    "disturbed_surface": {"value": 2.0},
                    "permanent_water": {"value": 3.0},
                    "dark_low_confidence": {"value": 4.0},
                },
                "sensitivity": [
                    {
                        "max_hand": {"value": 20.0},
                        "area_ha_by_class": {
                            "new_water": {"value": 1.0},
                            "disturbed_surface": {"value": 2.0},
                            "permanent_water": {"value": 3.0},
                            "dark_low_confidence": {"value": 4.0},
                        },
                    }
                ],
            },
        },
    )
    result = runner.invoke(app, ["flood", "--run-id", "test"])

    assert result.exit_code == 0
    assert "VV Otsu water threshold: -19.50 dB" in plain(result.output)
    assert "Valid pixel fraction: 98.0000%" in plain(result.output)
    assert "new_water" in plain(result.output)
    assert "20.0 m: new water 1.00" in plain(result.output)


def test_ingest_defaults_to_all_stages(tmp_path, monkeypatch) -> None:
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
    assert captured["skip"] == set()
    assert captured["s1_baseline_count"] == 3


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
    assert f"Environment file: {settings.env_file_path}" in plain(result.output)
    assert "exists:" in plain(result.output)
    for name in (
        "CDSE_CLIENT_ID",
        "CDSE_CLIENT_SECRET",
        "CDSE_USERNAME",
        "CDSE_PASSWORD",
        "OSM_PROVIDER",
        "OHSOME_API_KEY",
    ):
        assert f"{name}: set" in plain(result.output)
    assert all(
        secret not in plain(result.output)
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
    assert "HTTP 401; error code: invalid_client" in plain(result.output)
    assert "private-client-id" not in plain(result.output)
    assert "private-client-secret" not in plain(result.output)


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
    assert "Authentication: OK" in plain(result.output)
    assert "private-access-token" not in plain(result.output)


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

    result = runner.invoke(
        app,
        ["doctor", "--check-catalog", "--show-properties", "--verbose"],
    )

    assert result.exit_code == 0
    assert "Catalog request body:" in plain(result.output)
    assert "Catalog with Authorization: HTTP 200; features 1;" in plain(result.output)
    assert "Catalog without Authorization: HTTP 401; features 0;" in plain(result.output)
    assert "platform=sentinel-1d" in plain(result.output)
    assert "sat:relative_orbit=-1" in plain(result.output)
    assert (
        "Catalog first item property keys: "
        "['platform', 'sat:orbit_state', 'sat:relative_orbit']"
    ) in plain(result.output)
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
        assert secret not in plain(result.output)


def test_doctor_list_acquisitions_prints_sorted_groups_and_exclusions(monkeypatch) -> None:
    from shapely.geometry import box, mapping

    import sahayak.ingest.cdse_auth as cdse_auth
    import sahayak.ingest.s1 as s1_module

    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    class SuccessfulTokenClient:
        def __init__(self, configured_settings, *, session):
            assert configured_settings is settings
            assert session is not None

        def get_token(self):
            return "private-access-token"

    monkeypatch.setattr(cdse_auth, "CdseTokenClient", SuccessfulTokenClient)

    items = [
        {
            "id": (
                "S1A_IW_GRDH_1SDV_20260820T050000_20260820T050025_000045_PRE"
            ),
            "geometry": mapping(box(85.3, 28.1, 85.42, 28.2)),
            "properties": {
                "datetime": "2026-08-20T05:00:00Z",
                "platform": "sentinel-1a",
                "sat:relative_orbit": 45,
                "sat:orbit_state": "ascending",
                "sar:instrument_mode": "IW",
                "s1:polarization": "DV",
            },
        },
        {
            "id": (
                "S1A_IW_GRDH_1SDV_20260826T050000_20260826T050025_000045_POST"
            ),
            "geometry": mapping(box(85.3, 28.1, 85.42, 28.2)),
            "properties": {
                "datetime": "2026-08-26T05:00:00Z",
                "platform": "sentinel-1a",
                "sat:relative_orbit": 45,
                "sat:orbit_state": "ascending",
                "sar:instrument_mode": "IW",
                "s1:polarization": "DV",
            },
        },
        {
            "id": "malformed-item",
            "properties": {"platform": "sentinel-1a"},
        },
    ]

    class CatalogClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def catalog_search(self, body):
            assert body["datetime"] == "2026-06-27T00:00:00Z/2026-10-10T23:59:59Z"
            return {"features": items, "context": {}}

        def resolve_relative_orbit(self, *args):
            raise AssertionError("No orbit recovery should be needed.")

        def process(self, *args):
            raise AssertionError("Listing must not call the Process API.")

    monkeypatch.setattr(s1_module, "SentinelHubClient", CatalogClient)

    result = runner.invoke(app, ["doctor", "--list-acquisitions"])

    assert result.exit_code == 0
    assert "sentinel-1a | 2026-08-20T05:00:00Z | 45 | ASCENDING | IW" in plain(result.output)
    assert "sentinel-1a | 2026-08-26T05:00:00Z | 45 | ASCENDING | IW" in plain(result.output)
    assert "orbit 45 ASCENDING: 1 / 1" in plain(result.output)
    assert "malformed-item" in plain(result.output)
    assert "is not a supported Sentinel-1 ID" in plain(result.output)


def test_doctor_list_s2_prints_catalog_metadata_without_pixel_requests(monkeypatch) -> None:
    import sahayak.ingest.cdse_auth as cdse_auth
    import sahayak.ingest.s1 as s1_module

    settings = Settings(_env_file=None)
    monkeypatch.setattr(cli, "Settings", lambda: settings)

    class SuccessfulTokenClient:
        def __init__(self, configured_settings, *, session):
            assert configured_settings is settings
            assert session is not None

        def get_token(self):
            return "private-access-token"

    monkeypatch.setattr(cdse_auth, "CdseTokenClient", SuccessfulTokenClient)

    class CatalogClient:
        def __init__(self, configured_settings, cache, *, session, token_client):
            assert configured_settings is settings
            assert cache is not None
            assert session is not None
            assert token_client is not None
            self.requests = []

        def catalog_search(self, body):
            self.requests.append(body)
            return {
                "features": [
                    {
                        "id": "S2A_MSIL2A_20260815T100000_T45RTM_N0000",
                        "properties": {
                            "datetime": "2026-08-15T10:00:00Z",
                            "s2:tile_id": "45RTM",
                            "sat:relative_orbit": 85,
                            "eo:cloud_cover": 42.5,
                        },
                    },
                    {
                        "id": "S2B_MSIL2A_20260901T100000_T45RTM_N0000",
                        "properties": {"datetime": "2026-09-01T10:00:00Z"},
                    },
                ],
                "context": {},
            }

    catalog_clients = []

    def make_catalog(*args, **kwargs):
        client = CatalogClient(*args, **kwargs)
        catalog_clients.append(client)
        return client

    monkeypatch.setattr(s1_module, "SentinelHubClient", make_catalog)
    result = runner.invoke(
        app,
        [
            "doctor",
            "--list-s2",
            "--bbox",
            "85.30,28.10,85.42,28.20",
            "--from",
            "2026-05-01",
            "--to",
            "2026-10-23",
        ],
    )

    assert result.exit_code == 0, plain(result.output)
    assert (
        "Acquisition date | Tile id | Relative orbit | eo:cloud_cover (%) | Event side"
        in plain(result.output)
    )
    assert "2026-08-15 | 45RTM | 85 | 42.50 | before" in plain(result.output)
    assert "2026-09-01 | 45RTM | not provided | not provided | after" in plain(result.output)
    assert "eo:cloud_cover absent on 1 of 2 returned items." in plain(result.output)
    assert "eo:cloud_cover (percentage, as documented)" in plain(result.output)
    assert catalog_clients[0].requests == [
        {
            "bbox": [85.3, 28.1, 85.42, 28.2],
            "datetime": "2026-05-01T00:00:00Z/2026-10-23T23:59:59Z",
            "collections": ["sentinel-2-l2a"],
            "limit": 100,
        }
    ]
