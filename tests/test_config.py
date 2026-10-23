"""Tests for application settings."""

from pathlib import Path

from pydantic import SecretStr

from sahayak.config import Settings


def test_settings_defaults_and_resolved_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None)
    project_root = Path(__file__).resolve().parents[1]

    assert settings.ohsome_url == "https://api.ohsome.org/v1"
    assert settings.resolved_data_dir == project_root / "data"
    assert settings.resolved_runs_dir == project_root / "runs"
    assert settings.reference_dir == project_root / "data" / "reference"


def test_settings_load_environment_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "custom-runs"))
    monkeypatch.setenv("CDSE_USERNAME", "test-user")

    settings = Settings(_env_file=None)

    assert settings.cdse_username == "test-user"
    assert settings.resolved_data_dir == (tmp_path / "custom-data").resolve()
    assert settings.resolved_runs_dir == (tmp_path / "custom-runs").resolve()


def test_settings_absolute_paths_are_preserved(tmp_path: Path) -> None:
    data_path = tmp_path / "custom-data" / ".." / "data"
    runs_path = tmp_path / "custom-runs"
    settings = Settings(data_dir=data_path, runs_dir=runs_path, _env_file=None)

    assert settings.resolved_data_dir == data_path
    assert settings.resolved_runs_dir == runs_path
    assert settings.reference_dir == data_path / "reference"


def test_settings_secret_values_are_redacted() -> None:
    secrets = ("private-password", "private-client-id", "private-client-secret", "private-api-key")
    settings = Settings(
        cdse_password=SecretStr(secrets[0]),
        cdse_client_id=SecretStr(secrets[1]),
        cdse_client_secret=SecretStr(secrets[2]),
        llm_api_key=SecretStr(secrets[3]),
        _env_file=None,
    )

    assert all(secret not in repr(settings) for secret in secrets)
    assert all(secret not in str(settings) for secret in secrets)


def test_sahayak_root_environment_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SAHAYAK_ROOT", str(tmp_path))
    settings = Settings(_env_file=None)

    assert settings.resolved_data_dir == tmp_path / "data"
    assert settings.resolved_runs_dir == tmp_path / "runs"
    assert settings.reference_dir == tmp_path / "data" / "reference"


def test_settings_load_project_env_independent_of_working_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    working_directory = tmp_path / "elsewhere"
    project_root.mkdir()
    working_directory.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='synthetic'\n", encoding="utf-8")
    dotenv = project_root / ".env"
    dotenv.write_text(
        "\ufeffCDSE_CLIENT_ID=dotenv-client-id\n"
        "CDSE_CLIENT_SECRET=dotenv-client-secret\n"
        "OSM_PROVIDER=ohsome\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(working_directory)
    monkeypatch.delenv("SAHAYAK_ROOT", raising=False)
    monkeypatch.delenv("CDSE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("OSM_PROVIDER", raising=False)
    monkeypatch.setenv("CDSE_CLIENT_ID", "environment-client-id")

    settings = Settings(sahayak_root=project_root)

    assert settings.env_file_path == dotenv.resolve()
    assert settings.env_file_path.is_absolute()
    assert settings.cdse_client_id == SecretStr("environment-client-id")
    assert settings.cdse_client_secret == SecretStr("dotenv-client-secret")
    assert settings.osm_provider == "ohsome"
    assert "dotenv-client-secret" not in repr(settings)
    assert "environment-client-id" not in str(settings)
