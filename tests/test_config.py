"""Tests for application settings."""

from pathlib import Path

from sahayak.config import Settings


def test_settings_defaults_and_resolved_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None)

    assert settings.ohsome_url == "https://api.ohsome.org/v1"
    assert settings.resolved_data_dir == (tmp_path / "data").resolve()
    assert settings.resolved_runs_dir == (tmp_path / "runs").resolve()
    assert settings.reference_dir == (tmp_path / "data" / "reference").resolve()


def test_settings_load_environment_values(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "custom-data"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "custom-runs"))
    monkeypatch.setenv("CDSE_USERNAME", "test-user")

    settings = Settings(_env_file=None)

    assert settings.cdse_username == "test-user"
    assert settings.resolved_data_dir == (tmp_path / "custom-data").resolve()
    assert settings.resolved_runs_dir == (tmp_path / "custom-runs").resolve()
