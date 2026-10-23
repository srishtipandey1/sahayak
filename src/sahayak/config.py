"""Application configuration loaded from the environment and optional .env file."""

import os
from pathlib import Path
from typing import Any

from pydantic import PrivateAttr, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_UNSET = object()


def _project_root(root_override: Path | str | None = None) -> Path:
    """Find the project root, honoring an optional environment override."""
    if root_override:
        return Path(root_override).expanduser().resolve()

    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(f"Could not find pyproject.toml above {__file__}.")


def _resolve_project_path(path: Path, root_override: Path | None) -> Path:
    """Resolve relative paths from the project root and preserve absolute paths."""
    if path.is_absolute():
        return path
    return (_project_root(root_override) / path).resolve()


class Settings(BaseSettings):
    """Settings for credentials, external services, and local data paths."""

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    _env_file_path: Path = PrivateAttr()

    sahayak_root: Path | None = None
    cdse_username: str | None = None
    cdse_password: SecretStr | None = None
    cdse_client_id: SecretStr | None = None
    cdse_client_secret: SecretStr | None = None
    llm_provider: str | None = None
    llm_api_key: SecretStr | None = None
    osm_provider: str = "overpass"
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    ohsome_url: str = "https://api.ohsome.org/v1"
    ohsome_api_key: SecretStr | None = None
    ohsome_api_version: str = "v2"
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    def __init__(self, **values: Any) -> None:
        """Load the project-root dotenv file unless a caller explicitly overrides it."""
        env_file_override = values.pop("_env_file", _UNSET)
        root_override = values.get("sahayak_root") or os.environ.get("SAHAYAK_ROOT")
        project_root = _project_root(root_override)
        env_file_path = project_root / ".env"
        env_file = env_file_path if env_file_override is _UNSET else env_file_override
        super().__init__(_env_file=env_file, **values)
        self._env_file_path = env_file_path

    @field_validator("sahayak_root", mode="before")
    @classmethod
    def _empty_root_is_unset(cls, value: Path | str | None) -> Path | str | None:
        """Treat an empty optional root setting as unset."""
        return value or None

    def __repr__(self) -> str:
        """Return a settings representation that never exposes credential values."""
        return "Settings(<sensitive values redacted>)"

    def __str__(self) -> str:
        """Return a safe string representation."""
        return repr(self)

    @property
    def resolved_data_dir(self) -> Path:
        """Return the data directory, resolving relative paths against the project root."""
        return _resolve_project_path(self.data_dir.expanduser(), self.sahayak_root)

    @property
    def resolved_runs_dir(self) -> Path:
        """Return the run directory, resolving relative paths against the project root."""
        return _resolve_project_path(self.runs_dir.expanduser(), self.sahayak_root)

    @property
    def reference_dir(self) -> Path:
        """Return the validation-only reference data directory."""
        return self.resolved_data_dir / "reference"

    @property
    def env_file_path(self) -> Path:
        """Return the absolute project dotenv path selected for this settings instance."""
        return self._env_file_path
