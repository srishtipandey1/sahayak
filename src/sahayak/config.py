"""Application configuration loaded from the environment and optional .env file."""

from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root(root_override: Path | None = None) -> Path:
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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sahayak_root: Path | None = None
    cdse_username: str | None = None
    cdse_password: SecretStr | None = None
    cdse_client_id: SecretStr | None = None
    cdse_client_secret: SecretStr | None = None
    llm_provider: str | None = None
    llm_api_key: SecretStr | None = None
    ohsome_url: str = "https://api.ohsome.org/v1"
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

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
