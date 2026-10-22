"""Application configuration loaded from the environment and optional .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for credentials, external services, and local data paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    cdse_username: str | None = None
    cdse_password: str | None = None
    llm_provider: str | None = None
    llm_api_key: str | None = None
    ohsome_url: str = "https://api.ohsome.org/v1"
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    @property
    def resolved_data_dir(self) -> Path:
        """Return the absolute data directory with user and relative paths resolved."""
        return self.data_dir.expanduser().resolve()

    @property
    def resolved_runs_dir(self) -> Path:
        """Return the absolute run-output directory."""
        return self.runs_dir.expanduser().resolve()

    @property
    def reference_dir(self) -> Path:
        """Return the validation-only reference data directory."""
        return (self.resolved_data_dir / "reference").resolve()
