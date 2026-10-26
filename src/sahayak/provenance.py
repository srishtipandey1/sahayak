"""Input provenance records and guards for project data rules."""

import hashlib
import inspect
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from sahayak.config import Settings

FORBIDDEN_SOURCE_TERMS = (
    "emsr",
    "copernicus emergency",
    "cems",
    "unosat",
    "damage map",
    "rapid mapping",
)


class ForbiddenInputError(ValueError):
    """Raised when a data source or path is forbidden as an input."""


class RuleViolation(ValueError):
    """Raised when a project data rule is violated."""


def _resolved_normalized_path(path: str | Path) -> str:
    """Resolve a path and normalize case and separators for containment checks."""
    resolved = Path(path).expanduser().resolve()
    return os.path.normcase(os.path.normpath(str(resolved)))


def assert_allowed_input(path: str | Path, settings: Settings) -> None:
    """Reject files in or beneath the validation-only reference directory."""
    candidate = _resolved_normalized_path(path)
    reference = _resolved_normalized_path(settings.reference_dir)
    try:
        is_reference = os.path.commonpath((candidate, reference)) == reference
    except ValueError:
        is_reference = False
    if is_reference:
        raise ForbiddenInputError(f"Reference-only data cannot be used as an input: {path}")


def require_input(path: str | Path, settings: Settings) -> Path:
    """Validate an input path and return its fully resolved location."""
    assert_allowed_input(path, settings)
    return Path(path).expanduser().resolve()


def open_reference(
    path: str | Path,
    *,
    layer: str,
    settings: Settings | None = None,
    **kwargs: object,
) -> Any:
    """Read a validation-only vector layer, only from the validation package."""
    caller = inspect.currentframe()
    caller = caller.f_back if caller is not None else None
    caller_module = caller.f_globals.get("__name__", "") if caller is not None else ""
    if caller_module != "sahayak.validate" and not caller_module.startswith(
        "sahayak.validate."
    ):
        raise ForbiddenInputError(
            "Validation reference files may only be opened from sahayak.validate."
        )

    from pyogrio import read_dataframe

    active_settings = settings or Settings()
    candidate = Path(path).expanduser().resolve()
    reference = active_settings.reference_dir.resolve()
    try:
        is_reference = os.path.commonpath((str(candidate), str(reference))) == str(reference)
    except ValueError:
        is_reference = False
    if not is_reference:
        raise ForbiddenInputError(
            f"Validation reference path must be inside {active_settings.reference_dir}."
        )
    return read_dataframe(candidate, layer=layer, **kwargs)


def assert_allowed_source(source: str) -> None:
    """Reject known prohibited source descriptions."""
    lowered_source = source.lower()
    if any(term in lowered_source for term in FORBIDDEN_SOURCE_TERMS):
        raise ForbiddenInputError(f"Forbidden data source: {source}")


def _as_date(value: date | str) -> date:
    """Parse a date value or raise a project-rule error."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise RuleViolation(f"Invalid date {value!r}; expected YYYY-MM-DD.") from error


def assert_osm_snapshot_valid(snapshot_date: date | str, event_date: date | str) -> None:
    """Require an OSM snapshot to be strictly earlier than the event."""
    snapshot = _as_date(snapshot_date)
    event = _as_date(event_date)
    if snapshot >= event:
        raise RuleViolation(
            f"OSM snapshot date {snapshot.isoformat()} must be earlier than "
            f"event date {event.isoformat()}."
        )


class SceneTrack(BaseModel):
    """The orbit metadata needed to compare a pair of radar scenes."""

    relative_orbit: int
    pass_direction: Literal["ASCENDING", "DESCENDING"]


def assert_same_orbit_track(scene_a: SceneTrack, scene_b: SceneTrack) -> None:
    """Require two radar scenes to share relative orbit and pass direction."""
    if scene_a.relative_orbit != scene_b.relative_orbit:
        raise RuleViolation("Radar scenes must share the same relative orbit.")
    if scene_a.pass_direction != scene_b.pass_direction:
        raise RuleViolation("Radar scenes must share the same pass direction.")


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class InputRecord(BaseModel):
    """A provenance record for an input, validation reference, or generated output."""

    name: str
    role: Literal["input", "reference", "output"]
    source: str
    license: str
    retrieved_utc: datetime
    sha256: str | None = None


class RunManifest(BaseModel):
    """Provenance for a mapping run and its input files."""

    run_id: str
    created_utc: datetime
    bbox: tuple[float, float, float, float]
    event_date: date
    inputs: list[InputRecord] = Field(default_factory=list)

    def add_input(
        self,
        *,
        name: str,
        role: Literal["input", "reference", "output"],
        source: str,
        license: str,
        retrieved_utc: datetime,
        path: str | Path | None = None,
        settings: Settings | None = None,
    ) -> InputRecord:
        """Validate and append a provenance record, hashing a supplied file."""
        assert_allowed_source(source)
        assert_allowed_source(name)
        digest = None
        if path is not None:
            if role in {"input", "output"}:
                validated_path = require_input(path, settings or Settings())
            else:
                validated_path = Path(path).expanduser().resolve()
            digest = sha256_file(validated_path)
        record = InputRecord(
            name=name,
            role=role,
            source=source,
            license=license,
            retrieved_utc=retrieved_utc,
            sha256=digest,
        )
        self.inputs.append(record)
        return record

    def write(self, path: str | Path) -> None:
        """Write this manifest as indented JSON, creating its parent directory."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "RunManifest":
        """Read and validate a manifest from JSON."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
