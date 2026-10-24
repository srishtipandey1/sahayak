"""Best-effort atomic writes for nonessential PNG visualizations."""

import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

LOGGER = logging.getLogger(__name__)
PNG_WRITE_ATTEMPTS = 3
PNG_WRITE_RETRY_SECONDS = 1.0


def write_png_atomically(
    path: Path,
    writer: Callable[[Path], object],
    *,
    attempts: int = PNG_WRITE_ATTEMPTS,
    retry_seconds: float = PNG_WRITE_RETRY_SECONDS,
) -> bool:
    """Write a PNG beside its destination and atomically replace it; failures are nonfatal."""
    if attempts < 1:
        raise ValueError("PNG write attempts must be at least one.")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as error:
        LOGGER.warning(
            "Quicklook %s could not prepare its directory: %s: %s",
            path.name,
            type(error).__name__,
            error,
        )
        return False

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.stem}.",
                suffix=path.suffix or ".png",
                dir=path.parent,
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            writer(temporary_path)
            os.replace(temporary_path, path)
            return True
        except Exception as error:
            last_error = error
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    LOGGER.warning(
                        "Could not remove partial quicklook temporary file %s: %s: %s",
                        temporary_path.name,
                        type(cleanup_error).__name__,
                        cleanup_error,
                    )
            if attempt < attempts:
                time.sleep(retry_seconds)

    if last_error is not None:
        LOGGER.warning(
            "Quicklook %s failed after %d attempts: %s: %s",
            path.name,
            attempts,
            type(last_error).__name__,
            last_error,
        )
    return False
