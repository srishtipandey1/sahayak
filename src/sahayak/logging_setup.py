"""Central logging configuration."""

import logging
import os

_configured = False


def configure_logging() -> None:
    """Configure root logging once, using the LOG_LEVEL environment variable."""
    global _configured
    if _configured:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        message = f"Invalid LOG_LEVEL {level_name!r}; use DEBUG, INFO, WARNING, ERROR, or CRITICAL."
        raise ValueError(message)

    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _configured = True
