"""Shared pytest setup for headless, color-independent test output."""

import os
import re

os.environ.setdefault("MPLBACKEND", "Agg")

_ANSI_CSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain(text: str) -> str:
    """Strip ANSI CSI control sequences from terminal output."""
    return _ANSI_CSI_PATTERN.sub("", text)
