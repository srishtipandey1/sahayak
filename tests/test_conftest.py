"""Tests for shared pytest helpers."""

from conftest import plain


def test_plain_strips_ansi_csi_sequences() -> None:
    sample = "\x1b[1;33mUsage:\x1b[0m command \x1b[38;5;196m--option\x1b[0m"

    assert plain(sample) == "Usage: command --option"
