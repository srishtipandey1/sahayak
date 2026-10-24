"""Tests for best-effort, atomic quicklook writes."""

import logging
from pathlib import Path

from sahayak.quicklook import write_png_atomically


def test_failed_png_writer_retries_and_preserves_destination(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    destination = tmp_path / "preview.png"
    destination.write_bytes(b"previous image")
    attempts: list[Path] = []
    replacements = 0

    def writer(path: Path) -> None:
        attempts.append(path)
        path.write_bytes(b"partial image")

    def fail_replace(source: Path, target: Path) -> None:
        nonlocal replacements
        replacements += 1
        raise OSError("viewer lock")

    monkeypatch.setattr("sahayak.quicklook.os.replace", fail_replace)
    monkeypatch.setattr("sahayak.quicklook.time.sleep", lambda _seconds: None)
    caplog.set_level(logging.WARNING, logger="sahayak.quicklook")

    assert not write_png_atomically(destination, writer)
    assert len(attempts) == replacements == 3
    assert destination.read_bytes() == b"previous image"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["preview.png"]
    assert "OSError: viewer lock" in caplog.text


def test_png_writer_replaces_destination_atomically(tmp_path: Path) -> None:
    destination = tmp_path / "preview.png"
    destination.write_bytes(b"old")

    def writer(path: Path) -> None:
        path.write_bytes(b"new")

    assert write_png_atomically(destination, writer)
    assert destination.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [destination]
