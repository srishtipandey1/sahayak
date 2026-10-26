"""Numeric grounding checks for generated report and answer text."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

_DATE_PATTERN = re.compile(
    r"(?<!\d)\d{4}-\d{2}-\d{2}"
    r"(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2})?"
)
_NUMBER_PATTERN = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)?%?")
_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


class _TextExtractor(HTMLParser):
    """Extract visible text while ignoring document metadata and styles."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "head"}:
            self._ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "head"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)


def visible_text(content: str) -> str:
    """Return visible HTML text or the unchanged input when it is not HTML."""
    if "<" not in content or ">" not in content:
        return content
    parser = _TextExtractor()
    parser.feed(content)
    parser.close()
    return " ".join(parser.parts)


def normalize_digits(value: str) -> str:
    """Normalize Devanagari decimal digits to Western digits."""
    return value.translate(_DEVANAGARI_DIGITS)


def _allowed_values(
    headline: list[dict[str, Any]],
) -> tuple[list[float], list[float], set[str]]:
    """Collect numeric values and date tokens directly from curated headline entries."""
    allowed_numbers: list[float] = []
    allowed_percentages: list[float] = []
    allowed_dates: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            allowed_numbers.append(float(value))
            return
        if isinstance(value, str):
            for date_text in _DATE_PATTERN.findall(value):
                allowed_dates.add(date_text)
                allowed_dates.add(date_text[:4])
            for token in _NUMBER_PATTERN.findall(_DATE_PATTERN.sub(" ", value)):
                try:
                    allowed_numbers.append(float(token.removesuffix("%").replace(",", ".")))
                except ValueError:
                    continue
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)

    for item in headline:
        value = item.get("value")
        visit(value)
        if item.get("unit") == "fraction" and isinstance(value, (int, float)):
            allowed_percentages.append(float(value) * 100)
        elif item.get("unit") == "percent" and isinstance(value, (int, float)):
            allowed_percentages.append(float(value))
    return allowed_numbers, allowed_percentages, allowed_dates


def verify_text(
    content: str,
    headline: list[dict[str, Any]],
    *,
    tolerance: float = 0.011,
) -> dict[str, Any]:
    """Report numeric tokens that cannot be traced to headline values or recorded dates."""
    if tolerance < 0:
        raise ValueError("Numeric verification tolerance cannot be negative.")
    allowed_numbers, allowed_percentages, allowed_dates = _allowed_values(headline)
    text = normalize_digits(visible_text(content))
    text_without_dates = _DATE_PATTERN.sub(" ", text)
    unverified: list[str] = []
    for token in _NUMBER_PATTERN.findall(text_without_dates):
        is_percentage = token.endswith("%")
        numeric_text = token.removesuffix("%").replace(",", ".")
        try:
            numeric_value = float(numeric_text)
        except ValueError:
            unverified.append(token)
            continue
        if token in allowed_dates or numeric_text in allowed_dates:
            continue
        if is_percentage:
            decimal_places = len(numeric_text.partition(".")[2])
            rounding_tolerance = 0.5 * (10**-decimal_places)
            verified = any(
                abs(numeric_value - expected) <= rounding_tolerance + 1e-12
                for expected in allowed_percentages
            )
        else:
            verified = any(
                abs(numeric_value - expected) <= tolerance
                for expected in allowed_numbers
            )
        if not verified:
            unverified.append(token)
    return {"verified": not unverified, "unverified_numbers": unverified}
