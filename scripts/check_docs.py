"""Check README environment settings and required attribution text."""

from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class FooterTextParser(HTMLParser):
    """Collect text from the dashboard's attribution footer."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_attribution = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "footer":
            classes = dict(attrs).get("class", "") or ""
            self.in_attribution = "attribution" in classes.split()

    def handle_endtag(self, tag: str) -> None:
        if tag == "footer":
            self.in_attribution = False

    def handle_data(self, data: str) -> None:
        if self.in_attribution:
            self.parts.append(data)


def _normalize(text: str) -> str:
    text = text.translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _environment_file_names(text: str) -> set[str]:
    return {
        match.group(1)
        for line in text.splitlines()
        if (match := re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line))
    }


def _readme_environment_names(text: str) -> set[str]:
    names: set[str] = set()
    for line in text.splitlines():
        match = re.match(r"\|\s*`([A-Z][A-Z0-9_]*)`\s*\|", line)
        if match:
            names.add(match.group(1))
    return names


def _required_attributions(text: str) -> list[str]:
    section = re.search(
        r"^## Attribution\b(.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if section is None:
        raise ValueError("docs/data-rules.md has no Attribution section.")
    return [
        match.group(1).strip()
        for line in section.group(1).splitlines()
        if (match := re.match(r"\s*-\s+(.+?)\s*$", line))
    ]


def _missing_attributions(required: list[str], content: str) -> list[str]:
    normalized_content = _normalize(content)
    return [
        attribution
        for attribution in required
        if _normalize(attribution) not in normalized_content
    ]


def main() -> int:
    """Run repository documentation consistency checks."""
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    environment_example = (root / ".env.example").read_text(encoding="utf-8")
    data_rules = (root / "docs" / "data-rules.md").read_text(encoding="utf-8")
    dashboard = (root / "dashboard" / "index.html").read_text(encoding="utf-8")

    example_names = _environment_file_names(environment_example)
    readme_names = _readme_environment_names(readme)
    missing_from_readme = sorted(example_names - readme_names)
    missing_from_example = sorted(readme_names - example_names)
    failures: list[str] = []
    if missing_from_readme:
        failures.append(f"README omits .env.example variables: {', '.join(missing_from_readme)}")
    if missing_from_example:
        failures.append(f"README documents unknown variables: {', '.join(missing_from_example)}")

    required = _required_attributions(data_rules)
    footer_parser = FooterTextParser()
    footer_parser.feed(dashboard)
    footer_text = " ".join(footer_parser.parts)
    for label, content in (("README.md", readme), ("dashboard footer", footer_text)):
        missing = _missing_attributions(required, content)
        if missing:
            failures.append(f"{label} is missing required attribution(s): {'; '.join(missing)}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print("Documentation consistency check passed.")
    print(f"Environment variables: {len(example_names)} matched.")
    print(f"Attribution strings: {len(required)} present in README and dashboard footer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
