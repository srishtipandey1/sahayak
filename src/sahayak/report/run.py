"""One-page multilingual HTML and optional browser-rendered PDF reports."""

from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from sahayak.config import Settings
from sahayak.ingest.run import validate_run_id
from sahayak.report.facts import write_headline
from sahayak.report.i18n import message
from sahayak.report.verify import verify_text

REPORT_ROWS = (
    ("row_hazard", "hazard_area"),
    ("row_buildings_inside", "buildings_inside"),
    ("row_buildings_near", "buildings_near_30m"),
    ("row_road_km", "road_km_hit"),
    ("row_bridges", "bridges_hit_within_20m"),
    ("row_settlements", "settlements_affected"),
    ("row_cut_off", "cut_off"),
    ("row_degraded", "degraded"),
    ("row_not_assessable", "not_assessable"),
)
def _headline_value(headline: list[dict[str, Any]], fact_id: str) -> Any:
    """Return a headline measurement or None when it is absent."""
    for item in headline:
        if item.get("id") == fact_id:
            return item.get("value")
    return None


def _display(value: Any, language: str, unit: str | None = None) -> str:
    """Format curated values without translating or changing their digits."""
    if value is None:
        return message(language, "available")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if unit in {"buildings", "bridges", "settlements"}:
            return str(int(value)) if float(value).is_integer() else f"{value:g}"
        if unit == "fraction":
            return f"{value * 100:.1f}%"
        return f"{value:.2f}"
    if isinstance(value, list):
        return ", ".join(_display(part, language) for part in value)
    return str(value)


def _find_browser() -> Path | None:
    """Find an installed Edge or Chrome executable on supported desktop systems."""
    candidates: list[str] = []
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        executable = shutil.which(name)
        if executable:
            candidates.append(executable)
    for variable, relative in (
        ("PROGRAMFILES", Path("Microsoft/Edge/Application/msedge.exe")),
        ("PROGRAMFILES(X86)", Path("Microsoft/Edge/Application/msedge.exe")),
        ("LOCALAPPDATA", Path("Microsoft/Edge/Application/msedge.exe")),
        ("PROGRAMFILES", Path("Google/Chrome/Application/chrome.exe")),
        ("PROGRAMFILES(X86)", Path("Google/Chrome/Application/chrome.exe")),
        ("LOCALAPPDATA", Path("Google/Chrome/Application/chrome.exe")),
    ):
        base = os.environ.get(variable)
        if base:
            candidates.append(str(Path(base) / relative))
    return next((Path(path) for path in candidates if Path(path).is_file()), None)


def _pdf_page_count(path: Path) -> int:
    """Read the page-tree count emitted by Chromium-based browsers."""
    data = path.read_bytes()
    match = re.search(rb"/Type\s*/Pages\b(?:(?!endobj).)*?/Count\s+(\d+)", data, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not read the page count from generated PDF {path.name}.")
    return int(match.group(1))


def _render_pdf(browser: Path, html_path: Path, pdf_path: Path) -> int:
    """Render a local report with a headless Chromium-compatible browser."""
    result = subprocess.run(
        [
            str(browser),
            "--headless",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            f"--print-to-pdf={pdf_path}",
            "--print-to-pdf-no-header",
            html_path.resolve().as_uri(),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "no browser error text"
        raise RuntimeError(f"Browser PDF rendering failed: {detail}")
    if not pdf_path.is_file():
        raise RuntimeError("Browser exited without creating the requested PDF.")
    return _pdf_page_count(pdf_path)


def _report_html(
    language: str,
    headline: list[dict[str, Any]],
    *,
    run_dir: Path,
) -> str:
    """Create the single-page localized report from headline values only."""
    scenarios = ("strict", "default", "lenient")
    header = "".join(
        f"<th>{html.escape(message(language, f'scenario_{scenario}'))}</th>"
        for scenario in scenarios
    )
    rows: list[str] = []
    for label_key, fact_prefix in REPORT_ROWS:
        cells = []
        for scenario in scenarios:
            metric_id = (
                f"{fact_prefix}_{scenario}"
                if fact_prefix not in {"cut_off", "degraded", "not_assessable"}
                else f"{fact_prefix}_{scenario}"
            )
            entry = next((item for item in headline if item.get("id") == metric_id), None)
            value = entry.get("value") if entry else None
            unit = entry.get("unit") if entry else None
            cells.append(f"<td>{html.escape(_display(value, language, unit))}</td>")
        rows.append(
            f"<tr><th scope='row'>{html.escape(message(language, label_key))}</th>"
            + "".join(cells)
            + "</tr>"
        )

    reliability = _headline_value(headline, "optical_reliability")
    valid_fraction = _headline_value(headline, "optical_valid_fraction")
    radar = _headline_value(headline, "radar_finding")
    event_date = _headline_value(headline, "event_date")
    bbox = _headline_value(headline, "focus_bbox")
    dates = [
        str(_headline_value(headline, f"optical_{role}_date") or "")[:10]
        for role in ("pre", "post")
        if _headline_value(headline, f"optical_{role}_date")
    ]
    cut_off = _headline_value(headline, "cut_off_names_default") or []
    selected_names = cut_off[:10]
    sources = _headline_value(headline, "data_sources") or []
    names_markup = "".join(
        f"<li>{html.escape(str(name))}</li>" for name in selected_names
    )
    sources_markup = "".join(
        f"<li>{html.escape(str(source))}</li>" for source in sources
    )
    attributions = _headline_value(headline, "report_attributions") or []
    attribution_markup = "".join(
        f"<li>{html.escape(str(value))}</li>" for value in attributions
    )
    reliability_class = (
        "low" if reliability in {None, "low", "unverified"} else "normal"
    )
    available = html.escape(message(language, "available"))
    return f"""<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(message(language, "title"))}</title>
<style>
@font-face{{
 font-family:NotoSans;
 src:url('fonts/NotoSans-Regular.ttf') format('truetype');
 font-weight:400
}}
@font-face{{
 font-family:NotoDevanagari;
 src:url('fonts/NotoSansDevanagari-Regular.ttf') format('truetype');
 font-weight:400
}}
*{{box-sizing:border-box}}
body{{
 font-family:NotoSans,NotoDevanagari,sans-serif;
 color:#18302d;margin:0 auto;padding:12px;max-width:980px;
 font-size:10px;line-height:1.25
}}
h1{{font-size:19px;margin:0 0 6px}} h2{{font-size:12px;margin:9px 0 4px}}
.meta,.box{{border:1px solid #b8c9c4;border-radius:5px;padding:6px;margin:4px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.warn{{background:#fff2e8;border-color:#db8b52}} .low{{background:#ffe5e5;border-color:#cb4444}}
table{{border-collapse:collapse;width:100%;font-size:9px}}
th,td{{border:1px solid #c4d1ce;padding:3px 4px;text-align:left}}
thead{{background:#e9f1ef}} img{{max-width:100%;max-height:145px;object-fit:contain}}
ul{{margin:2px 0;padding-left:18px}} footer{{font-size:8px;margin-top:6px}}
@page{{size:A4 landscape;margin:7mm}}
@media print{{body{{padding:0;max-width:none}} h2{{margin-top:7px}}}}
</style>
</head>
<body>
<h1>{html.escape(message(language, "title"))}</h1>
<div class="meta"><strong>{html.escape(message(language, "area"))}:</strong>
 {html.escape(str(event_date) if event_date is not None else available)} |
 {html.escape(_display(bbox, language))}</div>
<h2>{html.escape(message(language, "headline"))}</h2>
<table><thead><tr><th> </th>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<div class="grid">
 <section class="box"><h2>{html.escape(message(language, "access"))}</h2>
  <strong>{html.escape(message(language, "robust"))}:</strong>
  {html.escape(_display(_headline_value(headline, "cut_off_names_robust"), language))}
  <ol>{names_markup}</ol>
 </section>
 <section class="box {reliability_class}"><h2>{html.escape(message(language, "limitations"))}</h2>
  <p>{html.escape(message(language, "row_reliability"))}:
  {html.escape(str(reliability) if reliability is not None else available)}.
  {html.escape(message(language, "row_valid"))}:
  {html.escape(_display(valid_fraction, language, "fraction"))}.</p>
  <p>{html.escape(str(radar) if radar is not None else available)}</p>
  <p>{html.escape(message(language, "detections"))}</p>
 </section>
 <section class="box"><h2>{html.escape(message(language, "data_used"))}</h2>
  <p>{html.escape(', '.join(dates) if dates else available)}</p><ul>{sources_markup}</ul>
 </section>
 <section class="box"><h2>{html.escape(message(language, "headline"))}</h2>
  <img src="../access_map.png" alt="Settlement access map">
  <img src="../optical/optical_overlay.png" alt="Optical detection overlay">
 </section>
</div>
<footer><ul>{attribution_markup}</ul></footer>
</body></html>"""


def generate_reports(
    *,
    run_id: str,
    languages: tuple[str, ...] = ("en", "hi", "ne"),
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Write and verify localized one-page reports and available PDFs."""
    validate_run_id(run_id)
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("Report languages must be a non-empty unique sequence.")
    if set(languages) - {"en", "hi", "ne"}:
        raise ValueError("Report languages must be en, hi, or ne.")
    active_settings = settings or Settings()
    run_dir = active_settings.resolved_runs_dir / run_id
    headline_path = run_dir / "headline.json"
    if not headline_path.is_file():
        headline_path = write_headline(run_id, active_settings)
    headline = json.loads(headline_path.read_text(encoding="utf-8"))
    if not isinstance(headline, list):
        raise ValueError("headline.json must contain a list of curated facts.")
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    font_dir = report_dir / "fonts"
    source_font_dir = Path(__file__).parent / "fonts"
    font_sources = {
        "NotoSans-Regular.ttf": source_font_dir / "NotoSans-Regular.ttf",
        "NotoSansDevanagari-Regular.ttf": source_font_dir / "NotoSansDevanagari-Regular.ttf",
    }
    if all(path.is_file() for path in font_sources.values()):
        font_dir.mkdir(parents=True, exist_ok=True)
        for name, path in font_sources.items():
            shutil.copy2(path, font_dir / name)

    browser = _find_browser()
    results: list[dict[str, Any]] = []
    for language in languages:
        content = _report_html(language, headline, run_dir=run_dir)
        verification = verify_text(content, headline)
        if not verification["verified"]:
            raise RuntimeError(
                "Report contains unverified numeric token(s): "
                + ", ".join(verification["unverified_numbers"])
            )
        html_path = report_dir / f"situation-{language}.html"
        html_path.write_text(content, encoding="utf-8")
        pdf_path: Path | None = None
        page_count: int | None = None
        if browser is not None:
            pdf_path = report_dir / f"situation-{language}.pdf"
            page_count = _render_pdf(browser, html_path, pdf_path)
            if page_count != 1:
                raise RuntimeError(
                    f"Expected a one-page {language} report; browser produced {page_count} pages."
                )
        results.append(
            {
                "language": language,
                "html": html_path,
                "html_bytes": html_path.stat().st_size,
                "pdf": pdf_path,
                "pdf_bytes": pdf_path.stat().st_size if pdf_path is not None else None,
                "page_count": page_count,
                "verification": verification,
                "browser_available": browser is not None,
            }
        )
    return results
