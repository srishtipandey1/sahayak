# AGENTS.md: Sahayak

Read docs/project-brief.md, docs/data-rules.md and docs/architecture.md before changing anything.

## Mission
Sahayak maps flood and debris-flow damage in the Himalaya from free satellite data and tells rescuers where the flood hit, what was damaged, and which settlements are cut off. It is an educational prototype for a national youth hackathon, not an operational tool.

## Hard rules
1. Allowed inputs: Sentinel-1, Sentinel-2, Copernicus DEM GLO-30, OSM snapshot strictly before the event date (ohsome API), Kuro Siwo, Sen1Floods11. Forbidden as inputs: Copernicus EMS, UNOSAT or any published damage map, OSM edits after the event. data/reference/ is validation only and pipeline code must never read it.
2. Radar change detection: same relative orbit and same pass direction only.
3. Every number in any report, answer or dashboard must come from facts.json produced by the maps. A language model may phrase text but never create or alter numbers.
4. Never fabricate metrics, results, citations, file contents or command output. If something fails or you are unsure, say so plainly.
5. Never commit or push. The humans run git manually. Never write secrets, data or model weights into tracked files.
6. Keep the required attribution strings in docs and UI (see docs/data-rules.md).
7. Never open, read, print or log the .env file or any secret value.

## Environment
Windows 11, PowerShell, Python 3.11, virtualenv at .venv (already active when you work). Use pathlib, not string paths. No bash-only scripts and no Makefile. Do not install packages globally. Heavy ML dependencies live in the optional extra "ml".

## Stack
src layout package sahayak. Typer CLI, pydantic-settings, rasterio, geopandas, shapely, pyproj, numpy, scipy, scikit-image, pystac-client, requests, networkx, pysheds, FastAPI + uvicorn. ML: torch and segmentation-models-pytorch. Frontend: a Leaflet-based dashboard in dashboard/. Do not substitute libraries without saying why.

## Engineering standards
Small focused functions with type hints and docstrings. Every module has tests in tests/ using small synthetic data, no network calls in unit tests. Code passes ruff check . and pytest. Config via environment and configs/, never hard-coded paths. Log with the logging module. Deterministic seeds for ML. Fail loudly with clear messages.

## Scientific honesty
Evaluate on held-out events. Compare against a simple threshold baseline. Report failures next to successes. Document limitations in docs/limitations.md as they are discovered.

## How to report back
End every task with: (a) files created or changed, (b) the exact commands you ran and their real output, (c) anything uncertain, skipped or different from the request. Never claim something works unless you ran it.