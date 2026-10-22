# Copilot instructions: Sahayak

Full rules are in AGENTS.md. Context is in docs/project-brief.md, docs/data-rules.md and docs/architecture.md. Read them first.

Non-negotiable:
- Forbidden inputs: Copernicus EMS, UNOSAT, any published damage map, OSM edits after the event. data/reference/ is validation only.
- Allowed inputs: Sentinel-1, Sentinel-2, Copernicus DEM GLO-30, pre-event OSM via ohsome, Kuro Siwo, Sen1Floods11.
- Radar pairs must share relative orbit and pass direction.
- Numbers in reports come only from facts.json. A language model never invents numbers.
- Never fabricate results or command output. Say when unsure.
- Never run git commit or git push.
- Never open, read, print or log the .env file or any secret value.
- Windows PowerShell, Python 3.11, pathlib, no bash-only tooling.
- Code must pass ruff check . and pytest. Add tests for new code.
- End every task with: files changed, exact commands run with real output, uncertainties.