# AI assistance and human verification

I built Sahayak as a solo project, and the final responsibility for the
research direction, product decisions, scientific interpretation, and evidence
standards remains with me. I used AI as a development partner to accelerate
implementation, debugging, testing, and documentation, but AI did not define
the problem, set the project constraints, or validate the scientific claims.

AI was useful for:
- implementation support across ingestion, terrain, optical, access, reporting,
  and dashboard work;
- test authoring and debugging across edge cases and integration flows;
- drafting and refining technical documentation and submission materials;
- checking documentation consistency, attribution text, and evidence links.

My responsibilities were to:
- drive the system design and project decisions;
- verify that every number and claim traces to repo artifacts such as run facts,
  metrics, or validation files;
- maintain scientific honesty, especially around transfer limits, radar-null
  findings, and operational caveats;
- review provenance, licensing, and deployment suitability before presenting
  the work.

| Area | AI contribution | Human responsibility and evidence |
|---|---|---|
| Data ingestion and provenance | Implementation support for provider logic, selection rules, provenance checks, and test scaffolding. | I reviewed the manifests, selection records, and rule guards to confirm source dates, metadata, and allowed-input constraints. |
| Sentinel-1 flood and terrain | Helped with preprocessing, validity masks, Lee filtering, terrain logic, and diagnostics. | I checked the recorded thresholds, null-pair results, and terrain assumptions in `facts.json`, configuration, and diagnostics before interpreting the radar output. |
| Sentinel-2 optical detection | Assisted with reflectance logic, class definitions, reliability checks, and placebo tests. | Spectral detections remain mapped estimates, not field truth; I evaluated the selected scenes and measured classes against the produced run artifacts. |
| ML dataset, training, and inference | Assisted with dataset code, model workflow, baselines, and evaluation support. | I reviewed [`metrics.json`](../models/flood_unet_sen1floods11/metrics.json) and the recorded threshold to confirm the model’s results and the transfer limitation. The Trishuli case was reported as a transfer experiment, not as established event ground truth. |
| Exposure and access analysis | Helped with scenario logic, OSM exposure extraction, graph construction, and routing logic. | I checked the access assumptions, route outputs, and scenario counts in the run facts and GeoPackages; the mapped exposure remains a modelled estimate, not a field-confirmed closure list. |
| Reports and grounded Q&A | Assisted with facts curation, multilingual template drafting, and numeric verification support. | I retained responsibility for using only verified facts and for reviewing the final wording before any public-facing use. |
| Dashboard and human review | Assisted with static UI generation, API wiring, visual iteration, and review workflow support. | I reviewed the interaction flow, persisted review actions, and the resulting screenshots and tests; usability and field-readiness remain separate questions. |
| Validation and comparisons | Helped with validation code, metrics orchestration, and documentation drafting. | I reviewed the actual comparison outputs in [`validation.json`](../runs/trishuli_syapru/validate/validation.json) and kept the EMS product clearly framed as validation-only, not as input to threshold tuning. |
| Project documentation and submission materials | Assisted with README wording, technical explanations, and artifact packaging. | I verified the final claims against the repo’s run artifacts, environment files, and attribution requirements before inclusion in the submission package. |

AI-generated code and prose can speed up a project, but they do not make a
mapped feature real, a route safe to trust in the field, or a model reliable on
a new event type. Those decisions and claims remain mine, along with the
scientific and operational limitations documented in this repository.
