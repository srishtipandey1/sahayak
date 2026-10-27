# AI assistance and human verification

AI coding assistants (GitHub Copilot and a chat assistant) contributed to
implementation, test authoring, debugging, and documentation. The repository
does not track line-by-line authorship or a percentage allocation, so the
shares below are qualitative estimates rather than measured proportions.

Automated tests, lint results, stored run facts, and evaluation artifacts are
identified as evidence. They are not represented as independent human
sign-off. No separate project-wide human approval log is currently kept.
Human review remains necessary for scientific interpretation, field
suitability, data licensing, and translated reports.

| Area | AI role | Human verification and evidence recorded in the repository | Approx. AI share |
|---|---|---|---|
| Data ingestion and provenance | Assisted with provider/client implementation, selection rules, provenance guards, and tests. | Test suite exercises code paths and rule guards; per-run manifests and selection records expose the actual dates, sources, and metadata for review. Live-data suitability still requires event-specific review. | High; percentage not measured |
| Sentinel-1 flood and terrain | Assisted with preprocessing, validity masks, Lee filtering, terrain logic, diagnostics, and tests. | `facts.json`, configuration, diagnostics, and tests preserve thresholds and null-pair results. Human interpretation of shadow, layover, and terrain screening is still required. | High; percentage not measured |
| Sentinel-2 optical detection | Assisted with indices, classes, confidence/reliability, placebo logic, and tests. | Run facts, quicklooks, configuration, and tests make the selected scenes and measured classes inspectable. Spectral detections are not field-confirmed. | High; percentage not measured |
| ML dataset, training, and inference | Assisted with dataset code, model workflow, baselines, inference, evaluation, and focused tests. | [`metrics.json`](../models/flood_unet_sen1floods11/metrics.json) records held-out model/baseline results and the threshold; tests cover implementation behavior. The Trishuli transfer result remains unvalidated and must not be described as event ground truth. | High; percentage not measured |
| Exposure and access analysis | Assisted with hazard scenarios, OSM exposure, road graph construction, routing, and tests. | Scenario counts and assumptions are saved in run facts and GeoPackages; synthetic tests exercise edge cases. Field verification of routes, bridges, and mapped feature completeness is not recorded. | High; percentage not measured |
| Reports and grounded Q&A | Assisted with facts curation, multilingual templates, numeric verification, and tests. | The verifier and tests constrain numeric claims to curated facts. No native-speaker sign-off is recorded; translations require review before publication. | High; percentage not measured |
| Dashboard and human review | Assisted with the static map UI, API integration, visual changes, and review workflow. | The repository includes API/tests and screenshots; reviewers can persist decisions and reasons separately from detections. No independent usability or accessibility study is recorded. | High; percentage not measured |
| Validation and comparisons | Assisted with validation code, metrics, and documentation. | [`validation.json`](../runs/trishuli_syapru/validate/validation.json) and the comparison figure preserve the overlap metrics and timing caveat. The EMS product is validation-only and is not used to tune thresholds. | High; percentage not measured |
| Project documentation and submission materials | Assisted with technical explanations, README, design documentation, and artifact preparation. | Claims are linked to repository artifacts and attribution rules; the documentation checker compares environment settings and required credits. A human must still review claims and external licensing before submission. | High; percentage not measured |

AI-generated code or prose does not establish that a mapped feature is real,
that a route is blocked in the field, or that a model generalizes to a new
hazard type. Scientific and operational claims remain the responsibility of
the project team.
