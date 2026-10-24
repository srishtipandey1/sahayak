# Sahayak: project brief

## Problem
After a Himalayan flash flood or glacial-lake outburst, roads, bridges, phone lines and ground sensors fail. In the first days rescuers do not know where the damage is, what was destroyed, or which villages are cut off. Free satellite data covers every valley, and radar sees through monsoon cloud. Sahayak turns that raw data into answers.

## What the system takes and gives
Input: an area (bounding box) and a flood date.
Output: three answers plus supporting products.
1. Where did the flood hit? Flooded and debris-covered areas, from before/after Sentinel-1 radar (same orbit track only) and Sentinel-2 optical where the sky is clear, with a per-pixel confidence layer.
2. What was damaged? Pre-event OpenStreetMap buildings, roads and bridges intersected with the flood layer.
3. Who is cut off? A road graph from pre-event OSM, with flooded or damaged roads and bridges removed, then routing from each settlement to the nearest town and hospital. Settlements with no route are reported as cut off.

## Two modes
- Respond mode (after a flood): the three answers plus a bilingual situation report.
- Prepare mode (before a flood): pick an upstream point such as a glacial lake. Trace the likely flood path down the valley using DEM and list the settlements, roads and bridges along it. This supports disaster mitigation.

## AI components (both are built)
1. Flood segmentation model: trained on Sen1Floods11 hand-labeled Sentinel-1 chips. Kuro Siwo is not enabled unless a labeled GRD subset with a documented download under 10 GB is verified. Evaluate on held-out regions and Himalayan scenes never seen in training; report honestly against simple threshold baselines.
2. Situation-report copilot: answers rescuers' questions and writes short reports in English, Hindi and Nepali. All numbers come from a facts JSON produced by the maps. A verifier rejects any output containing a number that is not in the facts JSON. A template-only fallback works with no internet and no language model.

## Terrain-aware radar logic
Steep valleys cause radar shadow and layover, which create false flood detections. Use DEM-derived slope and HAND (height above nearest drainage) masks. Ignore detections on terrain too steep or too high above the river to hold water. Debris flows raise radar backscatter roughness while ponded water lowers it, so detect change in both directions.

## Validation (honesty first)
- Case study: the 26 August 2026 Bhote Koshi-Trishuli flood, Nepal. Compare with Copernicus EMS EMSR927, checking only.
- A second, Indian Himalayan event for transfer testing, confirmed during validation.
- Report precision, recall, F1 and IoU. Report failures, such as narrow gorges and radar shadow, in the same tables as successes.

## Deliverables
Map dashboard, one-page situation report (PDF), public GitHub repository with README, 6 to 7 slide deck including the AI-usage disclosure, 3 to 5 minute demo video, limitations document, attributions.

## Limitations to document
Sentinel revisit is days, so no early warning. Radar shadow and layover in steep terrain. Optical cloud cover. OSM completeness varies and is pre-event, so unmapped buildings and roads are invisible. Exposure is estimated, not ground-truthed. 10 m pixels cannot see small damage. Educational prototype, not an operational tool.

## Priority order
Must: ingestion, flood/debris mapping, damage exposure, cut-off analysis, validation, docs. Then: segmentation model. Then: flow path and Prepare mode, copilot and bilingual report, dashboard polish. Always keep a runnable end-to-end path.