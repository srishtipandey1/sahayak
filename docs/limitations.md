# Limitations

This document is updated as limitations are discovered.

Sentinel revisit intervals are measured in days, so the system cannot provide early warning.

Radar shadow and layover in steep terrain can produce errors in mapped flood and debris areas.

Cloud cover can prevent useful optical observations.

Sen2Cor can classify bright bare sediment as cloud or snow/ice. The optional `bright_recovered`
class recovers a constrained subset using SCL, elevation, HAND, stream proximity, normalized B04
change, NDVI, and blue reflectance; it is a heuristic, not a validated sediment classifier.
Optical reliability uses a post-event control scene when its valid fraction is sufficient, with a
pre-event placebo retained as a secondary check. Either control can be cloud-limited, and a control
below the clear-fraction gate makes reliability `unverified`.

OpenStreetMap completeness varies, and pre-event data means unmapped buildings and roads remain invisible.

OSM access has service constraints: ohsome v1 geometry extraction is currently restricted, ohsome v2 requires an API key, and public Overpass instances require fair-use request pacing. Complex OSM relations may not be reconstructable from a regional extraction and are counted when skipped.

When Sentinel Hub Catalog metadata has a missing or invalid relative orbit, ingestion queries the main Copernicus Data Space STAC for a matching platform and acquisition time. If no valid relative orbit and pass direction are found, that acquisition is excluded and the reason is recorded in `selection.json`; it is never paired by an inferred track. The official Process API FAQ documents a maximum output width and height of 2500 pixels; larger grids are tiled, and smaller grids use one request. Processing-unit totals are shown only when the response provides the expected header. Relative/absolute-orbit formula cross-checking is skipped because no platform-specific formula was verified in the permitted sources.

Exposure is estimated from mapped data and is not ground-truthed.

Optical hazard footprints are rule-based detections rather than confirmed flood or debris-flow
boundaries. Strict/default/lenient masks are sensitivity scenarios, not confidence intervals.
Damage exposure counts inherit omissions and positional errors in both the optical raster and
pre-event OSM; the 30 m building, 20 m bridge, 100 m settlement, and 300 m hit-building proximity
rules are analysis choices, not field-validated impact thresholds.

EMSR927 comparison is a validation-only measurement, not threshold tuning, and EMSR927 is
also not perfect ground truth. Reference and Sahayak optical acquisitions may be on different
dates; for the Syapru Besi comparison the EMSR927 post-event image is from 2026-08-27 while the
Sahayak optical post-scene is from 2026-10-01. This temporal gap means mapped changes may not
represent the same conditions. The graded product's mapping scale, minimum mapping unit,
positional accuracy, and visual interpretation also constrain feature-level comparison.

Access analysis assumes fixed speeds by OSM road class, snaps nearby endpoints within 3 m, and
snaps settlements and targets to a road within 500 m. A road intersecting a mapped detection is
treated as blocked, and a bridge within 20 m is treated as blocked. The graph does not model
traffic, temporary passability, road direction changes, or field-confirmed bridge status.
`not_assessable` indicates missing connectivity or a settlement without a road snap; it does not
mean the settlement is cut off. Nearest-town and health-facility routes are estimates from OSM,
not verified travel-time predictions. No population is inferred from settlement tags.

Ten-metre pixels cannot resolve small-scale damage.

The DEM is Copernicus GLO-30. Terrain conditioning, flow routing, and HAND use an averaged, configurable 30 m working grid, then HAND and streams are resampled to the 10 m map grid; resampling does not add terrain detail, so effective terrain resolution remains about 30 m. HAND stream routing and the default 1 km2 upstream-area threshold are sensitive to this source resolution and to the chosen D8 flow-routing method. Small negative HAND values are clipped only when they affect at most 1% of valid cells and the minimum is no lower than -20 m; larger negative-value prevalence or depth stops mapping and is recorded as a terrain failure. This QC is not a validation of the resulting HAND surface.

Flowpaths are terrain-only D8 screening traces on the DEM-derived 30 m working grid.
They do not model hydraulics, discharge, flood depth, channel capacity, or structures,
and a traced path is not a forecast of where water will travel.

Flood-water and disturbed-surface labels are rule-based screening results, not ground-truth damage assessments. The VV water threshold is estimated from the post-event histogram and clamped to configured bounds. The confidence score combines change magnitude, VV/VH directional agreement, distance from invalid pixels, and slope using fixed documented weights; it has not been calibrated as a probability. Permanent-water classification uses the pre-event radar baseline and may include radar-dark surfaces.

The Sentinel Hub `shadowMask` marks pixels likely in or near radar shadow based on proximity to low normalized scattering area. Its documented flag is not a separate layover detector; layover may remain in the inputs. Pixels above the configured physical backscatter ceilings are excluded, but residual terrain-flattening artifacts may still affect the mapped classes. Permanent-water candidates are further restricted by HAND, slope, and component size; other baseline-dark pixels are reported separately as low-confidence dark surfaces, not water.

The pre-event null-pair is an internal placebo calibration: two pre-event scenes form the baseline and the next pre-event scene is treated as a pseudo-post. Its detections are false alarms by construction, but a single null pair does not estimate all seasonal, orbit, or acquisition variability and is not independent ground truth. `eo:cloud_cover` is the Catalog's per-tile estimate and may not describe cloud conditions over the specific valley.

The optional Sen1Floods11 U-Net is a separate open-water segmentation experiment,
not a debris-flow model. It has been evaluated on Sen1Floods11 and a held-out
Bolivia split, but not validated for Himalayan events; transfer to this region
and hazard type is unverified. Synthetic smoke runs are not performance results.
The dataset license should be confirmed from the applicable upstream terms
before redistributing derived weights.

Running the optional Sen1Floods11 model on an event uses gamma0 terrain backscatter
even though its training labels are sigma0 open-water floods. Its probability
threshold is selected on the training dataset's validation split, not this event.
The pre-event placebo is not independent ground truth. Reported optical/Otsu
overlaps are descriptive comparisons only; they do not establish accuracy or
justify threshold changes. Training `config.json` currently records normalization
means but no complete training-band distributions, so distributional similarity
cannot be measured from these artifacts.
In the recorded Trishuli run, the terrain-limited placebo area exceeded the
post-event model area and the post-event model mask had no overlap with the
optical default footprint; these observations argue against interpreting the
model output as event evidence. The exact measurements are retained in that
run's `facts.json`.

Sahayak is an educational prototype and is not an operational tool.

Situation reports and question answers are assembled from curated `headline.json`
measurements; the numeric verifier is a guardrail, not proof that the underlying maps
are correct. Hindi and Nepali translations require review by native speakers before
any public or operational use.

The static dashboard's optional OpenStreetMap tile layer requires an internet
connection; its default analytical layers and imagery are served locally. Dashboard
layers are simplified for display, so they are not substitutes for source GeoPackages
or analysis outputs.
