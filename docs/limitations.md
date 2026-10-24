# Limitations

This document is updated as limitations are discovered.

Sentinel revisit intervals are measured in days, so the system cannot provide early warning.

Radar shadow and layover in steep terrain can produce errors in mapped flood and debris areas.

Cloud cover can prevent useful optical observations.

OpenStreetMap completeness varies, and pre-event data means unmapped buildings and roads remain invisible.

OSM access has service constraints: ohsome v1 geometry extraction is currently restricted, ohsome v2 requires an API key, and public Overpass instances require fair-use request pacing. Complex OSM relations may not be reconstructable from a regional extraction and are counted when skipped.

When Sentinel Hub Catalog metadata has a missing or invalid relative orbit, ingestion queries the main Copernicus Data Space STAC for a matching platform and acquisition time. If no valid relative orbit and pass direction are found, that acquisition is excluded and the reason is recorded in `selection.json`; it is never paired by an inferred track. The official Process API FAQ documents a maximum output width and height of 2500 pixels; larger grids are tiled, and smaller grids use one request. Processing-unit totals are shown only when the response provides the expected header. Relative/absolute-orbit formula cross-checking is skipped because no platform-specific formula was verified in the permitted sources.

Exposure is estimated from mapped data and is not ground-truthed.

Ten-metre pixels cannot resolve small-scale damage.

The DEM is Copernicus GLO-30. Terrain conditioning, flow routing, and HAND use an averaged, configurable 30 m working grid, then HAND and streams are resampled to the 10 m map grid; resampling does not add terrain detail, so effective terrain resolution remains about 30 m. HAND stream routing and the default 1 km2 upstream-area threshold are sensitive to this source resolution and to the chosen D8 flow-routing method. Small negative HAND values are clipped only when they affect at most 1% of valid cells and the minimum is no lower than -20 m; larger negative-value prevalence or depth stops mapping and is recorded as a terrain failure. This QC is not a validation of the resulting HAND surface.

Flood-water and disturbed-surface labels are rule-based screening results, not ground-truth damage assessments. The VV water threshold is estimated from the post-event histogram and clamped to configured bounds. The confidence score combines change magnitude, VV/VH directional agreement, distance from invalid pixels, and slope using fixed documented weights; it has not been calibrated as a probability. Permanent-water classification uses the pre-event radar baseline and may include radar-dark surfaces.

The Sentinel Hub `shadowMask` marks pixels likely in or near radar shadow based on proximity to low normalized scattering area. Its documented flag is not a separate layover detector; layover may remain in the inputs. Pixels above the configured physical backscatter ceilings are excluded, but residual terrain-flattening artifacts may still affect the mapped classes. Permanent-water candidates are further restricted by HAND, slope, and component size; other baseline-dark pixels are reported separately as low-confidence dark surfaces, not water.

The pre-event null-pair is an internal placebo calibration: two pre-event scenes form the baseline and the next pre-event scene is treated as a pseudo-post. Its detections are false alarms by construction, but a single null pair does not estimate all seasonal, orbit, or acquisition variability and is not independent ground truth. `eo:cloud_cover` is the Catalog's per-tile estimate and may not describe cloud conditions over the specific valley.

Sahayak is an educational prototype and is not an operational tool.
