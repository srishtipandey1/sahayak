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

Sahayak is an educational prototype and is not an operational tool.
