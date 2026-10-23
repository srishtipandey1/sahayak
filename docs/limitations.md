# Limitations

This document is updated as limitations are discovered.

Sentinel revisit intervals are measured in days, so the system cannot provide early warning.

Radar shadow and layover in steep terrain can produce errors in mapped flood and debris areas.

Cloud cover can prevent useful optical observations.

OpenStreetMap completeness varies, and pre-event data means unmapped buildings and roads remain invisible.

OSM access has service constraints: ohsome v1 geometry extraction is currently restricted, ohsome v2 requires an API key, and public Overpass instances require fair-use request pacing. Complex OSM relations may not be reconstructable from a regional extraction and are counted when skipped.

Exposure is estimated from mapped data and is not ground-truthed.

Ten-metre pixels cannot resolve small-scale damage.

Sahayak is an educational prototype and is not an operational tool.
