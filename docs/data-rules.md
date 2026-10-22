# Data rules

## Allowed as inputs
Sentinel-1, Sentinel-2, Copernicus DEM GLO-30, OpenStreetMap as of a snapshot date strictly before the event (via the ohsome API), Kuro Siwo and Sen1Floods11 for training, ESA/JRC auxiliary layers listed in the brief.

## Forbidden as inputs
Copernicus EMS, UNOSAT and any other published damage map, and any OSM edit made after the event. Files under data/reference/ are validation only and must never be read by pipeline code. Violating this rule disqualifies the project.

## Technical rules
- Radar change detection pairs scenes from the same relative orbit and the same pass direction.
- The OSM snapshot date must be earlier than the event date. For the Trishuli case study the latest snapshot available at time of writing is 27 July 2026.
- Every input is recorded in the run manifest with source, license and sha256.

## Attribution (keep in README, dashboard footer and reports)
- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors (ODbL). Any published modified OSM data stays under ODbL.
- Kuro Siwo: Bountos et al., NeurIPS 2024 (Datasets and Benchmarks). Sen1Floods11, if used: Bonafilia et al., CVPR Workshops 2020.
- Validation only: European Union, Copernicus Emergency Management Service data (EMSR927).

## Responsibility
No images of victims in demos or reports. This is an educational prototype, not an operational tool.