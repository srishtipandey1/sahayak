# Sahayak

Satellite-based flood and debris-flow damage mapping for the Indian Himalaya:
where the flood hit, what was damaged, and which settlements are cut off.

Status: under development. Educational prototype, not an operational tool.

## Quick start

From Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
sahayak --help
```

OSM ingestion defaults to the public Overpass API (`OSM_PROVIDER=overpass`).
To use ohsome v2 instead, set `OSM_PROVIDER=ohsome`, configure `OHSOME_URL` and
`OHSOME_API_VERSION=v2`, and provide an API key from
[HeiGIT's ohsome account signup](https://account.heigit.org/signup).

## Data attributions
- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors (ODbL).
- Training data: Bountos et al., "Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood mapping", NeurIPS 2024 (Datasets and Benchmarks).
- Validation only: European Union, Copernicus Emergency Management Service data (EMSR927). Never used as an input to the system.