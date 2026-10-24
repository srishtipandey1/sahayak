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

Sentinel-1 ingestion runs by default and requires the Copernicus Data Space OAuth
client ID and secret (`CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`) in the environment
or project-root `.env` file. Production settings load that file; use `sahayak doctor`
to check configured-setting status without revealing values, and
`sahayak doctor --check-auth` to test OAuth credentials.
Use `--s1-baseline-count` to choose one to three pre-event acquisitions.
For ingestion commands, install the `geo` extra in addition to the development extra:
`pip install -e ".[dev,geo]"`.

## Data attributions
- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors (ODbL).
- Sen1Floods11 training data, if used: Bonafilia et al., "Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1", CVPR Workshops 2020.
- Kuro Siwo attribution applies only if Kuro Siwo data are used: Bountos et al., "Kuro Siwo: 33 billion m² under the water. A global multi-temporal satellite dataset for rapid flood mapping", NeurIPS 2024 (Datasets and Benchmarks).
- Validation only: European Union, Copernicus Emergency Management Service data (EMSR927). Never used as an input to the system.

## Optional flood-model training

The optional Sentinel-1 model pipeline and its data provenance, limitations, Kaggle
workflow, and smoke-test instructions are documented in [docs/ml.md](docs/ml.md).