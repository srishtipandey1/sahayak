# Architecture

## Pipeline
bbox + event date
  -> ingest      (Sentinel-1 pair, Sentinel-2 scenes + cloud mask, DEM, OSM snapshot)
  -> terrain     (slope, HAND, valley mask from DEM)
  -> flood       (radar change detection + optical indices + terrain masks -> flood/debris mask + confidence)
  -> ml          (segmentation model refines or replaces the baseline mask)
  -> damage      (OSM buildings, roads, bridges intersected with the mask)
  -> access      (road graph, remove hit edges, route to nearest town/hospital, list cut-off settlements)
  -> flowpath    (DEM flow tracing from an upstream point, settlements along the path)
  -> facts.json  (single source of truth for every reported number)
  -> copilot     (answers and reports from facts.json only; EN / HI / NE)
  -> report + api + dashboard

## Modules
ingest: download and cache inputs, record provenance. flood: baseline mapping. ml: dataset, training, inference, evaluation. damage: exposure analysis. access: network analysis. flowpath: bonus tracing. copilot: grounded language layer and verifier. report: one-page PDF. api: FastAPI serving layers and facts to the dashboard. provenance: run manifest and rule guards.

OSM ingestion uses a provider interface selected by `OSM_PROVIDER`: Overpass (default) makes date-pinned, sequential tile queries; ohsome v2 uses its authenticated extraction API. Overpass requests metadata and geometry, verifies every returned element timestamp against the selected snapshot, stores only layer-whitelisted columns plus the complete tags as JSON, and publishes its GeoPackage atomically.

Sentinel-1 ingestion authenticates with Copernicus Data Space OAuth, searches the Catalog for IW GRD slices, groups slices into data-takes, and selects only pre/post acquisitions sharing the same positive relative orbit and pass direction. Instrument mode, polarisation, platform, and acquisition time use the Catalog properties when available, with the Sentinel-1 item ID as a checked fallback; unsupported or malformed items are excluded individually and recorded in `selection.json`. If a Catalog relative orbit is missing or invalid, ingestion looks for a matching platform and acquisition time in the main Copernicus Data Space STAC and caches that lookup. Acquisitions without valid track metadata are excluded and recorded in `selection.json`. Process API output is requested in windows aligned to the common UTM grid and written as float32 Cloud Optimized GeoTIFFs. The implementation uses the documented 2500-pixel maximum per output dimension; grids within that limit use one request, and larger grids are tiled.

## Formats
Rasters: Cloud Optimized GeoTIFF in a projected UTM CRS chosen from the bbox. Vectors: GeoPackage. Facts: JSON. Dashboard layers: GeoJSON or tiles derived from these.

## Run folder layout
runs/<run_id>/manifest.json, inputs/, masks/, vectors/, facts.json, report/, logs/

## facts.json principles
Every value carries its unit, the module that produced it, and the input scenes it depends on. No number appears in a report unless it is in facts.json.