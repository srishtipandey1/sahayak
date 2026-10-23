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

Flood mapping uses the Sentinel-1 VV/VH, local-incidence-angle, shadow-mask, and data-mask bands by description. The Sentinel Hub Sentinel-1 GRD documentation defines `localIncidenceAngle` as the angle in degrees for each output pixel (available with orthorectification), `dataMask` as the data/nodata mask (0 no data, 1 data), and `shadowMask` as true for pixels in or near radar shadow: the nearest GRD source pixel is no more than one GRD pixel from a pixel with normalized `scatteringArea < 0.05` (0 likely not in shadow; 1 likely in/near shadow). This stage excludes positive shadow-mask values. These documented mask flags do not establish a separate layover flag.

The flood stage applies a masked Lee filter to linear power, converts filtered VV/VH to dB, and uses the three pre-event images for the per-pixel baseline. Terrain conditioning, D8 flow direction, accumulation, and HAND run on a configurable 30 m working grid: the GLO-30 DEM is averaged from the 10 m common grid, and HAND is bilinearly resampled while streams are nearest-neighbour resampled back to the 10 m output grid. PySheds computes HAND from the same conditioned DEM used for flow direction. Stream cells use an upstream-area threshold from `configs/flood.toml`. The DEM is Copernicus GLO-30; resampling does not add terrain detail, so effective terrain resolution remains about 30 m. HAND negative-value QC and the fraction of stream cells below the selected pre-event VV water threshold are recorded in `facts.json`. Flood and disturbed-surface thresholds, cleanup, confidence, and HAND sensitivity values are also recorded there; confidence is a documented heuristic, not a calibrated probability. Outputs are written under `runs/<run_id>/flood/`.

## Formats
Rasters: Cloud Optimized GeoTIFF in a projected UTM CRS chosen from the bbox. Vectors: GeoPackage. Facts: JSON. Dashboard layers: GeoJSON or tiles derived from these.

## Run folder layout
runs/<run_id>/manifest.json, inputs/, masks/, vectors/, facts.json, report/, logs/

## facts.json principles
Every value carries its unit, the module that produced it, and the input scenes it depends on. No number appears in a report unless it is in facts.json.