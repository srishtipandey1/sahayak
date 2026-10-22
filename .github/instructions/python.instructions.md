---
applyTo: "**/*.py"
---
Use type hints and short docstrings. Use pathlib. No network calls in tests. Raise clear errors instead of silently continuing. Raster work: keep CRS explicit, prefer UTM for metric calculations, write Cloud Optimized GeoTIFF. Vector work: GeoPackage. Never read from data/reference/ in pipeline code.