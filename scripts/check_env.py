"""Verify optional geospatial dependencies and a basic GeoTIFF read/write cycle."""

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas
import networkx
import numpy
import pyproj
import pysheds
import pystac_client
import rasterio
import scipy
import shapely
import skimage
from rasterio.transform import from_origin


def main() -> None:
    """Print installed geospatial package versions and test a tiny GeoTIFF."""
    packages = (
        ("rasterio", rasterio, "rasterio"),
        ("geopandas", geopandas, "geopandas"),
        ("shapely", shapely, "shapely"),
        ("pyproj", pyproj, "pyproj"),
        ("numpy", numpy, "numpy"),
        ("scipy", scipy, "scipy"),
        ("skimage", skimage, "scikit-image"),
        ("pystac_client", pystac_client, "pystac-client"),
        ("networkx", networkx, "networkx"),
        ("pysheds", pysheds, "pysheds"),
    )
    for module_name, _module, distribution_name in packages:
        print(f"{module_name}: {version(distribution_name)}")

    sample = numpy.array([[1, 2], [3, 4]], dtype=numpy.uint8)
    with TemporaryDirectory() as temporary_directory:
        raster_path = Path(temporary_directory) / "smoke-test.tif"
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=sample.shape[0],
            width=sample.shape[1],
            count=1,
            dtype=sample.dtype,
            crs="EPSG:4326",
            transform=from_origin(85.0, 28.5, 0.1, 0.1),
        ) as dataset:
            dataset.write(sample, 1)

        with rasterio.open(raster_path) as dataset:
            read_back = dataset.read(1)

        numpy.testing.assert_array_equal(read_back, sample)

    print("GeoTIFF write/read: passed")


if __name__ == "__main__":
    main()
