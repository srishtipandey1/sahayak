"""Common projected raster grid selection and metadata."""

from dataclasses import asdict, dataclass
from math import ceil, floor

from rasterio import CRS
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform_bounds


@dataclass(frozen=True)
class GridSpec:
    """A pixel-aligned UTM grid shared by all pipeline rasters."""

    bbox: tuple[float, float, float, float]
    epsg: int
    resolution: float
    left: float
    bottom: float
    right: float
    top: float

    @classmethod
    def from_bbox(
        cls,
        bbox: tuple[float, float, float, float],
        resolution: float = 10.0,
    ) -> "GridSpec":
        """Choose the centre UTM zone and snap the projected extent to pixel edges."""
        min_lon, min_lat, max_lon, max_lat = bbox
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError("Longitude bounds must be ordered and within -180 to 180.")
        if not (-80 <= min_lat < max_lat <= 84):
            raise ValueError("UTM grid latitudes must be ordered and within -80 to 84.")
        if resolution <= 0:
            raise ValueError("Grid resolution must be greater than zero.")

        centre_lon = (min_lon + max_lon) / 2
        centre_lat = (min_lat + max_lat) / 2
        zone = min(60, max(1, floor((centre_lon + 180) / 6) + 1))
        epsg = (32600 if centre_lat >= 0 else 32700) + zone
        projected_bounds = transform_bounds(
            "EPSG:4326",
            f"EPSG:{epsg}",
            min_lon,
            min_lat,
            max_lon,
            max_lat,
            densify_pts=41,
        )
        left, bottom, right, top = projected_bounds
        return cls(
            bbox=bbox,
            epsg=epsg,
            resolution=resolution,
            left=floor(left / resolution) * resolution,
            bottom=floor(bottom / resolution) * resolution,
            right=ceil(right / resolution) * resolution,
            top=ceil(top / resolution) * resolution,
        )

    @property
    def width(self) -> int:
        """Return the number of columns in the grid."""
        return round((self.right - self.left) / self.resolution)

    @property
    def height(self) -> int:
        """Return the number of rows in the grid."""
        return round((self.top - self.bottom) / self.resolution)

    @property
    def crs(self) -> str:
        """Return the grid CRS as an EPSG identifier."""
        return f"EPSG:{self.epsg}"

    @property
    def transform(self) -> Affine:
        """Return the north-up affine transform for the grid."""
        return from_origin(self.left, self.top, self.resolution, self.resolution)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return left, bottom, right, top in the projected grid CRS."""
        return self.left, self.bottom, self.right, self.top

    @property
    def geographic_bounds(self) -> tuple[float, float, float, float]:
        """Transform the grid extent back to a WGS84 bounding box."""
        return transform_bounds(
            self.crs,
            "EPSG:4326",
            self.left,
            self.bottom,
            self.right,
            self.top,
            densify_pts=41,
        )

    def bbox_to_grid_bounds(self) -> tuple[float, float, float, float]:
        """Return the snapped projected bounds corresponding to the input bbox."""
        return self.bounds

    def grid_bounds_to_bbox(self) -> tuple[float, float, float, float]:
        """Return the geographic envelope of this snapped grid."""
        return self.geographic_bounds

    def rasterio_profile(
        self,
        *,
        dtype: str = "float32",
        nodata: float | int | None = None,
        count: int = 1,
    ) -> dict[str, object]:
        """Build a rasterio creation profile aligned to this grid."""
        return {
            "driver": "GTiff",
            "height": self.height,
            "width": self.width,
            "count": count,
            "dtype": dtype,
            "crs": self.crs,
            "transform": self.transform,
            "nodata": nodata,
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize grid metadata for the run selection record."""
        return {
            **asdict(self),
            "width": self.width,
            "height": self.height,
            "crs": self.crs,
            "transform": tuple(self.transform),
            "geographic_bounds": self.geographic_bounds,
        }


def transform_bbox_to_grid_bounds(
    bbox: tuple[float, float, float, float],
    resolution: float = 10.0,
) -> tuple[GridSpec, tuple[float, float, float, float]]:
    """Select a common grid and return the snapped bounds."""
    grid = GridSpec.from_bbox(bbox, resolution)
    return grid, grid.bbox_to_grid_bounds()


def grid_bounds_to_bbox(
    bounds: tuple[float, float, float, float],
    epsg: int,
) -> tuple[float, float, float, float]:
    """Transform projected bounds into their WGS84 bounding envelope."""
    left, bottom, right, top = bounds
    return transform_bounds(
        CRS.from_epsg(epsg),
        CRS.from_epsg(4326),
        left,
        bottom,
        right,
        top,
        densify_pts=41,
    )
