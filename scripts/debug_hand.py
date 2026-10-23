"""Reproduce the original Trishuli raw-versus-conditioned HAND mismatch."""

from pathlib import Path

import numpy as np
import rasterio
from pysheds.grid import Grid
from pysheds.sview import Raster

from sahayak.flood.config import load_flood_config

ROOT = Path(__file__).resolve().parents[1]
DEM_PATH = ROOT / "runs" / "trishuli_syapru" / "dem.tif"
FLOOD_CONFIG = load_flood_config(ROOT / "configs" / "flood.toml")
STREAM_AREA_THRESHOLD_KM2 = FLOOD_CONFIG.stream_area_threshold_km2
DIRECTIONS = {
    64: (-1, 0),
    128: (-1, 1),
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
}


def _stream_length_km(
    stream: np.ndarray,
    flow_direction: np.ndarray,
    x_size: float,
    y_size: float,
) -> float:
    """Sum D8 edge lengths where both upstream and downstream cells are streams."""
    total_m = 0.0
    rows, columns = stream.shape
    for code, (row_step, column_step) in DIRECTIONS.items():
        edge_sources = stream & (flow_direction == code)
        source_rows, source_columns = np.nonzero(edge_sources)
        destination_rows = source_rows + row_step
        destination_columns = source_columns + column_step
        inside = (
            (destination_rows >= 0)
            & (destination_rows < rows)
            & (destination_columns >= 0)
            & (destination_columns < columns)
        )
        if np.any(inside):
            connects_stream = stream[
                destination_rows[inside],
                destination_columns[inside],
            ]
            step_length = float(np.hypot(row_step * y_size, column_step * x_size))
            total_m += float(connects_stream.sum()) * step_length
    return total_m / 1000.0


def _negative_summary(
    name: str,
    hand: np.ndarray,
    valid: np.ndarray,
    streams: np.ndarray,
) -> None:
    """Print negative HAND counts, fractions, minimums, and representative locations."""
    negative = valid & np.isfinite(hand) & (hand < 0)
    stream_negative = negative & streams
    other_negative = negative & ~streams
    valid_count = int(valid.sum())
    print(f"{name} negative HAND:")
    print(f"  valid cells: {valid_count}")
    print(f"  negative count: {int(negative.sum())}")
    print(f"  negative fraction of valid cells: {negative.sum() / valid_count:.8%}")
    minimum = float(hand[negative].min()) if negative.any() else "n/a"
    print(f"  minimum among negative cells: {minimum}")
    print(f"  negative on stream cells: {int(stream_negative.sum())}")
    print(f"  negative elsewhere: {int(other_negative.sum())}")
    print(f"  stream locations (row, col; first 20): {np.argwhere(stream_negative)[:20].tolist()}")
    print(f"  other locations (row, col; first 20): {np.argwhere(other_negative)[:20].tolist()}")


def main() -> None:
    """Run non-mutating terrain diagnostics on the existing DEM."""
    with rasterio.open(DEM_PATH) as source:
        source_dem = source.read(1).astype(np.float32)
        nodata = source.nodata
        valid = np.isfinite(source_dem)
        if nodata is not None:
            valid &= source_dem != nodata
        dem = source_dem.copy()
        dem[~valid] = np.nan
        x_size = abs(source.transform.a)
        y_size = abs(source.transform.e)
        pixel_area_m2 = x_size * y_size
        print(f"DEM: {DEM_PATH}")
        print(f"Shape: {source.height} rows x {source.width} columns")
        print(f"CRS: {source.crs}")
        print(f"Pixel size: {x_size:g} m x {y_size:g} m")
        print(f"Minimum elevation: {float(np.nanmin(dem)):.6f} m")
        print(f"Maximum elevation: {float(np.nanmax(dem)):.6f} m")
        print(f"Nodata value: {nodata!r}")
        print(f"Nodata/invalid cell count: {int((~valid).sum())}")
        print(f"Valid cell count: {int(valid.sum())}")

    grid = Grid.from_raster(str(DEM_PATH))
    raw_dem = grid.read_raster(str(DEM_PATH))
    pit_filled = grid.fill_pits(raw_dem)
    depression_filled = grid.fill_depressions(pit_filled)
    conditioned_dem = grid.resolve_flats(depression_filled)
    for stage_name, stage in (
        ("pit fill", pit_filled),
        ("depression fill", depression_filled),
        ("flat resolution", conditioned_dem),
    ):
        stage_values = np.asarray(stage)
        stage_valid = np.isfinite(stage_values)
        difference = stage_values - np.asarray(raw_dem)
        print(
            f"{stage_name}: min={float(np.nanmin(stage_values)):.6f}, "
            f"max={float(np.nanmax(stage_values)):.6f}, "
            f"changed cells vs raw={int(np.count_nonzero(stage_valid & (difference != 0)))}"
        )

    flow_direction = grid.flowdir(conditioned_dem, routing="d8")
    accumulation = grid.accumulation(flow_direction, routing="d8")
    minimum_cells = int(np.ceil(STREAM_AREA_THRESHOLD_KM2 * 1_000_000 / pixel_area_m2))
    stream_mask = valid & (np.asarray(accumulation) >= minimum_cells)
    stream_count = int(stream_mask.sum())
    stream_length_km = _stream_length_km(
        stream_mask,
        np.asarray(flow_direction),
        x_size,
        y_size,
    )
    print(f"Stream-area threshold: {STREAM_AREA_THRESHOLD_KM2:g} km2 ({minimum_cells} cells)")
    print(f"Stream cell count: {stream_count}")
    print(f"Total D8 stream length: {stream_length_km:.6f} km")

    stream_raster = Raster(stream_mask.astype(np.float32), viewfinder=conditioned_dem.viewfinder)
    raw_hand = np.asarray(
        grid.compute_hand(
            flow_direction,
            raw_dem,
            stream_raster,
            nodata_out=-9999.0,
            routing="d8",
        ),
        dtype=np.float32,
    )
    conditioned_hand = np.asarray(
        grid.compute_hand(
            flow_direction,
            conditioned_dem,
            stream_raster,
            nodata_out=-9999.0,
            routing="d8",
        ),
        dtype=np.float32,
    )
    raw_valid = valid & np.isfinite(raw_hand) & (raw_hand != -9999.0)
    conditioned_valid = valid & np.isfinite(conditioned_hand) & (conditioned_hand != -9999.0)
    print("HAND consistency diagnostic (pre-fix reproduction):")
    print("  failing flow direction input: flat-resolved conditioned DEM")
    print("  failing compute_hand DEM input: raw DEM")
    print("  comparison compute_hand DEM input: same flat-resolved DEM")
    _negative_summary("Historical raw-DEM HAND", raw_hand, raw_valid, stream_mask)
    _negative_summary(
        "Conditioned-DEM HAND",
        conditioned_hand,
        conditioned_valid,
        stream_mask,
    )


if __name__ == "__main__":
    main()
