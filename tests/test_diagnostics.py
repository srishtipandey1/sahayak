"""Tests for human-readable flood diagnostics."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sahayak.flood.config import load_flood_config
from sahayak.flood.diagnostics import (
    DIAGNOSTIC_PERCENTILES,
    corridor_validity_breakdown,
    format_flood_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_flood_config(ROOT / "configs" / "flood.toml")


def _fact(value):
    return {"value": value, "unit": "dB"}


def _pair():
    distributions = {
        metric: {
            str(percentile): (_fact(float(percentile)) if index % 2 else float(percentile))
            for index, percentile in enumerate(DIAGNOSTIC_PERCENTILES)
        }
        for metric in (
            "delta_vv_db",
            "delta_vh_db",
            "z_vv",
            "z_vh",
            "post_vv_db",
            "post_vh_db",
            "baseline_mean_vv_db",
            "baseline_mean_vh_db",
        )
    }
    return {
        "distributions": distributions,
        "raw_otsu_threshold_db": _fact(-19.0),
        "clamped_otsu_threshold_db": -19.0,
        "valid_pixel_count": 10,
        "valid_fraction_in_hand_40m_corridor": _fact(0.9),
        "valid_fraction_whole_tile": 0.8,
        "valid_stream_cell_fraction": _fact(0.7),
        "water": {"valid": 10},
        "disturbed_surface": {"valid": 10},
        "permanent_water": {"valid": 2},
        "permanent_hand_and_slope_percentiles": {
            "before_restriction": {"50": 5},
            "after_restriction": {"50": _fact(4)},
            "slope_before": {"50": 3},
            "pixel_count": 2,
        },
        "class_areas_ha": {"new_water": 1},
        "false_alarms_per_1000_valid_pixels": {},
        "sigma_null_db": {
            "overall": {"vv": 0.2, "vh": _fact(0.3), "valid_pixel_count": 10},
            "by_slope_band": {},
        },
    }


def test_formatter_accepts_scalar_and_wrapped_percentiles() -> None:
    diagnostics = {"real_pair": _pair(), "null_pair": _pair()}

    output = format_flood_diagnostics(diagnostics)

    assert "delta_vv_db | 0.1000 / 0.1000" in output
    assert "Otsu raw/clamped (dB): -19.0000 / -19.0000" in output


def test_corridor_validity_attributes_each_invalid_mask_reason(tmp_path: Path) -> None:
    data = np.zeros((5, 2, 5), dtype=np.float32)
    data[0] = 0.1
    data[1] = 0.01
    data[2] = 35
    data[4] = 1
    data[4, 0, 0] = 0
    data[3, 0, 1] = 1
    data[2, 0, 2] = 10
    data[0, 0, 3] = 10
    data[0, 0, 4] = np.nan
    path = tmp_path / "synthetic.tif"
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 5,
        "count": 5,
        "dtype": "float32",
        "transform": from_origin(0, 20, 10, 10),
        "crs": "EPSG:32645",
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(data)
        for index, name in enumerate(
            (
                "VV gamma0",
                "VH gamma0",
                "Local incidence angle (degrees)",
                "Shadow mask",
                "Data mask",
            ),
            start=1,
        ):
            destination.set_band_description(index, name)

    hand = np.zeros((2, 5), dtype=np.float32)
    streams = np.ones((2, 5), dtype=bool)
    main_river = np.zeros((2, 5), dtype=bool)
    main_river[0, 2:5] = True
    result = corridor_validity_breakdown(
        [path],
        hand,
        streams,
        main_river,
        CONFIG,
    )

    zone = result["per_image"]["synthetic.tif"]["stream_cells"]
    assert zone["invalid_count"] == 5
    assert zone["invalid_reason_counts"] == {
        "dataMask_zero": 1,
        "shadowMask_flagged": 1,
        "incidence_angle_outside_range": 1,
        "physical_ceiling_exceeded": 1,
        "nonfinite_or_nonpositive": 1,
    }
    angle = result["combined"]["main_river_incidence_angle_degrees"]
    assert angle["count"] == 3
    assert angle["below_30_fraction"] == 1 / 3
