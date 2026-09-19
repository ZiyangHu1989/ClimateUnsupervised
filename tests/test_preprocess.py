import numpy as np
import xarray as xr

from climate_unsupervised.preprocess import (
    area_weights,
    kelvin_to_celsius,
    precipitation_m_per_day_to_mm_month,
    aggregate_year,
)


def test_temperature_conversion():
    np.testing.assert_allclose(kelvin_to_celsius(np.array([273.15, 300.0])), [0.0, 26.85])


def test_precipitation_conversion_handles_leap_year():
    daily = np.full((12, 1), 0.001)
    normal = precipitation_m_per_day_to_mm_month(daily, 1999)[:, 0]
    leap = precipitation_m_per_day_to_mm_month(daily, 2000)[:, 0]
    assert normal[1] == 28
    assert leap[1] == 29
    assert normal[0] == leap[0] == 31


def test_area_weights():
    weights = area_weights(np.array([-60.0, 0.0, 60.0]))
    np.testing.assert_allclose(weights, [0.5, 1.0, 0.5], atol=1e-6)


def test_aggregate_year_coarsens_and_converts_units(tmp_path):
    temperature = np.full((12, 10, 10), 273.15)
    precipitation = np.full((12, 10, 10), 0.001)
    dataset = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), temperature),
         "tp": (("time", "latitude", "longitude"), precipitation)},
        coords={"time": np.arange(12), "latitude": np.arange(10) / 10,
                "longitude": np.arange(10) / 10},
    )
    path = tmp_path / "year.nc"
    dataset.to_netcdf(path)
    result = aggregate_year(path, 2000, 0.1, 0.5)
    assert result["temperature_c"].shape == (12, 2, 2)
    np.testing.assert_allclose(result["temperature_c"], 0, atol=1e-5)
    np.testing.assert_allclose(result["precipitation_mm"][1], 29)
