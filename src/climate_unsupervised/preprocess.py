from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import xarray as xr

from .artifacts import manifest, write_json
from .config import config_hash, project_path


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def kelvin_to_celsius(values: np.ndarray) -> np.ndarray:
    return np.asarray(values) - 273.15


def precipitation_m_per_day_to_mm_month(values: np.ndarray, year: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] != 12:
        raise ValueError("Precipitation must have 12 monthly values on axis 0")
    days = np.asarray([calendar.monthrange(year, month)[1] for month in range(1, 13)])
    shape = (12,) + (1,) * (values.ndim - 1)
    return values * 1000.0 * days.reshape(shape)


def area_weights(latitude: np.ndarray) -> np.ndarray:
    weights = np.cos(np.deg2rad(np.asarray(latitude, dtype=np.float64)))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Latitude produced invalid area weights")
    return weights.astype(np.float32)


def _coordinate_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in dataset.coords:
            return name
    raise ValueError(f"Missing coordinate; expected one of {candidates}")


def _variable(dataset: xr.Dataset, candidates: tuple[str, ...]) -> xr.DataArray:
    for name in candidates:
        if name in dataset.data_vars:
            return dataset[name]
    raise ValueError(f"Missing variable; expected one of {candidates}")


def _open_dataset(path: Path) -> xr.Dataset:
    try:
        return xr.open_dataset(path)
    except Exception as netcdf_error:
        try:
            return xr.open_dataset(path, engine="cfgrib")
        except Exception as grib_error:
            raise RuntimeError(f"Could not open {path} as NetCDF or GRIB") from grib_error


def aggregate_year(path: Path, year: int, source_resolution: float, target_resolution: float) -> dict[str, np.ndarray]:
    ratio = target_resolution / source_resolution
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor):
        raise ValueError("Target resolution must be an integer multiple of source resolution")
    with _open_dataset(path) as dataset:
        lat_name = _coordinate_name(dataset, ("latitude", "lat"))
        lon_name = _coordinate_name(dataset, ("longitude", "lon"))
        temperature = _variable(dataset, ("t2m", "2m_temperature"))
        precipitation = _variable(dataset, ("tp", "total_precipitation"))
        time_name = next((name for name in ("valid_time", "time", "date") if name in temperature.dims), None)
        if time_name is None or temperature.sizes[time_name] != 12:
            raise ValueError(f"{path} must contain exactly 12 monthly time steps")
        temperature = temperature.transpose(time_name, lat_name, lon_name)
        precipitation = precipitation.transpose(time_name, lat_name, lon_name)
        if factor > 1:
            temperature = temperature.coarsen({lat_name: factor, lon_name: factor}, boundary="trim").mean(skipna=True)
            precipitation = precipitation.coarsen({lat_name: factor, lon_name: factor}, boundary="trim").mean(skipna=True)
        return {
            "temperature_c": kelvin_to_celsius(temperature.values).astype(np.float32),
            "precipitation_mm": precipitation_m_per_day_to_mm_month(precipitation.values, year).astype(np.float32),
            "latitude": np.asarray(temperature[lat_name].values, dtype=np.float32),
            "longitude": np.asarray(temperature[lon_name].values, dtype=np.float32),
        }


def preprocess(config: dict, force: bool = False) -> Path:
    era5 = config["era5"]
    first_year, last_year = map(int, era5["years"])
    interim_dir = project_path("data", "interim")
    raw_dir = project_path("data", "raw", "era5")
    yearly_paths: list[Path] = []
    for year in range(first_year, last_year + 1):
        interim_path = interim_dir / f"era5_land_{year}_0p5.npz"
        if not interim_path.exists() or force:
            candidates = sorted(raw_dir.glob(f"era5_land_{year}.*"))
            if not candidates:
                raise FileNotFoundError(f"Missing downloaded ERA5-Land year {year} in {raw_dir}")
            arrays = aggregate_year(
                candidates[0], year, float(era5["source_resolution_degrees"]),
                float(era5["target_resolution_degrees"]),
            )
            interim_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(interim_path, **arrays)
            if era5.get("delete_native_after_aggregation", True):
                candidates[0].unlink()
        yearly_paths.append(interim_path)

    temp_sum = precip_sum = None
    temp_count = precip_count = None
    latitude = longitude = None
    for path in yearly_paths:
        with np.load(path) as year_data:
            if latitude is None:
                latitude = year_data["latitude"]
                longitude = year_data["longitude"]
                shape = year_data["temperature_c"].shape
                temp_sum = np.zeros(shape, dtype=np.float64)
                precip_sum = np.zeros(shape, dtype=np.float64)
                temp_count = np.zeros(shape, dtype=np.int16)
                precip_count = np.zeros(shape, dtype=np.int16)
            elif not (np.array_equal(latitude, year_data["latitude"]) and np.array_equal(longitude, year_data["longitude"])):
                raise ValueError(f"Grid mismatch in {path}")
            temperature = year_data["temperature_c"]
            precipitation = year_data["precipitation_mm"]
            temp_valid = np.isfinite(temperature)
            precip_valid = np.isfinite(precipitation)
            temp_sum += np.where(temp_valid, temperature, 0)
            precip_sum += np.where(precip_valid, precipitation, 0)
            temp_count += temp_valid
            precip_count += precip_valid

    assert latitude is not None and longitude is not None
    temperature = temp_sum / np.maximum(temp_count, 1)
    precipitation = precip_sum / np.maximum(precip_count, 1)
    valid = (temp_count == len(yearly_paths)).all(axis=0) & (precip_count == len(yearly_paths)).all(axis=0)
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    lat = lat_grid[valid]
    lon = lon_grid[valid]
    features = np.concatenate([temperature[:, valid].T, precipitation[:, valid].T], axis=1).astype(np.float32)
    names = np.asarray([f"temp_{month}_c" for month in MONTHS] + [f"precip_{month}_mm" for month in MONTHS])
    cell_ids = np.asarray([f"{a:.3f}_{b:.3f}" for a, b in zip(lat, lon)])
    output = project_path("data", "processed", "climate_features.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, latitude=lat, longitude=lon,
                        area_weights=area_weights(lat), cell_id=cell_ids, feature_names=names)
    write_json(project_path("results", "manifests", "preprocess.json"), manifest(
        "preprocess", config_hash(config), yearly_paths, output=str(output), locations=len(features),
        features=24, years=[first_year, last_year], resolution_degrees=era5["target_resolution_degrees"],
        units={"temperature": "degree_Celsius", "precipitation": "mm/month"},
    ))
    return output
