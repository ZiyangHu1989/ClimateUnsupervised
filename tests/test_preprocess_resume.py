from pathlib import Path

import numpy as np

import climate_unsupervised.config as config_module
from climate_unsupervised.preprocess import preprocess


def test_force_reuses_intermediates_when_native_files_were_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    interim = tmp_path / "data/interim"
    interim.mkdir(parents=True)
    (tmp_path / "data/raw/era5").mkdir(parents=True)
    latitude = np.array([-0.25, 0.25], dtype=np.float32)
    longitude = np.array([0.25, 180.25], dtype=np.float32)
    arrays = {
        "temperature_c": np.ones((12, 2, 2), dtype=np.float32),
        "precipitation_mm": np.ones((12, 2, 2), dtype=np.float32),
        "latitude": latitude,
        "longitude": longitude,
    }
    np.savez_compressed(interim / "era5_land_2000_0p5.npz", **arrays)
    config = {
        "era5": {"years": [2000, 2000], "source_resolution_degrees": 0.1,
                 "target_resolution_degrees": 0.5, "delete_native_after_aggregation": True}
    }
    output = preprocess(config, force=True)
    with np.load(output) as data:
        np.testing.assert_allclose(data["longitude"], [0.25, -179.75, 0.25, -179.75])
