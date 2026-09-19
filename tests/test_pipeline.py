import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import climate_unsupervised.config as config_module
from climate_unsupervised.evaluate import evaluate
from climate_unsupervised.freeze import freeze
from climate_unsupervised.modeling import fit_models
from climate_unsupervised.visualize import visualize


def test_offline_fit_freeze_evaluate_and_visualize(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    for path in ("data/processed", "data/external", "results/models", "results/metrics",
                 "results/manifests", "results/figures"):
        (tmp_path / path).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    locations = 120
    regime = np.repeat([0, 1, 2], locations // 3)
    base = np.stack([
        np.r_[np.linspace(-5, 15, 12), np.linspace(20, 80, 12)],
        np.r_[np.linspace(10, 30, 12), np.linspace(80, 180, 12)],
        np.r_[np.linspace(20, 35, 12), np.linspace(180, 320, 12)],
    ])
    features = base[regime] + rng.normal(0, 1, (locations, 24))
    features[:, 12:] = np.maximum(features[:, 12:], 0)
    latitude = np.linspace(-59.5, 59.5, locations).astype(np.float32)
    longitude = np.linspace(-179.5, 179.5, locations).astype(np.float32)
    weights = np.cos(np.deg2rad(latitude)).astype(np.float32)
    np.savez_compressed(tmp_path / "data/processed/climate_features.npz", features=features.astype(np.float32),
                        latitude=latitude, longitude=longitude, area_weights=weights,
                        cell_id=np.arange(locations).astype(str), feature_names=np.arange(24).astype(str))

    config = {
        "project": {"random_seed": 42},
        "features": {"pca_variance_target": 0.95, "training_sample_max": 100, "metric_sample_max": 80},
        "clustering": {"k_min": 2, "k_max": 3, "seeds": [42, 43],
                       "minimum_stability_ari": 0.0, "minimum_cluster_area_fraction": 0.01,
                       "gmm_covariance_type": "diag"},
    }
    fit_models(config)
    frozen = freeze(config)
    assert frozen.exists()

    raster_path = tmp_path / "data/external/koppen_geiger_1991_2020.tif"
    raster = np.tile(np.repeat(np.arange(1, 4), 120), (180, 1)).astype(np.uint8)
    with rasterio.open(raster_path, "w", driver="GTiff", height=180, width=360, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(-180, 90, 1, 1), nodata=0) as output:
        output.write(raster, 1)
    external = evaluate(config)
    payload = json.loads(external.read_text())
    assert set(payload) == {"kmeans", "gmm"}
    figures = visualize(config)
    assert len(figures) >= 9
    assert all(path.exists() and path.stat().st_size > 0 for path in figures)
