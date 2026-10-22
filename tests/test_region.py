import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

import climate_unsupervised.config as config_module
from climate_unsupervised.comprehensive import run_comprehensive_evaluation
from climate_unsupervised.config import run_path
from climate_unsupervised.evaluate import evaluate
from climate_unsupervised.freeze import freeze
from climate_unsupervised.modeling import fit_models
from climate_unsupervised.region import points_in_components, polygon_components


def test_country_components_are_sorted_and_limit_offshore_geometry():
    mainland = [[[110, -40], [150, -40], [150, -10], [110, -10], [110, -40]]]
    tasmania = [[[144, -44], [149, -44], [149, -40], [144, -40], [144, -44]]]
    offshore = [[[100, -5], [101, -5], [101, -4], [100, -4], [100, -5]]]
    payload = {"features": [{"properties": {"ADMIN": "Australia"},
                             "geometry": {"type": "MultiPolygon",
                                          "coordinates": [offshore, tasmania, mainland]}}]}
    components = polygon_components(payload, "Australia")
    mask = points_in_components(
        np.array([135.0, 146.0, 100.5, 174.8, 120.0]),
        np.array([-25.0, -42.0, -4.5, -41.3, -8.0]),
        components, component_count=2,
    )
    np.testing.assert_array_equal(mask, [True, True, False, False, False])


def test_run_path_isolated_and_cannot_escape_project(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    config = {"run": {"output_root": "runs/australia"}}
    assert run_path(config, "results", "metrics") == tmp_path / "runs/australia/results/metrics"
    try:
        run_path({"run": {"output_root": "../outside"}}, "result")
    except ValueError as error:
        assert "project-relative" in str(error)
    else:
        raise AssertionError("An escaping run path was accepted")


def test_regional_pipeline_isolated_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "data/external").mkdir(parents=True)
    rng = np.random.default_rng(9)
    locations = 120
    longitude = np.linspace(115, 150, locations).astype(np.float32)
    latitude = np.linspace(-40, -12, locations).astype(np.float32)
    regime = np.repeat([0, 1, 2], locations // 3)
    profiles = np.stack([
        np.r_[np.linspace(15, 25, 12), np.linspace(20, 50, 12)],
        np.r_[np.linspace(20, 30, 12), np.linspace(60, 120, 12)],
        np.r_[np.linspace(25, 35, 12), np.linspace(120, 220, 12)],
    ])
    features = profiles[regime] + rng.normal(0, .5, (locations, 24))
    features[:, 12:] = np.maximum(features[:, 12:], 0)
    np.savez_compressed(
        tmp_path / "data/processed/climate_features.npz", features=features.astype(np.float32),
        latitude=latitude, longitude=longitude,
        area_weights=np.cos(np.deg2rad(latitude)).astype(np.float32),
        cell_id=np.arange(locations).astype(str), feature_names=np.arange(24).astype(str),
    )
    boundary = {"features": [{"properties": {"ADMIN": "Australia"}, "geometry": {
        "type": "Polygon", "coordinates": [[[110, -45], [155, -45], [155, -10],
                                                 [110, -10], [110, -45]]],
    }}]}
    boundary_path = tmp_path / "data/external/australia.geojson"
    boundary_path.write_text(json.dumps(boundary))
    reference_path = tmp_path / "data/external/koppen_geiger_1991_2020.tif"
    raster = np.tile((np.arange(360) % 3) + 1, (180, 1)).astype(np.uint8)
    with rasterio.open(reference_path, "w", driver="GTiff", height=180, width=360, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(-180, 90, 1, 1),
                       nodata=0) as output:
        output.write(raster, 1)
    config = {
        "project": {"random_seed": 42}, "run": {"output_root": "runs/australia"},
        "region": {"name": "Australia", "country": "Australia", "component_count": 1,
                   "source_features": "data/processed/climate_features.npz",
                   "boundary_path": "data/external/australia.geojson", "boundary_url": "unused"},
        "features": {"pca_variance_target": .95, "training_sample_max": 100,
                     "metric_sample_max": 80},
        "clustering": {"k_min": 2, "k_max": 3, "seeds": [42, 43],
                       "minimum_stability_ari": 0, "minimum_cluster_area_fraction": .01,
                       "stability_sample_max": 100, "spatial_stability_folds": 2,
                       "spatial_block_degrees": 10, "final_n_init": 2,
                       "gmm_covariance_types": ["diag"]},
    }
    from climate_unsupervised.region import prepare_region
    regional_features = prepare_region(config)
    assert regional_features.exists()
    fit_models(config)
    freeze(config)
    evaluate(config)
    report = run_comprehensive_evaluation(config)
    assert report.exists()
    assert not (tmp_path / "results/models/kmeans.joblib").exists()
