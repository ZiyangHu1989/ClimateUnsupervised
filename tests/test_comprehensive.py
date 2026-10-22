import numpy as np
import rasterio
from rasterio.transform import from_origin

from climate_unsupervised.comprehensive import (
    _alignment_audit,
    overlap_diagnostics,
    pairwise_ari_summary,
    predictive_entropy,
    weighted_quantile,
)
from climate_unsupervised.evaluate import sample_raster
from climate_unsupervised.evaluation_plots import _categorical_map


def test_predictive_entropy_has_expected_bounds():
    probabilities = np.array([[1.0, 0.0], [0.5, 0.5]])
    entropy = predictive_entropy(probabilities)
    np.testing.assert_allclose(entropy, [0.0, 1.0])


def test_pairwise_stability_reports_distribution():
    runs = [np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0]), np.array([0, 1, 0, 1])]
    rows, summary = pairwise_ari_summary(runs)
    assert len(rows) == 3
    assert summary["pair_count"] == 3
    assert summary["minimum"] <= summary["mean"] <= summary["maximum"]
    assert summary["standard_deviation"] >= 0


def test_overlap_diagnostics_do_not_require_equal_class_counts():
    clusters = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([4, 4, 5, 5, 6, 6])
    rows, strongest = overlap_diagnostics(clusters, reference, np.ones(6))
    assert len(rows) == 6
    assert strongest == {0: 4, 1: 6}
    match = next(row for row in rows if row["cluster"] == 0 and row["reference_class"] == 4)
    assert np.isclose(match["precision"], 2 / 3)
    assert np.isclose(match["recall"], 1.0)
    assert np.isclose(match["jaccard"], 2 / 3)


def test_weighted_quantile_respects_dominant_weight():
    values = np.array([0.0, 10.0, 20.0])
    weights = np.array([1.0, 100.0, 1.0])
    median = weighted_quantile(values, np.array([0.5]), weights)
    np.testing.assert_allclose(median, [10.0])


def test_spatial_alignment_reports_and_masks_nodata(tmp_path):
    path = tmp_path / "reference.tif"
    values = np.array([[1, 0], [2, 3]], dtype=np.uint8)
    with rasterio.open(path, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(-180, 90, 180, 90), nodata=0) as output:
        output.write(values, 1)
    longitude = np.array([-90.0, 90.0])
    latitude = np.array([45.0, 45.0])
    labels, valid = sample_raster(path, longitude, latitude)
    np.testing.assert_array_equal(labels, [1, 0])
    np.testing.assert_array_equal(valid, [True, False])
    audit = _alignment_audit(path, latitude, longitude, valid)
    assert audit["koppen_raster"]["crs"] == "EPSG:4326"
    assert audit["valid_overlap_cells"] == 1
    assert audit["invalid_or_missing_cells"] == 1


def test_categorical_map_accepts_subset_of_named_classes(tmp_path):
    path = tmp_path / "map.png"
    _categorical_map(
        np.array([115.0, 130.0, 150.0]), np.array([-35.0, -25.0, -15.0]),
        np.array([0, 1, 2]), np.ones(3, dtype=bool), "Australian groups", path,
        ["A Tropical", "B Arid", "C Temperate", "D Continental", "E Polar"],
    )
    assert path.exists() and path.stat().st_size > 0
