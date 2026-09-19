import numpy as np

from climate_unsupervised.modeling import (
    area_resample,
    area_sample,
    normalize_labels,
    spatial_block_folds,
    transformed_features,
)


def test_area_sample_is_deterministic():
    weights = np.arange(1, 21, dtype=float)
    np.testing.assert_array_equal(area_sample(weights, 8, 42), area_sample(weights, 8, 42))


def test_area_resample_is_deterministic_and_allows_repeats():
    weights = np.array([1.0, 100.0])
    first = area_resample(weights, 20, 42)
    np.testing.assert_array_equal(first, area_resample(weights, 20, 42))
    assert len(first) == 20
    assert len(np.unique(first)) < len(first)


def test_spatial_block_folds_keep_nearby_cells_together():
    latitude = np.array([1.0, 2.0, 21.0])
    longitude = np.array([1.0, 2.0, 21.0])
    folds = spatial_block_folds(latitude, longitude, block_degrees=10, folds=5)
    assert folds[0] == folds[1]
    assert folds[0] != folds[2]


def test_coordinates_cannot_enter_feature_transform():
    transformed = transformed_features(np.ones((3, 24)))
    assert transformed.shape == (3, 24)
    np.testing.assert_allclose(transformed[:, 12:], np.log(2))
    try:
        transformed_features(np.ones((3, 26)))
    except ValueError as error:
        assert "24" in str(error)
    else:
        raise AssertionError("Non-climate columns were accepted")


def test_label_normalization_orders_by_weighted_area():
    labels, mapping = normalize_labels(np.array([5, 5, 2]), np.array([1.0, 1.0, 3.0]))
    assert mapping == {2: 0, 5: 1}
    np.testing.assert_array_equal(labels, [1, 1, 0])
