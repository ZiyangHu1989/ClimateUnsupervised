import numpy as np

from climate_unsupervised.evaluate import koppen_primary_groups, weighted_ari, weighted_information_scores


def test_weighted_scores_are_perfect_for_relabeling():
    left = np.array([0, 0, 1, 1])
    right = np.array([4, 4, 9, 9])
    weights = np.array([0.5, 1.0, 2.0, 3.0])
    assert np.isclose(weighted_ari(left, right, weights), 1.0)
    scores = weighted_information_scores(left, right, weights)
    assert np.isclose(scores["nmi"], 1.0)
    assert np.isclose(scores["homogeneity"], 1.0)
    assert np.isclose(scores["completeness"], 1.0)
    assert np.isclose(scores["v_measure"], 1.0)


def test_koppen_primary_group_mapping():
    subclasses = np.array([1, 3, 4, 7, 8, 16, 17, 28, 29, 30, 0])
    np.testing.assert_array_equal(koppen_primary_groups(subclasses), [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1])
