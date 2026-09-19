import numpy as np

from climate_unsupervised.evaluate import weighted_ari, weighted_information_scores


def test_weighted_scores_are_perfect_for_relabeling():
    left = np.array([0, 0, 1, 1])
    right = np.array([4, 4, 9, 9])
    weights = np.array([0.5, 1.0, 2.0, 3.0])
    assert np.isclose(weighted_ari(left, right, weights), 1.0)
    scores = weighted_information_scores(left, right, weights)
    assert np.isclose(scores["nmi"], 1.0)
    assert np.isclose(scores["v_measure"], 1.0)
