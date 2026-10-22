from climate_unsupervised.multiscale import build_shortlist


def _candidate(method, family, k, silhouette, stability, eligible=True):
    return {"method": method, "family": family, "k": k, "silhouette": silhouette,
            "davies_bouldin": 1.0, "eligible": eligible,
            "bootstrap_stability": {"mean": stability}, "spatial_stability": {"mean": stability}}


def test_shortlist_rejects_coarse_or_unsupported_resolutions():
    candidates = [
        _candidate("kmeans", "kmeans", 3, .36, .90),
        _candidate("gmm_diag", "gmm", 3, .34, .88),
        _candidate("kmeans", "kmeans", 4, .31, .91),
        _candidate("gmm_diag", "gmm", 4, .30, .89),
        _candidate("kmeans", "kmeans", 5, .20, .95),
        _candidate("gmm_diag", "gmm", 5, .19, .94),
        _candidate("kmeans", "kmeans", 6, .33, .93),
    ]
    result = build_shortlist(candidates, {3: .75, 4: .65, 5: .80, 6: .90}, .80, .80, .60)
    assert [row["k"] for row in result] == [3, 4, 6]
    assert result[0]["representative_method"] == "kmeans"


def test_shortlist_tie_breaks_stability_before_silhouette():
    candidates = [
        _candidate("kmeans", "kmeans", 4, .40, .85),
        _candidate("birch", "birch", 4, .36, .95),
    ]
    result = build_shortlist(candidates, {4: .80}, .80, .80, .60)
    assert result[0]["representative_method"] == "birch"


def test_shortlist_ignores_k_with_no_eligible_method():
    candidates = [_candidate("kmeans", "kmeans", 7, .30, .50, eligible=False)]
    assert build_shortlist(candidates, {}, .80, .80, .60) == []
