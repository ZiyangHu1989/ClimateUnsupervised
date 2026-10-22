from __future__ import annotations

from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .artifacts import manifest, write_json
from .config import config_hash, run_path


def area_sample(weights: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Draw a unique area-biased subset for inexpensive metric estimation."""
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Sampling weights must be finite and positive")
    size = min(int(size), len(weights))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(weights), size=size, replace=False, p=weights / weights.sum()))


def area_resample(weights: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Draw with replacement so the empirical sample follows geographic area.

    Weighted sampling without replacement becomes biased when a large fraction
    of the population is selected: high-weight cells can appear only once. A
    bootstrap sample is appropriate for estimators that do not accept
    ``sample_weight`` (PCA and GaussianMixture).
    """
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Sampling weights must be finite and positive")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(weights), size=int(size), replace=True, p=weights / weights.sum()))


def transformed_features(features: np.ndarray) -> np.ndarray:
    transformed = np.asarray(features, dtype=np.float64).copy()
    if transformed.shape[1] != 24:
        raise ValueError("Expected exactly 24 climate features")
    if np.any(transformed[:, 12:] < 0):
        raise ValueError("Precipitation cannot be negative")
    transformed[:, 12:] = np.log1p(transformed[:, 12:])
    return transformed


def normalize_labels(labels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    labels = np.asarray(labels)
    totals = {int(label): float(weights[labels == label].sum()) for label in np.unique(labels)}
    mapping = {old: new for new, (old, _) in enumerate(sorted(totals.items(), key=lambda item: -item[1]))}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int16), mapping


def _stability(label_runs: list[np.ndarray]) -> float:
    scores = [adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)]
    return float(np.mean(scores)) if scores else 1.0


def _minimum_area_fraction(labels: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    return float(min(weights[labels == label].sum() / total for label in np.unique(labels)))


def spatial_block_folds(latitude: np.ndarray, longitude: np.ndarray, block_degrees: float, folds: int) -> np.ndarray:
    """Assign nearby cells to deterministic geographic blocks and folds."""
    if block_degrees <= 0 or folds < 2:
        raise ValueError("Spatial stability requires positive blocks and at least two folds")
    lat_block = np.floor((np.asarray(latitude) + 90.0) / block_degrees).astype(np.int64)
    lon_block = np.floor((np.asarray(longitude) + 180.0) / block_degrees).astype(np.int64)
    return ((lat_block * 37 + lon_block * 101) % folds).astype(np.int8)


def _spatial_stability(
    reduced: np.ndarray,
    weights: np.ndarray,
    metric_idx: np.ndarray,
    fold_ids: np.ndarray,
    sample_size: int,
    seed: int,
    factory,
) -> float:
    """Compare predictions after leaving out disjoint sets of spatial blocks."""
    label_runs: list[np.ndarray] = []
    for fold in np.unique(fold_ids):
        pool = np.flatnonzero(fold_ids != fold)
        local = area_resample(weights[pool], min(sample_size, len(pool)), seed + int(fold))
        model = factory(seed + int(fold))
        model.fit(reduced[pool[local]])
        label_runs.append(model.predict(reduced[metric_idx]))
    return _stability(label_runs)


def fit_models(config: dict, force: bool = False) -> Path:
    feature_path = run_path(config, "data", "processed", "climate_features.npz")
    if not feature_path.exists():
        raise FileNotFoundError("Run preprocess before fit")
    with np.load(feature_path) as data:
        physical = data["features"]
        weights = data["area_weights"].astype(np.float64)
        latitude, longitude = data["latitude"], data["longitude"]

    seed = int(config["project"]["random_seed"])
    feature_config = config["features"]
    cluster_config = config["clustering"]
    train_size = min(int(feature_config["training_sample_max"]), len(weights))
    train_idx = area_resample(weights, train_size, seed)
    metric_idx = area_sample(weights, int(feature_config["metric_sample_max"]), seed + 1)
    stability_size = min(int(cluster_config.get("stability_sample_max", train_size)), len(weights))

    transformed = transformed_features(physical)
    # Scaling can be exactly area weighted. PCA cannot accept sample weights,
    # so its training bootstrap follows the same area distribution.
    scaler = StandardScaler().fit(transformed, sample_weight=weights)
    scaled = scaler.transform(transformed)
    full_pca = PCA(random_state=seed).fit(scaled[train_idx])
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    components = int(np.searchsorted(cumulative, float(feature_config["pca_variance_target"])) + 1)
    pca = PCA(n_components=components, random_state=seed).fit(scaled[train_idx])
    reduced = pca.transform(scaled).astype(np.float32)

    models_dir = run_path(config, "results", "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(pca, models_dir / "pca.joblib")
    np.save(models_dir / "pca_features.npy", reduced)

    k_values = range(int(cluster_config["k_min"]), int(cluster_config["k_max"]) + 1)
    seeds = [int(value) for value in cluster_config["seeds"]]
    min_stability = float(cluster_config["minimum_stability_ari"])
    min_area = float(cluster_config["minimum_cluster_area_fraction"])
    final_n_init = int(cluster_config.get("final_n_init", 10))
    kmeans_rows: list[dict] = []
    kmeans_models: dict[int, tuple[KMeans, np.ndarray]] = {}
    for k in k_values:
        model = KMeans(n_clusters=k, n_init=final_n_init, max_iter=500, random_state=seed)
        labels = model.fit_predict(reduced, sample_weight=weights)
        stability_runs: list[np.ndarray] = []
        for run_seed in seeds:
            bootstrap_idx = area_resample(weights, stability_size, seed + 1000 + run_seed)
            bootstrap = KMeans(n_clusters=k, n_init=1, max_iter=500, random_state=run_seed)
            bootstrap.fit(reduced[bootstrap_idx])
            stability_runs.append(bootstrap.predict(reduced[metric_idx]))
        row = {
            "k": k,
            "silhouette": float(silhouette_score(reduced[metric_idx], labels[metric_idx])),
            "davies_bouldin": float(davies_bouldin_score(reduced[metric_idx], labels[metric_idx])),
            "stability_ari": _stability(stability_runs),
            "stability_kind": "area_bootstrap",
            "minimum_cluster_area_fraction": _minimum_area_fraction(labels, weights),
            "inertia": float(model.inertia_), "best_seed": seed,
        }
        row["eligible"] = row["stability_ari"] >= min_stability and row["minimum_cluster_area_fraction"] >= min_area
        kmeans_rows.append(row)
        kmeans_models[k] = (model, labels)

    eligible_kmeans = [row for row in kmeans_rows if row["eligible"]]
    if not eligible_kmeans:
        raise RuntimeError("No K-means candidate met stability and minimum-area safeguards")
    selected_kmeans = max(eligible_kmeans, key=lambda row: (row["silhouette"], -row["k"]))
    kmeans_model, kmeans_labels_raw = kmeans_models[selected_kmeans["k"]]
    spatial_folds = int(cluster_config.get("spatial_stability_folds", 0))
    if spatial_folds >= 2:
        fold_ids = spatial_block_folds(latitude, longitude,
                                       float(cluster_config.get("spatial_block_degrees", 10.0)), spatial_folds)
        selected_kmeans["spatial_block_stability_ari"] = _spatial_stability(
            reduced, weights, metric_idx, fold_ids, stability_size, seed + 3000,
            lambda run_seed: KMeans(n_clusters=selected_kmeans["k"], n_init=1,
                                    max_iter=500, random_state=run_seed),
        )
    kmeans_labels, kmeans_mapping = normalize_labels(kmeans_labels_raw, weights)
    kmeans_model.normalized_label_mapping_ = kmeans_mapping
    joblib.dump(kmeans_model, models_dir / "kmeans.joblib")
    np.save(models_dir / "kmeans_labels.npy", kmeans_labels)

    gmm_rows: list[dict] = []
    gmm_models: dict[tuple[str, int], tuple[GaussianMixture, np.ndarray, np.ndarray]] = {}
    sample = reduced[train_idx]
    covariance_types = cluster_config.get("gmm_covariance_types", [cluster_config.get("gmm_covariance_type", "diag")])
    for covariance_type in covariance_types:
        for k in k_values:
            model = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                    n_init=final_n_init, max_iter=500, random_state=seed, reg_covar=1e-6)
            model.fit(sample)
            labels = model.predict(reduced)
            probabilities = model.predict_proba(reduced).max(axis=1)
            stability_runs: list[np.ndarray] = []
            for run_seed in seeds:
                bootstrap_idx = area_resample(weights, stability_size, seed + 2000 + run_seed)
                bootstrap = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                            n_init=1, max_iter=500, random_state=run_seed, reg_covar=1e-6)
                bootstrap.fit(reduced[bootstrap_idx])
                stability_runs.append(bootstrap.predict(reduced[metric_idx]))
            row = {
                "k": k, "covariance_type": covariance_type,
                "bic": float(model.bic(sample)), "aic": float(model.aic(sample)),
                "silhouette": float(silhouette_score(reduced[metric_idx], labels[metric_idx])),
                "davies_bouldin": float(davies_bouldin_score(reduced[metric_idx], labels[metric_idx])),
                "stability_ari": _stability(stability_runs),
                "stability_kind": "area_bootstrap",
                "minimum_cluster_area_fraction": _minimum_area_fraction(labels, weights),
                "best_seed": seed,
            }
            row["eligible"] = row["stability_ari"] >= min_stability and row["minimum_cluster_area_fraction"] >= min_area
            gmm_rows.append(row)
            gmm_models[(covariance_type, k)] = (model, labels, probabilities)

    eligible_gmm = [row for row in gmm_rows if row["eligible"]]
    if not eligible_gmm:
        raise RuntimeError("No GMM candidate met stability and minimum-area safeguards")
    # BIC estimates density fit and kept decreasing as components were added in
    # the baseline. For interpretable hard climate regions, select partition
    # quality after stability/size safeguards; retain BIC as a diagnostic.
    selected_gmm = max(eligible_gmm, key=lambda row: (row["silhouette"], -row["davies_bouldin"], -row["k"]))
    gmm_key = (selected_gmm["covariance_type"], selected_gmm["k"])
    gmm_model, gmm_labels_raw, gmm_probabilities = gmm_models[gmm_key]
    if spatial_folds >= 2:
        selected_gmm["spatial_block_stability_ari"] = _spatial_stability(
            reduced, weights, metric_idx, fold_ids, stability_size, seed + 4000,
            lambda run_seed: GaussianMixture(
                n_components=selected_gmm["k"], covariance_type=selected_gmm["covariance_type"],
                n_init=1, max_iter=500, random_state=run_seed, reg_covar=1e-6,
            ),
        )
    gmm_labels, gmm_mapping = normalize_labels(gmm_labels_raw, weights)
    gmm_model.normalized_label_mapping_ = gmm_mapping
    joblib.dump(gmm_model, models_dir / "gmm.joblib")
    np.save(models_dir / "gmm_labels.npy", gmm_labels)
    np.save(models_dir / "gmm_confidence.npy", gmm_probabilities.astype(np.float32))

    metrics_path = run_path(config, "results", "metrics", "internal.json")
    write_json(metrics_path, {
        "pca_components": components,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "training_sample_size": len(train_idx), "training_sampling": "area_bootstrap_with_replacement",
        "metric_sample_size": len(metric_idx), "stability_sample_size": stability_size,
        "kmeans": {"selected": selected_kmeans, "candidates": kmeans_rows, "label_mapping": kmeans_mapping},
        "gmm": {"selected": selected_gmm, "candidates": gmm_rows, "label_mapping": gmm_mapping},
    })
    write_json(run_path(config, "results", "manifests", "fit.json"), manifest(
        "fit", config_hash(config), [feature_path], metrics=str(metrics_path),
        pca_components=components, selected_kmeans_k=selected_kmeans["k"], selected_gmm_k=selected_gmm["k"],
        random_seeds=seeds, stability_kind="area_bootstrap",
        spatial_stability_folds=spatial_folds,
    ))
    return metrics_path
