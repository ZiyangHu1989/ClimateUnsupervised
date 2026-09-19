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
from .config import config_hash, project_path


def area_sample(weights: np.ndarray, size: int, seed: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Sampling weights must be finite and positive")
    size = min(int(size), len(weights))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(weights), size=size, replace=False, p=weights / weights.sum()))


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


def fit_models(config: dict, force: bool = False) -> Path:
    feature_path = project_path("data", "processed", "climate_features.npz")
    if not feature_path.exists():
        raise FileNotFoundError("Run preprocess before fit")
    with np.load(feature_path) as data:
        physical = data["features"]
        weights = data["area_weights"].astype(np.float64)

    seed = int(config["project"]["random_seed"])
    feature_config = config["features"]
    cluster_config = config["clustering"]
    train_idx = area_sample(weights, int(feature_config["training_sample_max"]), seed)
    metric_idx = area_sample(weights, int(feature_config["metric_sample_max"]), seed + 1)

    transformed = transformed_features(physical)
    scaler = StandardScaler().fit(transformed[train_idx])
    scaled = scaler.transform(transformed)
    full_pca = PCA(random_state=seed).fit(scaled[train_idx])
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    components = int(np.searchsorted(cumulative, float(feature_config["pca_variance_target"])) + 1)
    pca = PCA(n_components=components, random_state=seed).fit(scaled[train_idx])
    reduced = pca.transform(scaled).astype(np.float32)

    models_dir = project_path("results", "models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.joblib")
    joblib.dump(pca, models_dir / "pca.joblib")
    np.save(models_dir / "pca_features.npy", reduced)

    k_values = range(int(cluster_config["k_min"]), int(cluster_config["k_max"]) + 1)
    seeds = [int(value) for value in cluster_config["seeds"]]
    min_stability = float(cluster_config["minimum_stability_ari"])
    min_area = float(cluster_config["minimum_cluster_area_fraction"])
    kmeans_rows: list[dict] = []
    kmeans_models: dict[int, tuple[KMeans, np.ndarray]] = {}
    for k in k_values:
        runs: list[tuple[KMeans, np.ndarray]] = []
        for run_seed in seeds:
            model = KMeans(n_clusters=k, n_init=1, max_iter=500, random_state=run_seed)
            labels = model.fit_predict(reduced, sample_weight=weights)
            runs.append((model, labels))
        best_model, best_labels = min(runs, key=lambda item: item[0].inertia_)
        row = {
            "k": k,
            "silhouette": float(silhouette_score(reduced[metric_idx], best_labels[metric_idx])),
            "davies_bouldin": float(davies_bouldin_score(reduced[metric_idx], best_labels[metric_idx])),
            "stability_ari": _stability([labels[metric_idx] for _, labels in runs]),
            "minimum_cluster_area_fraction": _minimum_area_fraction(best_labels, weights),
            "inertia": float(best_model.inertia_), "best_seed": int(best_model.random_state),
        }
        row["eligible"] = row["stability_ari"] >= min_stability and row["minimum_cluster_area_fraction"] >= min_area
        kmeans_rows.append(row)
        kmeans_models[k] = (best_model, best_labels)

    eligible_kmeans = [row for row in kmeans_rows if row["eligible"]]
    if not eligible_kmeans:
        raise RuntimeError("No K-means candidate met stability and minimum-area safeguards")
    selected_kmeans = max(eligible_kmeans, key=lambda row: (row["silhouette"], -row["k"]))
    kmeans_model, kmeans_labels_raw = kmeans_models[selected_kmeans["k"]]
    kmeans_labels, kmeans_mapping = normalize_labels(kmeans_labels_raw, weights)
    joblib.dump(kmeans_model, models_dir / "kmeans.joblib")
    np.save(models_dir / "kmeans_labels.npy", kmeans_labels)

    gmm_rows: list[dict] = []
    gmm_models: dict[int, tuple[GaussianMixture, np.ndarray, np.ndarray]] = {}
    sample = reduced[train_idx]
    for k in k_values:
        runs: list[tuple[GaussianMixture, np.ndarray, np.ndarray]] = []
        for run_seed in seeds:
            model = GaussianMixture(n_components=k, covariance_type=cluster_config["gmm_covariance_type"],
                                    n_init=1, max_iter=500, random_state=run_seed, reg_covar=1e-6)
            model.fit(sample)
            labels = model.predict(reduced)
            probabilities = model.predict_proba(reduced).max(axis=1)
            runs.append((model, labels, probabilities))
        best_model, best_labels, best_probabilities = min(runs, key=lambda item: item[0].bic(sample))
        row = {
            "k": k, "bic": float(best_model.bic(sample)), "aic": float(best_model.aic(sample)),
            "silhouette": float(silhouette_score(reduced[metric_idx], best_labels[metric_idx])),
            "davies_bouldin": float(davies_bouldin_score(reduced[metric_idx], best_labels[metric_idx])),
            "stability_ari": _stability([labels[metric_idx] for _, labels, _ in runs]),
            "minimum_cluster_area_fraction": _minimum_area_fraction(best_labels, weights),
            "best_seed": int(best_model.random_state),
        }
        row["eligible"] = row["stability_ari"] >= min_stability and row["minimum_cluster_area_fraction"] >= min_area
        gmm_rows.append(row)
        gmm_models[k] = (best_model, best_labels, best_probabilities)

    eligible_gmm = [row for row in gmm_rows if row["eligible"]]
    if not eligible_gmm:
        raise RuntimeError("No GMM candidate met stability and minimum-area safeguards")
    selected_gmm = min(eligible_gmm, key=lambda row: (row["bic"], row["k"]))
    gmm_model, gmm_labels_raw, gmm_probabilities = gmm_models[selected_gmm["k"]]
    gmm_labels, gmm_mapping = normalize_labels(gmm_labels_raw, weights)
    joblib.dump(gmm_model, models_dir / "gmm.joblib")
    np.save(models_dir / "gmm_labels.npy", gmm_labels)
    np.save(models_dir / "gmm_confidence.npy", gmm_probabilities.astype(np.float32))

    metrics_path = project_path("results", "metrics", "internal.json")
    write_json(metrics_path, {
        "pca_components": components,
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "training_sample_size": len(train_idx), "metric_sample_size": len(metric_idx),
        "kmeans": {"selected": selected_kmeans, "candidates": kmeans_rows, "label_mapping": kmeans_mapping},
        "gmm": {"selected": selected_gmm, "candidates": gmm_rows, "label_mapping": gmm_mapping},
    })
    write_json(project_path("results", "manifests", "fit.json"), manifest(
        "fit", config_hash(config), [feature_path], metrics=str(metrics_path),
        pca_components=components, selected_kmeans_k=selected_kmeans["k"], selected_gmm_k=selected_gmm["k"],
        random_seeds=seeds,
    ))
    return metrics_path
