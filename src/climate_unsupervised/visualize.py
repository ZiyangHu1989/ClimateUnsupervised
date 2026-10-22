from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import run_path
from .preprocess import MONTHS


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _weighted_profiles(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    profiles = []
    for label in np.unique(labels):
        mask = labels == label
        profiles.append(np.average(features[mask], axis=0, weights=weights[mask]))
    return np.asarray(profiles)


def visualize(config: dict) -> list[Path]:
    feature_path = run_path(config, "data", "processed", "climate_features.npz")
    internal_path = run_path(config, "results", "metrics", "internal.json")
    if not feature_path.exists() or not internal_path.exists():
        raise FileNotFoundError("Run preprocess and fit before visualize")
    with np.load(feature_path) as data:
        features = data["features"]
        latitude, longitude, weights = data["latitude"], data["longitude"], data["area_weights"]
    reduced = np.load(run_path(config, "results", "models", "pca_features.npy"))
    metrics = json.loads(internal_path.read_text(encoding="utf-8"))
    output_dir = run_path(config, "results", "figures")
    outputs: list[Path] = []

    pca = __import__("joblib").load(run_path(config, "results", "models", "pca.joblib"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker="o")
    ax.set(xlabel="Principal components", ylabel="Cumulative explained variance", title="PCA explained variance", ylim=(0, 1.02))
    outputs.append(output_dir / "pca-explained-variance.png"); _save(fig, outputs[-1])

    max_points = min(20000, len(features))
    sample = np.random.default_rng(int(config["project"]["random_seed"])).choice(len(features), max_points, replace=False)
    for method in ("kmeans", "gmm"):
        labels = np.load(run_path(config, "results", "models", f"{method}_labels.npy"))
        fig, ax = plt.subplots(figsize=(7, 5))
        pc2 = reduced[sample, 1] if reduced.shape[1] > 1 else np.zeros(len(sample))
        ax.scatter(reduced[sample, 0], pc2, c=labels[sample], s=2, cmap="tab20", rasterized=True)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{method.upper()} clusters in PCA space")
        outputs.append(output_dir / f"{method}-pca.png"); _save(fig, outputs[-1])

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.scatter(longitude, latitude, c=labels, s=1, cmap="tab20", rasterized=True)
        padding = float(config.get("region", {}).get("map_padding_degrees", 1.0))
        title_region = config.get("region", {}).get("name", "Global")
        ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{title_region} {method.upper()} climate regions",
               xlim=(float(longitude.min()) - padding, float(longitude.max()) + padding),
               ylim=(float(latitude.min()) - padding, float(latitude.max()) + padding))
        outputs.append(output_dir / f"{method}-map.png"); _save(fig, outputs[-1])

        profiles = _weighted_profiles(features, labels, weights)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for cluster, profile in enumerate(profiles):
            axes[0].plot(MONTHS, profile[:12], label=str(cluster))
            axes[1].plot(MONTHS, profile[12:], label=str(cluster))
        axes[0].set(title="Temperature profiles", ylabel="°C")
        axes[1].set(title="Precipitation profiles", ylabel="mm/month")
        axes[1].legend(title="Cluster", ncol=2, fontsize="small")
        outputs.append(output_dir / f"{method}-profiles.png"); _save(fig, outputs[-1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    candidates = metrics["kmeans"]["candidates"]
    axes[0].plot([row["k"] for row in candidates], [row["silhouette"] for row in candidates], marker="o", label="Silhouette")
    axes[0].plot([row["k"] for row in candidates], [row["stability_ari"] for row in candidates], marker="s", label="Bootstrap stability")
    axes[0].set(title="K-means", xlabel="K", ylim=(-0.05, 1.05)); axes[0].legend()
    candidates = metrics["gmm"]["candidates"]
    covariance_types = sorted({row.get("covariance_type", "diag") for row in candidates})
    for covariance_type in covariance_types:
        rows = [row for row in candidates if row.get("covariance_type", "diag") == covariance_type]
        axes[1].plot([row["k"] for row in rows], [row["silhouette"] for row in rows], marker="o",
                     label=f"{covariance_type} silhouette")
        axes[1].plot([row["k"] for row in rows], [row["stability_ari"] for row in rows], marker="s", linestyle="--",
                     label=f"{covariance_type} stability")
    axes[1].set(title="GMM", xlabel="K", ylim=(-0.05, 1.05)); axes[1].legend(fontsize="small")
    outputs.append(output_dir / "model-selection.png"); _save(fig, outputs[-1])

    external_data = run_path(config, "results", "metrics", "external-map-data.npz")
    if external_data.exists():
        with np.load(external_data) as data:
            for method in ("kmeans", "gmm"):
                key = f"{method}_primary_disagreement"
                disagreement = data[key] if key in data else data[f"{method}_disagreement"]
                valid = disagreement != 255
                fig, ax = plt.subplots(figsize=(10, 4.8))
                ax.scatter(longitude[valid], latitude[valid], c=disagreement[valid], s=1, cmap="coolwarm", vmin=0, vmax=1, rasterized=True)
                padding = float(config.get("region", {}).get("map_padding_degrees", 1.0))
                ax.set(xlabel="Longitude", ylabel="Latitude", title=f"{method.upper()} vs Köppen primary-group disagreement",
                       xlim=(float(longitude.min()) - padding, float(longitude.max()) + padding),
                       ylim=(float(latitude.min()) - padding, float(latitude.max()) + padding))
                outputs.append(output_dir / f"{method}-koppen-disagreement.png"); _save(fig, outputs[-1])
    return outputs
