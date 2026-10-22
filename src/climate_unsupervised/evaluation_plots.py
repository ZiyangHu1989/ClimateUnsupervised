from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from .config import run_path
from .preprocess import MONTHS


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _categorical_map(
    longitude: np.ndarray, latitude: np.ndarray, values: np.ndarray, valid: np.ndarray,
    title: str, path: Path, labels: list[str] | None = None,
) -> None:
    ids = np.unique(values[valid])
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(ids), 2)))
    cmap = ListedColormap(colors[:len(ids)])
    normalized = np.searchsorted(ids, values[valid])
    fig, ax = plt.subplots(figsize=(11, 5.2))
    scatter = ax.scatter(longitude[valid], latitude[valid], c=normalized, s=1.1, cmap=cmap,
                         vmin=-.5, vmax=len(ids)-.5, rasterized=True)
    colorbar = fig.colorbar(scatter, ax=ax, ticks=np.arange(len(ids)), fraction=.025, pad=.02)
    tick_labels = ([labels[int(value)] for value in ids] if labels is not None
                   else [str(int(value)) for value in ids])
    colorbar.ax.set_yticklabels(tick_labels)
    lon_values, lat_values = longitude[valid], latitude[valid]
    ax.set(title=title, xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
           xlim=(float(lon_values.min()) - 1, float(lon_values.max()) + 1),
           ylim=(float(lat_values.min()) - 1, float(lat_values.max()) + 1), aspect="auto")
    ax.grid(alpha=.18, linewidth=.5)
    _save(fig, path)


def _overlap_matrix(rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, list[int], list[int]]:
    clusters = sorted({int(row["cluster"]) for row in rows})
    classes = sorted({int(row["reference_class"]) for row in rows})
    matrix = np.zeros((len(clusters), len(classes)))
    ci, yi = {value: index for index, value in enumerate(clusters)}, {value: index for index, value in enumerate(classes)}
    for row in rows:
        matrix[ci[int(row["cluster"])], yi[int(row["reference_class"])]] = float(row[metric])
    return matrix, clusters, classes


def create_evaluation_figures(
    config: dict, internal_rows: list[dict[str, Any]], confidence: np.ndarray, entropy: np.ndarray,
    latitude: np.ndarray, longitude: np.ndarray, labels: dict[str, np.ndarray],
    references: dict[str, np.ndarray], valid: np.ndarray, overlap_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> list[Path]:
    output = run_path(config, "results", "figures", "evaluation")
    outputs: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, metric, title, better in zip(
        axes, ("silhouette", "davies_bouldin", "calinski_harabasz"),
        ("Silhouette", "Davies–Bouldin", "Calinski–Harabasz"), ("higher", "lower", "higher")
    ):
        ax.bar([row["method"].upper() for row in internal_rows], [row[metric] for row in internal_rows],
               color=["C0", "C1"])
        ax.set(title=f"{title} ({better} is better)", ylabel=title)
        ax.grid(axis="y", alpha=.2)
    outputs.append(output / "internal-quality.png"); _save(fig, outputs[-1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(confidence, bins=np.linspace(0, 1, 41), color="C1", edgecolor="none")
    axes[0].set(title="GMM maximum posterior probability", xlabel="Maximum posterior", ylabel="Grid-cell count", xlim=(0, 1))
    axes[1].hist(entropy, bins=np.linspace(0, 1, 41), color="C4", edgecolor="none")
    axes[1].set(title="GMM normalized predictive entropy", xlabel="Entropy / log(K)", ylabel="Grid-cell count", xlim=(0, 1))
    outputs.append(output / "gmm-confidence-distributions.png"); _save(fig, outputs[-1])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    scatter = ax.scatter(longitude, latitude, c=entropy, s=1.1, cmap="magma", vmin=0, vmax=1, rasterized=True)
    fig.colorbar(scatter, ax=ax, label="Normalized predictive entropy", fraction=.025, pad=.02)
    ax.set(title="GMM assignment uncertainty", xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
           xlim=(float(longitude.min()) - 1, float(longitude.max()) + 1),
           ylim=(float(latitude.min()) - 1, float(latitude.max()) + 1)); ax.grid(alpha=.18, linewidth=.5)
    outputs.append(output / "gmm-uncertainty-map.png"); _save(fig, outputs[-1])

    for method in ("kmeans", "gmm"):
        _categorical_map(longitude, latitude, labels[method], np.ones(len(latitude), dtype=bool),
                         f"{method.upper()} learned climate regions", output / f"{method}-clusters-map.png")
        outputs.append(output / f"{method}-clusters-map.png")
    primary_names = ["A Tropical", "B Arid", "C Temperate", "D Continental", "E Polar"]
    _categorical_map(longitude, latitude, references["primary_groups_5"], valid,
                     "Köppen–Geiger primary climate groups", output / "koppen-primary-map.png", primary_names)
    outputs.append(output / "koppen-primary-map.png")
    _categorical_map(longitude, latitude, references["subclasses_30"], valid,
                     "Köppen–Geiger subclasses", output / "koppen-subclasses-map.png")
    outputs.append(output / "koppen-subclasses-map.png")

    for method in ("kmeans", "gmm"):
        area_rows = [row for row in overlap_rows if row["method"] == method and
                     row["level"] == "primary_groups_5" and row["weighting"] == "area"]
        _, clusters, _ = _overlap_matrix(area_rows, "jaccard")
        strongest = {}
        for cluster in clusters:
            candidates = [row for row in area_rows if int(row["cluster"]) == cluster]
            strongest[cluster] = int(max(candidates, key=lambda row: row["jaccard"])["reference_class"])
        match = np.zeros(len(latitude), dtype=float)
        match[valid] = [strongest[int(cluster)] == int(ref) for cluster, ref in
                        zip(labels[method][valid], references["primary_groups_5"][valid])]
        fig, ax = plt.subplots(figsize=(11, 5.2))
        cmap = ListedColormap(["#c44e52", "#4c72b0"])
        scatter = ax.scatter(longitude[valid], latitude[valid], c=match[valid], s=1.1, cmap=cmap,
                             norm=BoundaryNorm([-.5, .5, 1.5], 2), rasterized=True)
        colorbar = fig.colorbar(scatter, ax=ax, ticks=[0, 1], fraction=.025, pad=.02)
        colorbar.ax.set_yticklabels(["Different structure", "Strongest-match group"])
        ax.set(title=f"{method.upper()} correspondence with Köppen primary groups",
               xlabel="Longitude (degrees)", ylabel="Latitude (degrees)",
               xlim=(float(longitude[valid].min()) - 1, float(longitude[valid].max()) + 1),
               ylim=(float(latitude[valid].min()) - 1, float(latitude[valid].max()) + 1))
        ax.grid(alpha=.18, linewidth=.5)
        outputs.append(output / f"{method}-primary-correspondence.png"); _save(fig, outputs[-1])

    for method in ("kmeans", "gmm"):
        for level in ("primary_groups_5", "subclasses_30"):
            rows = [row for row in overlap_rows if row["method"] == method and row["level"] == level
                    and row["weighting"] == "area"]
            jaccard, clusters, classes = _overlap_matrix(rows, "jaccard")
            intersection, _, _ = _overlap_matrix(rows, "intersection_weight")
            normalized = intersection / np.maximum(intersection.sum(axis=1, keepdims=True), np.finfo(float).eps)
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, .35 * len(clusters) + 2)))
            for ax, matrix, title, colorbar_label in (
                (axes[0], normalized, "Area-weighted contingency (row normalized)", "Cluster share"),
                (axes[1], jaccard, "Area-weighted Jaccard overlap", "Jaccard"),
            ):
                image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                fig.colorbar(image, ax=ax, fraction=.035, pad=.02, label=colorbar_label)
                ax.set(title=title, xlabel="Köppen class", ylabel="Learned cluster",
                       xticks=np.arange(len(classes)), yticks=np.arange(len(clusters)),
                       xticklabels=classes, yticklabels=clusters)
                ax.tick_params(axis="x", rotation=90)
            outputs.append(output / f"{method}-{level}-overlap.png"); _save(fig, outputs[-1])

    for method in ("kmeans", "gmm"):
        rows = [row for row in profile_rows if row["method"] == method]
        clusters = sorted({int(row["cluster"]) for row in rows})
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for cluster in clusters:
            temperature = sorted([row for row in rows if row["cluster"] == cluster and row["variable"] == "temperature_c"], key=lambda row: row["month"])
            precipitation = sorted([row for row in rows if row["cluster"] == cluster and row["variable"] == "precipitation_mm"], key=lambda row: row["month"])
            color = plt.get_cmap("tab20")(cluster % 20)
            for ax, values, ylabel in ((axes[0], temperature, "Temperature (°C)"), (axes[1], precipitation, "Precipitation (mm/month)")):
                mean = np.array([row["mean"] for row in values]); low = np.array([row["q25"] for row in values]); high = np.array([row["q75"] for row in values])
                ax.plot(MONTHS, mean, color=color, label=f"Cluster {cluster}")
                ax.fill_between(MONTHS, low, high, color=color, alpha=.15)
                ax.set(ylabel=ylabel, xlabel="Month"); ax.grid(alpha=.2)
        axes[0].set_title("Monthly temperature: mean and IQR")
        axes[1].set_title("Monthly precipitation: mean and IQR")
        axes[1].legend(ncol=2, fontsize="small")
        outputs.append(output / f"{method}-physical-profiles.png"); _save(fig, outputs[-1])
    return outputs
