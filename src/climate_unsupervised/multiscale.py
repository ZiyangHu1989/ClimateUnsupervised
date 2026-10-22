from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .artifacts import manifest, verify_manifest_inputs, write_json
from .config import config_hash, project_path, run_path
from .evaluate import _scores, koppen_primary_groups, sample_raster
from .modeling import area_resample, normalize_labels, spatial_block_folds, transformed_features
from .preprocess import MONTHS


METHODS = ("kmeans", "gmm_diag", "gmm_full", "birch", "ward")
FAMILIES = {"kmeans": "kmeans", "gmm_diag": "gmm", "gmm_full": "gmm",
            "birch": "birch", "ward": "ward"}


def _centroids(values: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray([np.average(values[labels == label], axis=0, weights=weights[labels == label])
                       for label in np.unique(labels)])


def _nearest_labels(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.argmin(cdist(values, centers), axis=1).astype(np.int16)


def _fit_predict(method: str, k: int, values: np.ndarray, weights: np.ndarray,
                 seed: int, pool: np.ndarray | None = None, sample_max: int | None = None,
                 n_init: int = 1, pool_is_resample: bool = False) -> tuple[Any, np.ndarray]:
    pool = np.arange(len(values)) if pool is None else np.asarray(pool)
    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=n_init, max_iter=500, random_state=seed)
        if pool_is_resample:
            model.fit(values[pool])
        else:
            model.fit(values[pool], sample_weight=weights[pool])
        return model, model.predict(values)
    if pool_is_resample:
        sample_idx = pool
    else:
        size = min(len(pool), int(sample_max or len(pool)))
        local = area_resample(weights[pool], size, seed)
        sample_idx = pool[local]
    sample = values[sample_idx]
    if method.startswith("gmm_"):
        covariance = method.removeprefix("gmm_")
        model = GaussianMixture(n_components=k, covariance_type=covariance, n_init=n_init,
                                max_iter=500, random_state=seed, reg_covar=1e-6).fit(sample)
        return model, model.predict(values)
    if method == "birch":
        model = Birch(n_clusters=k).fit(sample)
        return model, model.predict(values)
    if method == "ward":
        fitted = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(sample)
        centers = _centroids(sample, fitted, np.ones(len(sample)))
        model = {"method": "ward_nearest_centroid", "centers": centers}
        return model, _nearest_labels(values, centers)
    raise ValueError(f"Unknown benchmark method: {method}")


def _pairwise_summary(label_runs: list[np.ndarray]) -> dict[str, float]:
    scores = np.asarray([adjusted_rand_score(a, b) for a, b in combinations(label_runs, 2)])
    return {"mean": float(scores.mean()), "standard_deviation": float(scores.std(ddof=1)),
            "minimum": float(scores.min()), "maximum": float(scores.max()),
            "pair_count": int(len(scores))}


def _physical_separation(scaled: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    centers = _centroids(scaled, labels, weights)
    distances = cdist(centers, centers)
    distances[distances == 0] = np.inf
    return float(distances.min())


def build_shortlist(candidates: list[dict[str, Any]], cross_agreement: dict[int, float],
                    silhouette_ratio: float, stability_threshold: float,
                    cross_threshold: float, strong_stability_threshold: float = .85,
                    spatial_stability_threshold: float = .80) -> list[dict[str, Any]]:
    best_silhouette = max(row["silhouette"] for row in candidates)
    shortlist = []
    for k in sorted({int(row["k"]) for row in candidates}):
        eligible = [row for row in candidates if row["k"] == k and row["eligible"]]
        families = {row["family"] for row in eligible}
        if not eligible:
            continue
        representative = max(eligible, key=lambda row: (
            row["bootstrap_stability"]["mean"], row["silhouette"], -row["davies_bouldin"]
        ))
        corroborated = len(families) >= 2 and cross_agreement.get(k, -1) >= cross_threshold
        strong_single = (representative["bootstrap_stability"]["mean"] >= strong_stability_threshold and
                         representative["spatial_stability"]["mean"] >= spatial_stability_threshold)
        reasons = {
            "stability": representative["bootstrap_stability"]["mean"] >= stability_threshold,
            "silhouette_retention": representative["silhouette"] >= silhouette_ratio * best_silhouette,
            "corroborated_by_multiple_families": corroborated,
            "strong_single_family_replication": strong_single,
        }
        if reasons["stability"] and reasons["silhouette_retention"] and (corroborated or strong_single):
            shortlist.append({"k": k, "representative_method": representative["method"],
                              "cross_algorithm_ari": cross_agreement.get(k), "criteria": reasons})
    return shortlist


def run_multiscale_benchmark(config: dict, force: bool = False) -> Path:
    feature_path = run_path(config, "data", "processed", "climate_features.npz")
    if not feature_path.exists():
        raise FileNotFoundError("Run prepare-region before benchmark")
    with np.load(feature_path) as data:
        physical = data["features"].astype(np.float64)
        weights = data["area_weights"].astype(np.float64)
        latitude, longitude = data["latitude"], data["longitude"]
    settings = config["multiscale"]
    seed = int(config["project"]["random_seed"])
    transformed = transformed_features(physical)
    scaler = StandardScaler().fit(transformed, sample_weight=weights)
    scaled = scaler.transform(transformed)
    pca = PCA(n_components=float(config["features"]["pca_variance_target"]),
              svd_solver="full", random_state=seed).fit(scaled)
    reduced = pca.transform(scaled).astype(np.float32)
    output_models = run_path(config, "results", "models")
    output_metrics = run_path(config, "results", "metrics")
    output_models.mkdir(parents=True, exist_ok=True)
    output_metrics.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_models / "scaler.joblib")
    joblib.dump(pca, output_models / "pca.joblib")
    np.save(output_models / "pca_features.npy", reduced)

    seeds = [int(value) for value in settings["seeds"]]
    folds = int(settings["spatial_folds"])
    fold_ids = spatial_block_folds(latitude, longitude, float(settings["spatial_block_degrees"]), folds)
    sample_max = int(settings["algorithm_sample_max"])
    min_stability = float(settings["minimum_stability_ari"])
    min_area = float(settings["minimum_cluster_area_fraction"])
    candidates: list[dict[str, Any]] = []
    saved_labels: dict[str, np.ndarray] = {}
    saved_models: dict[str, Any] = {}
    for k in range(int(settings["k_min"]), int(settings["k_max"]) + 1):
        for method in METHODS:
            model, raw_labels = _fit_predict(method, k, reduced, weights, seed,
                                             sample_max=sample_max, n_init=int(settings["final_n_init"]))
            labels, mapping = normalize_labels(raw_labels, weights)
            bootstrap_runs = []
            for run_seed in seeds:
                idx = area_resample(weights, min(sample_max, len(weights)), 10000 + run_seed)
                _, run_labels = _fit_predict(method, k, reduced, weights, run_seed, pool=idx,
                                             sample_max=sample_max, n_init=1, pool_is_resample=True)
                bootstrap_runs.append(run_labels)
            spatial_runs = []
            for fold in range(folds):
                pool = np.flatnonzero(fold_ids != fold)
                _, run_labels = _fit_predict(method, k, reduced, weights, 20000 + fold, pool=pool,
                                             sample_max=sample_max, n_init=1)
                spatial_runs.append(run_labels)
            area_shares = np.asarray([weights[labels == label].sum() / weights.sum()
                                      for label in np.unique(labels)])
            bootstrap = _pairwise_summary(bootstrap_runs)
            spatial = _pairwise_summary(spatial_runs)
            row = {
                "method": method, "family": FAMILIES[method], "k": k,
                "silhouette": float(silhouette_score(reduced, labels)),
                "davies_bouldin": float(davies_bouldin_score(reduced, labels)),
                "calinski_harabasz": float(calinski_harabasz_score(reduced, labels)),
                "bootstrap_stability": bootstrap, "spatial_stability": spatial,
                "minimum_cluster_area_fraction": float(area_shares.min()),
                "cluster_balance_entropy": float(-np.sum(area_shares * np.log(area_shares)) / np.log(k)),
                "physical_profile_separation": _physical_separation(scaled, labels, weights),
                "label_mapping": mapping,
            }
            row["eligible"] = bootstrap["mean"] >= min_stability and row["minimum_cluster_area_fraction"] >= min_area
            candidates.append(row)
            key = f"{method}_k{k}"
            saved_labels[key] = labels
            saved_models[key] = model

    cross_agreement: dict[int, float] = {}
    cross_rows = []
    for k in range(int(settings["k_min"]), int(settings["k_max"]) + 1):
        eligible = [row for row in candidates if row["k"] == k and row["eligible"]]
        scores = []
        for left, right in combinations(eligible, 2):
            if left["family"] == right["family"]:
                continue
            score = float(adjusted_rand_score(saved_labels[f"{left['method']}_k{k}"],
                                              saved_labels[f"{right['method']}_k{k}"]))
            scores.append(score)
            cross_rows.append({"k": k, "left": left["method"], "right": right["method"], "ari": score})
        if scores:
            cross_agreement[k] = float(np.median(scores))
    shortlist = build_shortlist(
        candidates, cross_agreement, float(settings["silhouette_retention_ratio"]),
        min_stability, float(settings["minimum_cross_algorithm_ari"]),
        float(settings["strong_single_family_stability_ari"]),
        float(settings["minimum_spatial_stability_ari"]),
    )
    for item in shortlist:
        key = f"{item['representative_method']}_k{item['k']}"
        np.save(output_models / f"labels-k{item['k']}.npy", saved_labels[key])
        joblib.dump(saved_models[key], output_models / f"model-k{item['k']}.joblib")
        centers = _centroids(reduced, saved_labels[key], weights)
        distances = np.sort(cdist(reduced, centers), axis=1)[:, :2]
        confidence = (distances[:, 1] - distances[:, 0]) / np.maximum(distances[:, 1], np.finfo(float).eps)
        np.save(output_models / f"confidence-k{item['k']}.npy", confidence.astype(np.float32))
    np.savez_compressed(output_metrics / "candidate-labels.npz", **saved_labels)
    metrics = {
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "candidate_count": len(candidates), "candidates": candidates,
        "cross_algorithm_agreement": {str(k): value for k, value in cross_agreement.items()},
        "cross_algorithm_pairs": cross_rows, "shortlist": shortlist,
        "selection_policy": {
            "external_labels_used": False, "k2_role": "macro_diagnostic_only",
            "minimum_stability_ari": min_stability,
            "minimum_cluster_area_fraction": min_area,
            "silhouette_retention_ratio": settings["silhouette_retention_ratio"],
            "minimum_cross_algorithm_ari": settings["minimum_cross_algorithm_ari"],
            "strong_single_family_stability_ari": settings["strong_single_family_stability_ari"],
            "minimum_spatial_stability_ari": settings["minimum_spatial_stability_ari"],
        },
    }
    metrics_path = output_metrics / "multiscale-internal.json"
    write_json(metrics_path, metrics)
    write_json(run_path(config, "results", "manifests", "multiscale-fit.json"), manifest(
        "multiscale-fit", config_hash(config), [feature_path], metrics=str(metrics_path),
        shortlist=shortlist, candidate_count=len(candidates),
    ))
    if not shortlist:
        raise RuntimeError(f"No multiscale resolution passed the preregistered shortlist rules; audit={metrics_path}")
    return metrics_path


def freeze_multiscale(config: dict, force: bool = False) -> Path:
    metrics_path = run_path(config, "results", "metrics", "multiscale-internal.json")
    metrics = json.loads(metrics_path.read_text())
    inputs = [run_path(config, "data", "processed", "climate_features.npz"), metrics_path,
              run_path(config, "results", "models", "scaler.joblib"),
              run_path(config, "results", "models", "pca.joblib"),
              run_path(config, "results", "models", "pca_features.npy"),
              run_path(config, "results", "manifests", "prepare-region.json"),
              project_path(*Path(config["region"]["boundary_path"]).parts)]
    for item in metrics["shortlist"]:
        for prefix, suffix in (("labels", ".npy"), ("confidence", ".npy"), ("model", ".joblib")):
            inputs.append(run_path(config, "results", "models", f"{prefix}-k{item['k']}{suffix}"))
    output = run_path(config, "results", "manifests", "multiscale-frozen.json")
    write_json(output, manifest("multiscale-freeze", config_hash(config), inputs,
                                shortlist=metrics["shortlist"], external_labels_used=False), overwrite=force)
    return output


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def evaluate_multiscale(config: dict) -> Path:
    frozen_path = run_path(config, "results", "manifests", "multiscale-frozen.json")
    frozen = json.loads(frozen_path.read_text())
    verify_manifest_inputs(frozen)
    metrics = json.loads(run_path(config, "results", "metrics", "multiscale-internal.json").read_text())
    feature_path = run_path(config, "data", "processed", "climate_features.npz")
    with np.load(feature_path) as data:
        features = data["features"].astype(np.float64)
        latitude, longitude = data["latitude"], data["longitude"]
        weights = data["area_weights"].astype(np.float64)
    reference, valid = sample_raster(project_path("data", "external", "koppen_geiger_1991_2020.tif"),
                                     longitude, latitude)
    primary = koppen_primary_groups(reference)
    valid &= primary >= 0
    output_figures = run_path(config, "results", "figures", "multiscale")
    external: dict[str, Any] = {}
    profile_tables: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for method in METHODS:
        rows = [row for row in metrics["candidates"] if row["method"] == method]
        axes[0].plot([row["k"] for row in rows], [row["silhouette"] for row in rows], marker="o", label=method)
        axes[1].plot([row["k"] for row in rows], [row["bootstrap_stability"]["mean"] for row in rows], marker="o", label=method)
    axes[0].set(title="Detailed-candidate separation", xlabel="K", ylabel="Silhouette")
    axes[1].set(title="Area-bootstrap reproducibility", xlabel="K", ylabel="Mean pairwise ARI", ylim=(0, 1))
    axes[0].legend(fontsize="small"); axes[1].legend(fontsize="small");
    for ax in axes: ax.grid(alpha=.2)
    _save_figure(fig, output_figures / "candidate-diagnostics.png")
    for item in metrics["shortlist"]:
        k = int(item["k"]); key = f"k{k}"
        labels = np.load(run_path(config, "results", "models", f"labels-k{k}.npy"))
        external[key] = {
            "method": item["representative_method"],
            "primary_groups": _scores(labels[valid], primary[valid], weights[valid]),
            "subclasses": _scores(labels[valid], reference[valid], weights[valid]),
        }
        profiles = []
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(longitude, latitude, c=labels, s=4, cmap="tab20", rasterized=True)
        fig.colorbar(scatter, ax=ax, label="Cluster")
        ax.set(title=f"Australian climate classification: K={k} ({item['representative_method']})",
               xlabel="Longitude", ylabel="Latitude",
               xlim=(longitude.min()-1, longitude.max()+1), ylim=(latitude.min()-1, latitude.max()+1))
        _save_figure(fig, output_figures / f"classification-k{k}.png")
        confidence = np.load(run_path(config, "results", "models", f"confidence-k{k}.npy"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].hist(confidence, bins=np.linspace(0, 1, 31), color="C0")
        axes[0].set(title=f"K={k} centroid-margin confidence", xlabel="Normalized distance margin", ylabel="Cells")
        uncertainty = axes[1].scatter(longitude, latitude, c=1-confidence, s=4, cmap="magma", vmin=0, vmax=1, rasterized=True)
        fig.colorbar(uncertainty, ax=axes[1], label="Assignment uncertainty")
        axes[1].set(title=f"K={k} uncertainty", xlabel="Longitude", ylabel="Latitude",
                    xlim=(longitude.min()-1, longitude.max()+1), ylim=(latitude.min()-1, latitude.max()+1))
        _save_figure(fig, output_figures / f"uncertainty-k{k}.png")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for label in np.unique(labels):
            mask = labels == label
            profile = np.average(features[mask], axis=0, weights=weights[mask])
            area = float(weights[mask].sum() / weights.sum())
            profiles.append({"cluster": int(label), "area_fraction": area,
                             "annual_mean_temperature_c": float(profile[:12].mean()),
                             "temperature_range_c": float(profile[:12].max()-profile[:12].min()),
                             "annual_precipitation_mm": float(profile[12:].sum())})
            axes[0].plot(MONTHS, profile[:12], label=f"{label} ({area:.1%})")
            axes[1].plot(MONTHS, profile[12:], label=str(label))
        axes[0].set(title="Monthly temperature", ylabel="°C"); axes[1].set(title="Monthly precipitation", ylabel="mm/month")
        axes[0].legend(fontsize="small"); axes[1].legend(fontsize="small", ncol=2)
        _save_figure(fig, output_figures / f"profiles-k{k}.png")
        strongest = {}
        for cluster in np.unique(labels):
            mask = valid & (labels == cluster)
            classes = np.unique(primary[mask])
            strongest[int(cluster)] = int(max(classes, key=lambda value: weights[mask & (primary == value)].sum()))
        correspondence = np.zeros(len(labels), dtype=float)
        correspondence[valid] = [strongest[int(cluster)] == int(climate_class)
                                 for cluster, climate_class in zip(labels[valid], primary[valid])]
        fig, ax = plt.subplots(figsize=(10, 6))
        match = ax.scatter(longitude[valid], latitude[valid], c=correspondence[valid], s=4,
                           cmap="coolwarm_r", vmin=0, vmax=1, rasterized=True)
        fig.colorbar(match, ax=ax, ticks=[0, 1], label="Strongest-match correspondence")
        ax.set(title=f"K={k} correspondence with Köppen primary groups", xlabel="Longitude", ylabel="Latitude",
               xlim=(longitude.min()-1, longitude.max()+1), ylim=(latitude.min()-1, latitude.max()+1))
        _save_figure(fig, output_figures / f"koppen-correspondence-k{k}.png")
        profile_tables[key] = profiles
    write_json(run_path(config, "results", "metrics", "multiscale-external.json"), external)
    report = _write_multiscale_report(config, metrics, external, profile_tables, int(valid.sum()), len(valid))
    verify_manifest_inputs(frozen)
    write_json(run_path(config, "results", "manifests", "multiscale-evaluate.json"), manifest(
        "multiscale-evaluate", config_hash(config), [frozen_path, feature_path,
        project_path("data", "external", "koppen_geiger_1991_2020.tif")], report=str(report),
        valid_cells=int(valid.sum()),
    ))
    return report


def _write_multiscale_report(config: dict, metrics: dict[str, Any], external: dict[str, Any],
                             profiles: dict[str, Any], valid_cells: int, total_cells: int) -> Path:
    lookup = {(row["method"], int(row["k"])): row for row in metrics["candidates"]}
    lines = ["# Multiscale Australian climate classification", "",
             "## Why the earlier run produced two clusters", "",
             "The earlier rule maximized silhouette among eligible candidates. That criterion rewarded the strongest coarse north–south split and did not require useful classification detail. This benchmark treats K=2 only as a macro diagnostic and searches K=3–12 without Köppen labels.", "",
             "## Internally supported resolutions", "",
             "| K | Representative | Silhouette | Bootstrap ARI | Spatial ARI | Smallest area | Cross-algorithm ARI |", "|---:|---|---:|---:|---:|---:|---:|"]
    for item in metrics["shortlist"]:
        row = lookup[(item["representative_method"], int(item["k"]))]
        cross = "n/a" if item["cross_algorithm_ari"] is None else f"{item['cross_algorithm_ari']:.3f}"
        lines.append(f"| {item['k']} | {item['representative_method']} | {row['silhouette']:.3f} | {row['bootstrap_stability']['mean']:.3f} | {row['spatial_stability']['mean']:.3f} | {row['minimum_cluster_area_fraction']:.1%} | {cross} |")
    lines += ["", "No single K is declared authoritative. A resolution is listed when it passes the common stability, area, and silhouette-retention safeguards and is supported either by cross-algorithm agreement or by independently strong bootstrap and spatial replication within one family.", "",
              "## Descriptive Köppen comparison", "",
              f"Köppen labels were sampled only after the shortlist was frozen. {valid_cells:,} of {total_cells:,} Australian cells had valid reference labels.", "",
              "| K | Primary-group ARI | Area ARI | Subclass ARI | Area subclass ARI |", "|---:|---:|---:|---:|---:|"]
    for item in metrics["shortlist"]:
        result = external[f"k{item['k']}"]
        lines.append(f"| {item['k']} | {result['primary_groups']['unweighted']['ari']:.3f} | {result['primary_groups']['area_weighted']['ari']:.3f} | {result['subclasses']['unweighted']['ari']:.3f} | {result['subclasses']['area_weighted']['ari']:.3f} |")
    lines += ["", "These external scores describe correspondence and were not used to construct the shortlist.", "", "## Physical profiles", ""]
    for item in metrics["shortlist"]:
        k = int(item["k"]); lines += [f"### K={k}", "", "| Cluster | Area | Mean temperature | Temperature range | Annual precipitation |", "|---:|---:|---:|---:|---:|"]
        for row in profiles[f"k{k}"]:
            lines.append(f"| {row['cluster']} | {row['area_fraction']:.1%} | {row['annual_mean_temperature_c']:.1f} °C | {row['temperature_range_c']:.1f} °C | {row['annual_precipitation_mm']:.0f} mm |")
        lines.append("")
    lines += ["## Reproduction", "", "```bash", "climate-unsupervised multiscale-all --config configs/australia-multiscale.yaml", "```", ""]
    path = run_path(config, "results", "multiscale-report.md")
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text("\n".join(lines), encoding="utf-8")
    return path
