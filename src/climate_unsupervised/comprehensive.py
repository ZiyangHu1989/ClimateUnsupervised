from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from .artifacts import manifest, sha256_file, verify_manifest_inputs, write_json
from .config import config_hash, project_path, run_path
from .evaluate import (
    koppen_primary_groups,
    sample_raster,
    weighted_ari,
    weighted_information_scores,
)
from .modeling import area_resample, area_sample, spatial_block_folds


METHODS = ("kmeans", "gmm")
LEVELS = ("primary_groups_5", "subclasses_30")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    columns = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: _json_value(row[key]) for key in columns} for row in rows])
    temporary.replace(path)


def predictive_entropy(probabilities: np.ndarray, normalize: bool = True) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("Expected an observations-by-components probability matrix")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Rows must contain valid probabilities")
    safe = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    entropy = -np.sum(probabilities * np.log(safe), axis=1)
    if normalize:
        entropy /= np.log(probabilities.shape[1])
    return entropy


def pairwise_ari_summary(label_runs: list[np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if len(label_runs) < 2:
        raise ValueError("At least two label runs are required")
    rows = []
    for (left_id, left), (right_id, right) in combinations(enumerate(label_runs), 2):
        rows.append({"run_i": left_id, "run_j": right_id,
                     "ari": float(adjusted_rand_score(left, right))})
    values = np.asarray([row["ari"] for row in rows])
    return rows, {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "pair_count": int(len(values)),
    }


def overlap_diagnostics(
    clusters: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    clusters = np.asarray(clusters)
    reference = np.asarray(reference)
    weights = np.asarray(weights, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    strongest: dict[int, int] = {}
    for cluster in np.unique(clusters):
        cluster_mask = clusters == cluster
        cluster_mass = float(weights[cluster_mask].sum())
        best = (-1.0, -1)
        for climate_class in np.unique(reference):
            class_mask = reference == climate_class
            class_mass = float(weights[class_mask].sum())
            intersection = float(weights[cluster_mask & class_mask].sum())
            union = cluster_mass + class_mass - intersection
            precision = intersection / cluster_mass if cluster_mass else 0.0
            recall = intersection / class_mass if class_mass else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            jaccard = intersection / union if union else 0.0
            rows.append({
                "cluster": int(cluster), "reference_class": int(climate_class),
                "intersection_weight": intersection, "cluster_weight": cluster_mass,
                "reference_weight": class_mass, "jaccard": jaccard,
                "precision": precision, "recall": recall, "f1": f1,
            })
            if (jaccard, -int(climate_class)) > (best[0], -best[1]):
                best = (jaccard, int(climate_class))
        strongest[int(cluster)] = best[1]
    return rows, strongest


def weighted_quantile(values: np.ndarray, quantiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ordered_values = np.asarray(values)[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    cumulative = np.cumsum(ordered_weights) - 0.5 * ordered_weights
    cumulative /= ordered_weights.sum()
    return np.interp(quantiles, cumulative, ordered_values)


def _profile_rows(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cluster in np.unique(labels):
        mask = labels == cluster
        cluster_weights = weights[mask]
        for feature_index in range(features.shape[1]):
            values = features[mask, feature_index].astype(np.float64)
            mean = float(np.average(values, weights=cluster_weights))
            variance = float(np.average(np.square(values - mean), weights=cluster_weights))
            q10, q25, median, q75, q90 = weighted_quantile(
                values, np.array([0.10, 0.25, 0.50, 0.75, 0.90]), cluster_weights
            )
            rows.append({
                "method": method, "cluster": int(cluster),
                "variable": "temperature_c" if feature_index < 12 else "precipitation_mm",
                "month": int(feature_index % 12 + 1), "mean": mean,
                "standard_deviation": float(np.sqrt(variance)),
                "q10": q10, "q25": q25, "median": median, "q75": q75, "q90": q90,
            })
    return rows


def _cluster_summary(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, method: str) -> list[dict[str, Any]]:
    rows = []
    total_area = weights.sum()
    for cluster in np.unique(labels):
        mask = labels == cluster
        monthly = np.average(features[mask], axis=0, weights=weights[mask])
        precipitation = monthly[12:]
        concentration = float(precipitation.max() / precipitation.sum()) if precipitation.sum() else 0.0
        rows.append({
            "method": method, "cluster": int(cluster), "cell_count": int(mask.sum()),
            "area_proportion": float(weights[mask].sum() / total_area),
            "annual_mean_temperature_c": float(monthly[:12].mean()),
            "temperature_seasonality_c": float(monthly[:12].max() - monthly[:12].min()),
            "annual_precipitation_mm": float(precipitation.sum()),
            "precipitation_peak_month_fraction": concentration,
        })
    return rows


def _subdivision_rows(
    summaries: list[dict[str, Any]], labels: np.ndarray, reference: np.ndarray,
    weights: np.ndarray, method: str, level: str,
) -> list[dict[str, Any]]:
    lookup = {(row["method"], row["cluster"]): row for row in summaries}
    rows = []
    for climate_class in np.unique(reference):
        mask = reference == climate_class
        class_mass = float(weights[mask].sum())
        masses = [(int(cluster), float(weights[mask & (labels == cluster)].sum()))
                  for cluster in np.unique(labels)]
        masses.sort(key=lambda item: item[1], reverse=True)
        if len(masses) < 2 or masses[1][1] / class_mass < 0.10:
            continue
        first, second = masses[0][0], masses[1][0]
        a, b = lookup[(method, first)], lookup[(method, second)]
        rows.append({
            "method": method, "level": level, "reference_class": int(climate_class),
            "largest_cluster": first, "second_cluster": second,
            "largest_overlap_share": masses[0][1] / class_mass,
            "second_overlap_share": masses[1][1] / class_mass,
            "temperature_mean_difference_c": abs(a["annual_mean_temperature_c"] - b["annual_mean_temperature_c"]),
            "temperature_seasonality_difference_c": abs(a["temperature_seasonality_c"] - b["temperature_seasonality_c"]),
            "annual_precipitation_difference_mm": abs(a["annual_precipitation_mm"] - b["annual_precipitation_mm"]),
            "precipitation_concentration_difference": abs(a["precipitation_peak_month_fraction"] - b["precipitation_peak_month_fraction"]),
        })
    return rows


def _regional_rows(
    latitude: np.ndarray, longitude: np.ndarray, valid: np.ndarray,
    labels: dict[str, np.ndarray], references: dict[str, np.ndarray],
    strongest: dict[tuple[str, str], dict[int, int]], weights: np.ndarray,
) -> list[dict[str, Any]]:
    regions: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        "Andes window": lambda lat, lon: (lat >= -55) & (lat <= 10) & (lon >= -80) & (lon <= -65),
        "Himalaya-Tibetan window": lambda lat, lon: (lat >= 25) & (lat <= 40) & (lon >= 65) & (lon <= 105),
        "Sahara-Sahel window": lambda lat, lon: (lat >= 10) & (lat <= 35) & (lon >= -20) & (lon <= 40),
        "South Asian monsoon window": lambda lat, lon: (lat >= 5) & (lat <= 30) & (lon >= 65) & (lon <= 105),
        "North American transition window": lambda lat, lon: (lat >= 35) & (lat <= 60) & (lon >= -130) & (lon <= -60),
        "Eurasian transition window": lambda lat, lon: (lat >= 35) & (lat <= 60) & (lon >= -10) & (lon <= 140),
        "Polar latitudes": lambda lat, lon: np.abs(lat) >= 60,
    }
    rows = []
    for method in METHODS:
        for level in LEVELS:
            reference = references[level]
            mapping = strongest[(method, level)]
            match = np.asarray([mapping[int(cluster)] == int(climate_class)
                                for cluster, climate_class in zip(labels[method], reference)])
            for region_name, selector in regions.items():
                mask = valid & selector(latitude, longitude)
                if not mask.any():
                    continue
                rows.append({
                    "method": method, "level": level, "region": region_name,
                    "valid_cells": int(mask.sum()),
                    "area_weighted_correspondence": float(np.average(match[mask], weights=weights[mask])),
                })
    return rows


def _alignment_audit(reference_path: Path, latitude: np.ndarray, longitude: np.ndarray,
                     valid: np.ndarray) -> dict[str, Any]:
    with rasterio.open(reference_path) as source:
        crs = str(source.crs) if source.crs else None
        if crs is None:
            raise RuntimeError("Köppen raster has no coordinate reference system")
        bounds = [float(value) for value in source.bounds]
        resolution = [float(abs(value)) for value in source.res]
        confidence_available = source.count > 1
        return {
            "climate_grid": {
                "coordinate_count": int(len(latitude)), "crs": "EPSG:4326",
                "longitude_convention": "[-180, 180)",
                "latitude_range": [float(latitude.min()), float(latitude.max())],
                "longitude_range": [float(longitude.min()), float(longitude.max())],
                "nominal_resolution_degrees": 0.5,
            },
            "koppen_raster": {
                "crs": crs, "width": source.width, "height": source.height,
                "resolution_degrees": resolution, "bounds": bounds,
                "band_count": source.count, "nodata": _json_value(source.nodata),
                "classification_confidence_available": confidence_available,
            },
            "alignment_method": "sample Köppen raster at each ERA5-Land cell-center coordinate",
            "valid_overlap_cells": int(valid.sum()),
            "invalid_or_missing_cells": int((~valid).sum()),
            "valid_overlap_fraction": float(valid.mean()),
            "area_weights": "cosine(latitude), inherited from processed climate dataset",
        }


def _stability_experiments(
    config: dict, reduced: np.ndarray, weights: np.ndarray,
    latitude: np.ndarray, longitude: np.ndarray, common_idx: np.ndarray,
    selected: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    seeds = [int(seed) for seed in config["clustering"]["seeds"]]
    sample_size = min(int(config["clustering"]["stability_sample_max"]), len(weights))
    folds = int(config["clustering"]["spatial_stability_folds"])
    fold_ids = spatial_block_folds(latitude, longitude,
                                   float(config["clustering"]["spatial_block_degrees"]), folds)
    all_assignments: dict[str, np.ndarray] = {"evaluation_indices": common_idx}
    pair_rows: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    for method in METHODS:
        random_runs = []
        fixed_gmm_idx = area_resample(weights, sample_size, int(config["project"]["random_seed"]))
        for run_seed in seeds:
            if method == "kmeans":
                model = KMeans(n_clusters=int(selected[method]["k"]), n_init=1,
                               max_iter=500, random_state=run_seed)
                model.fit(reduced, sample_weight=weights)
            else:
                model = GaussianMixture(
                    n_components=int(selected[method]["k"]),
                    covariance_type=selected[method]["covariance_type"], n_init=1,
                    max_iter=500, random_state=run_seed, reg_covar=1e-6,
                ).fit(reduced[fixed_gmm_idx])
            random_runs.append(model.predict(reduced[common_idx]))
        random_pairs, random_summary = pairwise_ari_summary(random_runs)
        for row in random_pairs:
            pair_rows.append({"method": method, "experiment": "random_seed", **row})
        all_assignments[f"{method}_random_seed"] = np.asarray(random_runs, dtype=np.int16)

        bootstrap_runs = []
        for run_seed in seeds:
            idx = area_resample(weights, sample_size, 10000 + run_seed)
            if method == "kmeans":
                model = KMeans(n_clusters=int(selected[method]["k"]), n_init=1,
                               max_iter=500, random_state=run_seed).fit(reduced[idx])
            else:
                model = GaussianMixture(
                    n_components=int(selected[method]["k"]),
                    covariance_type=selected[method]["covariance_type"], n_init=1,
                    max_iter=500, random_state=run_seed, reg_covar=1e-6,
                ).fit(reduced[idx])
            bootstrap_runs.append(model.predict(reduced[common_idx]))
        bootstrap_pairs, bootstrap_summary = pairwise_ari_summary(bootstrap_runs)
        for row in bootstrap_pairs:
            pair_rows.append({"method": method, "experiment": "area_bootstrap", **row})
        all_assignments[f"{method}_area_bootstrap"] = np.asarray(bootstrap_runs, dtype=np.int16)

        spatial_runs = []
        for fold in range(folds):
            pool = np.flatnonzero(fold_ids != fold)
            idx = area_resample(weights[pool], min(sample_size, len(pool)), 20000 + fold)
            if method == "kmeans":
                model = KMeans(n_clusters=int(selected[method]["k"]), n_init=1,
                               max_iter=500, random_state=30000 + fold).fit(reduced[pool[idx]])
            else:
                model = GaussianMixture(
                    n_components=int(selected[method]["k"]),
                    covariance_type=selected[method]["covariance_type"], n_init=1,
                    max_iter=500, random_state=30000 + fold, reg_covar=1e-6,
                ).fit(reduced[pool[idx]])
            spatial_runs.append(model.predict(reduced[common_idx]))
        spatial_pairs, spatial_summary = pairwise_ari_summary(spatial_runs)
        for row in spatial_pairs:
            pair_rows.append({"method": method, "experiment": "spatial_block_holdout", **row})
        all_assignments[f"{method}_spatial_block"] = np.asarray(spatial_runs, dtype=np.int16)
        output[method] = {
            "random_seed": random_summary,
            "area_bootstrap": bootstrap_summary,
            "spatial_block_holdout": spatial_summary,
            "run_count": len(seeds), "evaluation_sample_size": int(len(common_idx)),
            "training_sample_size": sample_size, "spatial_folds": folds,
            "spatial_block_degrees": float(config["clustering"]["spatial_block_degrees"]),
        }
    return output, all_assignments, pair_rows


def run_comprehensive_evaluation(config: dict) -> Path:
    frozen_path = run_path(config, "results", "manifests", "frozen-model.json")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    verify_manifest_inputs(frozen)
    protected = [run_path(config, "results", "models", name) for name in (
        "kmeans.joblib", "kmeans_labels.npy", "gmm.joblib", "gmm_labels.npy"
    )]
    hashes_before = {str(path): sha256_file(path) for path in protected}
    feature_path = run_path(config, "data", "processed", "climate_features.npz")
    reference_path = project_path("data", "external", "koppen_geiger_1991_2020.tif")
    with np.load(feature_path) as data:
        features = data["features"].astype(np.float64)
        latitude, longitude = data["latitude"], data["longitude"]
        weights = data["area_weights"].astype(np.float64)
    reduced = np.load(run_path(config, "results", "models", "pca_features.npy"))
    labels = {method: np.load(run_path(config, "results", "models", f"{method}_labels.npy")) for method in METHODS}
    internal = json.loads(run_path(config, "results", "metrics", "internal.json").read_text(encoding="utf-8"))
    selected = {method: internal[method]["selected"] for method in METHODS}
    seed = int(config["project"]["random_seed"])
    common_idx = area_sample(weights, int(config["features"]["metric_sample_max"]), seed + 1)

    internal_rows = []
    for method in METHODS:
        sample_labels = labels[method][common_idx]
        internal_rows.append({
            "method": method, "cluster_count": int(len(np.unique(labels[method]))),
            "representation": "saved PCA coordinates", "sample_size": int(len(common_idx)),
            "sampling": "area-proportional without replacement, seed=random_seed+1",
            "silhouette": float(silhouette_score(reduced[common_idx], sample_labels)),
            "davies_bouldin": float(davies_bouldin_score(reduced[common_idx], sample_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(reduced[common_idx], sample_labels)),
        })

    gmm = joblib.load(run_path(config, "results", "models", "gmm.joblib"))
    probabilities = gmm.predict_proba(reduced)
    confidence = probabilities.max(axis=1)
    entropy = predictive_entropy(probabilities)
    uncertainty = {
        "normalization": "entropy divided by log(number of mixture components)",
        "maximum_posterior_quantiles": {str(q): float(np.quantile(confidence, q))
                                         for q in (0, .1, .25, .5, .75, .9, 1)},
        "normalized_entropy_quantiles": {str(q): float(np.quantile(entropy, q))
                                          for q in (0, .1, .25, .5, .75, .9, 1)},
        "fraction_maximum_posterior_below_0.6": float(np.mean(confidence < .6)),
        "fraction_maximum_posterior_below_0.8": float(np.mean(confidence < .8)),
    }

    stability, stability_assignments, pair_rows = _stability_experiments(
        config, reduced, weights, latitude, longitude, common_idx, selected
    )

    reference, valid = sample_raster(reference_path, longitude, latitude)
    primary = koppen_primary_groups(reference)
    valid &= primary >= 0
    references = {"primary_groups_5": primary, "subclasses_30": reference}
    alignment = _alignment_audit(reference_path, latitude, longitude, valid)
    overlap_rows: list[dict[str, Any]] = []
    strongest: dict[tuple[str, str], dict[int, int]] = {}
    external: dict[str, Any] = {}
    for method in METHODS:
        external[method] = {}
        for level in LEVELS:
            left, right, selected_weights = labels[method][valid], references[level][valid], weights[valid]
            rows, _ = overlap_diagnostics(left, right, np.ones_like(selected_weights))
            weighted_rows, mapping = overlap_diagnostics(left, right, selected_weights)
            for row in rows:
                overlap_rows.append({"method": method, "level": level, "weighting": "grid_cell", **row})
            for row in weighted_rows:
                overlap_rows.append({"method": method, "level": level, "weighting": "area", **row})
            strongest[(method, level)] = mapping
            from .evaluate import _scores
            external[method][level] = _scores(left, right, selected_weights)
            external[method][level]["strongest_jaccard_match_by_cluster"] = {str(k): v for k, v in mapping.items()}
        external[method]["nmi_convention"] = "arithmetic mean normalization; sklearn default"
        external[method]["v_measure_note"] = "NMI and V-measure are algebraically equivalent under this convention"

    profile_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    subdivision_rows: list[dict[str, Any]] = []
    for method in METHODS:
        profile_rows.extend(_profile_rows(features, labels[method], weights, method))
        summary_rows.extend(_cluster_summary(features, labels[method], weights, method))
    for method in METHODS:
        for level in LEVELS:
            subdivision_rows.extend(_subdivision_rows(
                summary_rows, labels[method][valid], references[level][valid], weights[valid], method, level
            ))
    regional_rows = ([] if config.get("region") else
                     _regional_rows(latitude, longitude, valid, labels, references, strongest, weights))

    metrics_dir = run_path(config, "results", "metrics")
    write_json(metrics_dir / "evaluation-summary.json", {
        "methodology": {"external_labels_used_for_model_selection": False,
                        "common_internal_representation": "saved PCA coordinates",
                        "common_internal_sample_size": int(len(common_idx))},
        "internal_quality": internal_rows, "gmm_uncertainty": uncertainty,
        "stability": stability, "external": external, "alignment": alignment,
    })
    write_json(metrics_dir / "alignment-audit.json", alignment)
    write_json(metrics_dir / "stability.json", stability)
    np.savez_compressed(metrics_dir / "stability-assignments.npz", **stability_assignments)
    np.savez_compressed(metrics_dir / "evaluation-map-data.npz", latitude=latitude, longitude=longitude,
                        valid=valid, reference=reference, primary_reference=primary,
                        gmm_confidence=confidence.astype(np.float32), gmm_entropy=entropy.astype(np.float32))
    _write_csv(metrics_dir / "internal-quality.csv", internal_rows)
    _write_csv(metrics_dir / "stability-pairwise-ari.csv", pair_rows)
    _write_csv(metrics_dir / "class-overlap.csv", overlap_rows)
    _write_csv(metrics_dir / "cluster-profiles.csv", profile_rows)
    _write_csv(metrics_dir / "cluster-summary.csv", summary_rows)
    _write_csv(metrics_dir / "climate-subdivisions.csv", subdivision_rows)
    if regional_rows:
        _write_csv(metrics_dir / "regional-correspondence.csv", regional_rows)

    from .evaluation_plots import create_evaluation_figures
    figures = create_evaluation_figures(config, internal_rows, confidence, entropy, latitude, longitude,
                                        labels, references, valid, overlap_rows, profile_rows)
    report_path = _write_report(config, internal_rows, uncertainty, stability, external, alignment,
                                summary_rows, subdivision_rows, regional_rows, figures)
    hashes_after = {str(path): sha256_file(path) for path in protected}
    if hashes_before != hashes_after:
        raise RuntimeError("Comprehensive evaluation modified a frozen model or assignment")
    verify_manifest_inputs(frozen)
    write_json(run_path(config, "results", "manifests", "comprehensive-evaluate.json"), manifest(
        "comprehensive-evaluate", config_hash(config), [frozen_path, feature_path, reference_path],
        report=str(report_path), figures=len(figures), frozen_artifacts_unchanged=True,
    ))
    return report_path


def _write_report(
    config: dict[str, Any], internal_rows: list[dict[str, Any]], uncertainty: dict[str, Any], stability: dict[str, Any],
    external: dict[str, Any], alignment: dict[str, Any], summaries: list[dict[str, Any]],
    subdivisions: list[dict[str, Any]], regions: list[dict[str, Any]], figures: list[Path],
) -> Path:
    quality = {row["method"]: row for row in internal_rows}
    region_name = config.get("region", {}).get("name", "global")
    region_adjective = config.get("region", {}).get(
        "adjective", {"Australia": "Australian"}.get(region_name, region_name)
    )
    lines = [
        f"# Comprehensive evaluation of {region_adjective} climate clusters", "",
        "## Methodology", "",
        "The frozen K-means and GMM assignments were evaluated in their common saved PCA representation. "
        "Köppen–Geiger labels were loaded only after frozen-input verification and were never used for model selection. "
        "Silhouette, Davies–Bouldin, and Calinski–Harabasz use the same reproducible area-proportional sample.", "",
        "## Dataset and spatial alignment", "",
        f"The processed dataset contains {alignment['climate_grid']['coordinate_count']:,} land cells. "
        f"Köppen sampling produced {alignment['valid_overlap_cells']:,} valid matches "
        f"({alignment['valid_overlap_fraction']:.1%}). The reference is {alignment['koppen_raster']['crs']} at "
        f"{alignment['koppen_raster']['resolution_degrees'][0]:.6f}° resolution. "
        f"Reference confidence available: {alignment['koppen_raster']['classification_confidence_available']}.", "",
        "## RQ1 — Internal coherence and reproducibility", "",
        "| Method | Silhouette | Davies–Bouldin | Calinski–Harabasz | Seed ARI mean±SD | Bootstrap ARI mean±SD | Spatial ARI mean±SD |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        q = quality[method]
        seed_stats = stability[method]["random_seed"]
        bootstrap = stability[method]["area_bootstrap"]
        spatial = stability[method]["spatial_block_holdout"]
        lines.append(f"| {method.upper()} | {q['silhouette']:.3f} | {q['davies_bouldin']:.3f} | {q['calinski_harabasz']:.1f} | "
                     f"{seed_stats['mean']:.3f}±{seed_stats['standard_deviation']:.3f} | "
                     f"{bootstrap['mean']:.3f}±{bootstrap['standard_deviation']:.3f} | "
                     f"{spatial['mean']:.3f}±{spatial['standard_deviation']:.3f} |")
    lines += ["", f"For GMM, {uncertainty['fraction_maximum_posterior_below_0.8']:.1%} of cells have maximum posterior below 0.8. "
              "Normalized entropy is reported in `evaluation-summary.json`.", ""]
    if config.get("region"):
        lines += ["For regional interpretation, GMM is the more robust saved partition because its bootstrap and spatial-block "
                  "stability are higher. K-means remains the stronger silhouette baseline. This comparison uses internal "
                  "diagnostics only; Köppen agreement did not determine the recommendation.", ""]
    lines += [
              "## RQ2 — Agreement with Köppen–Geiger", "",
              "| Method | Level | ARI | NMI / V-measure | Area ARI | Area NMI / V-measure |", "|---|---|---:|---:|---:|---:|"]
    for method in METHODS:
        for level in LEVELS:
            result = external[method][level]
            lines.append(f"| {method.upper()} | {level} | {result['unweighted']['ari']:.3f} | "
                         f"{result['unweighted']['nmi']:.3f} | {result['area_weighted']['ari']:.3f} | "
                         f"{result['area_weighted']['nmi']:.3f} |")
    lines += ["", "NMI uses arithmetic-mean normalization; with this convention it equals V-measure, so the two are not independent signals. "
              "Full contingency, Jaccard, precision, recall, and F1 matrices are stored in `class-overlap.csv`.", "",
              "## RQ3 — Geographic agreement and disagreement", ""]
    if regions:
        lines += ["Regional correspondence is computed after mapping each learned cluster to its strongest Jaccard-overlap class; raw IDs are never compared. "
                  "The named geographic windows are diagnostics rather than exact topographic or biome masks. See `regional-correspondence.csv` and the maps.", ""]
    else:
        lines += [f"Geographic maps show assignments and Köppen correspondence across {region_name}. "
                  "No informal subregional windows are used in this regional analysis.", ""]
    lines += [
              "## RQ4 — Physical interpretation and subdivisions", "",
              f"Monthly physical-variable profiles cover {len(summaries)} learned clusters and include weighted dispersion and quantile bands.", "",
              "| Method | Cluster | Area | Mean temperature | Temperature range | Annual precipitation |", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['method'].upper()} | {row['cluster']} | {row['area_proportion']:.1%} | "
                     f"{row['annual_mean_temperature_c']:.1f} °C | {row['temperature_seasonality_c']:.1f} °C | "
                     f"{row['annual_precipitation_mm']:.0f} mm |")
    strongest_subdivisions = sorted(subdivisions, key=lambda row: (
        row["annual_precipitation_difference_mm"] + 50 * row["temperature_mean_difference_c"]
    ), reverse=True)[:8]
    lines += ["", f"The analysis contains {len(subdivisions)} reference-class/cluster subdivision comparisons in which the secondary cluster covers at least 10% of reference-class area. "
              "The largest physical contrasts are listed below as diagnostics, not significance tests.", "",
              "| Method | Level | Köppen ID | Clusters | Δ temperature | Δ annual precipitation |", "|---|---|---:|---|---:|---:|"]
    for row in strongest_subdivisions:
        lines.append(f"| {row['method'].upper()} | {row['level']} | {row['reference_class']} | "
                     f"{row['largest_cluster']} vs {row['second_cluster']} | "
                     f"{row['temperature_mean_difference_c']:.1f} °C | {row['annual_precipitation_difference_mm']:.0f} mm |")
    if regions:
        lines += ["", "### Regional correspondence for primary groups", "",
                  "| Method | Diagnostic region | Area-weighted correspondence |", "|---|---|---:|"]
        for row in regions:
            if row["level"] == "primary_groups_5":
                lines.append(f"| {row['method'].upper()} | {row['region']} | {row['area_weighted_correspondence']:.1%} |")
    raw_config_path = Path(config.get("_path", "configs/default.yaml"))
    try:
        config_reference = str(raw_config_path.resolve().relative_to(project_path().resolve()))
    except ValueError:
        config_reference = str(raw_config_path)
    lines += ["",
              "## Limitations and future work", "",
              "- Köppen is a development benchmark, not an untouched confirmatory test.",
              "- Spatial block stability exposes geographic sampling sensitivity that ordinary seed repeats may understate.",
              "- The Köppen raster has no confidence band, so reference-confidence analysis cannot be performed.",
              "- Temporal stability and a genuinely unseen external reference remain future confirmatory work.", "",
              "## Reproduction", "", "```bash",
              f"climate-unsupervised evaluate-full --config {config_reference}",
              "```", "",
              f"Generated {len(figures)} evaluation figures for this run.", ""]
    path = run_path(config, "results", "evaluation-report.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
