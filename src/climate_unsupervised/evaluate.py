from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

from .artifacts import manifest, verify_manifest_inputs, write_json
from .config import config_hash, project_path


def weighted_contingency(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> np.ndarray:
    _, left_index = np.unique(left, return_inverse=True)
    _, right_index = np.unique(right, return_inverse=True)
    table = np.zeros((left_index.max() + 1, right_index.max() + 1), dtype=np.float64)
    np.add.at(table, (left_index, right_index), weights)
    return table


def weighted_ari(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    """Adjusted Rand analogue for pairs drawn by geographic-area probability."""
    table = weighted_contingency(left, right, weights)
    probabilities = table / table.sum()
    same_both = float(np.square(probabilities).sum())
    same_left = float(np.square(probabilities.sum(axis=1)).sum())
    same_right = float(np.square(probabilities.sum(axis=0)).sum())
    expected = same_left * same_right
    maximum = 0.5 * (same_left + same_right)
    if np.isclose(maximum, expected):
        return 1.0
    return float((same_both - expected) / (maximum - expected))


def weighted_information_scores(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    table = weighted_contingency(left, right, weights)
    probabilities = table / table.sum()
    row = probabilities.sum(axis=1)
    column = probabilities.sum(axis=0)
    expected = row[:, None] * column[None, :]
    positive = probabilities > 0
    mutual_information = float(np.sum(probabilities[positive] * np.log(probabilities[positive] / expected[positive])))
    row_entropy = float(-np.sum(row[row > 0] * np.log(row[row > 0])))
    column_entropy = float(-np.sum(column[column > 0] * np.log(column[column > 0])))
    nmi = mutual_information / math.sqrt(row_entropy * column_entropy) if row_entropy and column_entropy else 1.0
    homogeneity = mutual_information / column_entropy if column_entropy else 1.0
    completeness = mutual_information / row_entropy if row_entropy else 1.0
    v_measure = 2 * homogeneity * completeness / (homogeneity + completeness) if homogeneity + completeness else 0.0
    return {"nmi": float(nmi), "v_measure": float(v_measure)}


def sample_raster(path: Path, longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as source:
        xs, ys = longitude.tolist(), latitude.tolist()
        if source.crs and source.crs.to_epsg() != 4326:
            xs, ys = transform("EPSG:4326", source.crs, xs, ys)
        labels = np.asarray([value[0] for value in source.sample(zip(xs, ys))])
        valid = np.isfinite(labels)
        if source.nodata is not None:
            valid &= labels != source.nodata
        valid &= labels > 0
    return labels.astype(np.int16), valid


def _majority_disagreement(clusters: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> np.ndarray:
    disagreement = np.ones(len(clusters), dtype=np.uint8)
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        values = np.unique(reference[mask])
        majority = max(values, key=lambda value: weights[mask & (reference == value)].sum())
        disagreement[mask] = reference[mask] != majority
    return disagreement


def evaluate(config: dict) -> Path:
    frozen_path = project_path("results", "manifests", "frozen-model.json")
    if not frozen_path.exists():
        raise FileNotFoundError("Run freeze before evaluate")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    verify_manifest_inputs(frozen)
    feature_path = project_path("data", "processed", "climate_features.npz")
    reference_path = project_path("data", "external", "koppen_geiger_1991_2020.tif")
    if not reference_path.exists():
        raise FileNotFoundError("Run download-koppen before evaluate")
    with np.load(feature_path) as data:
        latitude, longitude, weights = data["latitude"], data["longitude"], data["area_weights"]
    reference, valid = sample_raster(reference_path, longitude, latitude)
    if not valid.any():
        raise RuntimeError("No climate cells overlap valid Koppen-Geiger labels")
    results: dict[str, dict] = {}
    saved = {"reference": reference, "valid": valid, "latitude": latitude, "longitude": longitude}
    for method in ("kmeans", "gmm"):
        labels = np.load(project_path("results", "models", f"{method}_labels.npy"))
        left, right, selected_weights = labels[valid], reference[valid], weights[valid]
        information = weighted_information_scores(left, right, selected_weights)
        results[method] = {
            "valid_cells": int(valid.sum()),
            "cluster_ids": np.unique(left).astype(int).tolist(),
            "reference_class_ids": np.unique(right).astype(int).tolist(),
            "unweighted": {
                "ari": float(adjusted_rand_score(right, left)),
                "nmi": float(normalized_mutual_info_score(right, left)),
                "v_measure": float(v_measure_score(right, left)),
            },
            "area_weighted": {"ari": weighted_ari(left, right, selected_weights), **information},
            "contingency": weighted_contingency(left, right, selected_weights).tolist(),
        }
        disagreement = np.full(len(labels), 255, dtype=np.uint8)
        disagreement[valid] = _majority_disagreement(left, right, selected_weights)
        saved[f"{method}_disagreement"] = disagreement
    metrics_path = project_path("results", "metrics", "external.json")
    write_json(metrics_path, results)
    np.savez_compressed(project_path("results", "metrics", "external-map-data.npz"), **saved)
    write_json(project_path("results", "manifests", "evaluate.json"), manifest(
        "evaluate", config_hash(config), [frozen_path, reference_path], metrics=str(metrics_path),
        valid_cells=int(valid.sum()),
    ))
    return metrics_path
