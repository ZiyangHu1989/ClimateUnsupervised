from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

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
    # Arithmetic normalization matches sklearn's default NMI and V-measure,
    # making weighted and unweighted results directly comparable.
    nmi = 2 * mutual_information / (row_entropy + column_entropy) if row_entropy + column_entropy else 1.0
    homogeneity = mutual_information / column_entropy if column_entropy else 1.0
    completeness = mutual_information / row_entropy if row_entropy else 1.0
    v_measure = 2 * homogeneity * completeness / (homogeneity + completeness) if homogeneity + completeness else 0.0
    return {
        "nmi": float(nmi),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
    }


def koppen_primary_groups(labels: np.ndarray) -> np.ndarray:
    """Map the 30 Köppen-Geiger subclasses to the five A--E groups."""
    labels = np.asarray(labels)
    groups = np.full(labels.shape, -1, dtype=np.int8)
    groups[(labels >= 1) & (labels <= 3)] = 0   # A: tropical
    groups[(labels >= 4) & (labels <= 7)] = 1   # B: arid
    groups[(labels >= 8) & (labels <= 16)] = 2  # C: temperate
    groups[(labels >= 17) & (labels <= 28)] = 3 # D: continental
    groups[(labels >= 29) & (labels <= 30)] = 4 # E: polar
    return groups


def _scores(clusters: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> dict:
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(reference, clusters)
    return {
        "reference_class_count": int(len(np.unique(reference))),
        "cluster_count": int(len(np.unique(clusters))),
        "unweighted": {
            "ari": float(adjusted_rand_score(reference, clusters)),
            "ami": float(adjusted_mutual_info_score(reference, clusters)),
            "nmi": float(normalized_mutual_info_score(reference, clusters)),
            "homogeneity": float(homogeneity),
            "completeness": float(completeness),
            "v_measure": float(v_measure),
        },
        "area_weighted": {
            "ari": weighted_ari(clusters, reference, weights),
            **weighted_information_scores(clusters, reference, weights),
        },
        "contingency": weighted_contingency(clusters, reference, weights).tolist(),
    }


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
    primary = koppen_primary_groups(reference)
    valid &= primary >= 0
    results: dict[str, dict] = {}
    saved = {"reference": reference, "primary_reference": primary, "valid": valid,
             "latitude": latitude, "longitude": longitude}
    for method in ("kmeans", "gmm"):
        labels = np.load(project_path("results", "models", f"{method}_labels.npy"))
        left, right, selected_weights = labels[valid], reference[valid], weights[valid]
        primary_right = primary[valid]
        subclasses = _scores(left, right, selected_weights)
        primary_groups = _scores(left, primary_right, selected_weights)
        results[method] = {
            "valid_cells": int(valid.sum()),
            "cluster_ids": np.unique(left).astype(int).tolist(),
            "reference_class_ids": np.unique(right).astype(int).tolist(),
            "evaluation_policy": "descriptive_external_validation_only; never used for model selection",
            "subclasses_30": subclasses,
            "primary_groups_5": primary_groups,
            # Backward-compatible aliases refer to the original 30-class test.
            "unweighted": subclasses["unweighted"],
            "area_weighted": subclasses["area_weighted"],
            "contingency": subclasses["contingency"],
        }
        subclass_disagreement = np.full(len(labels), 255, dtype=np.uint8)
        subclass_disagreement[valid] = _majority_disagreement(left, right, selected_weights)
        primary_disagreement = np.full(len(labels), 255, dtype=np.uint8)
        primary_disagreement[valid] = _majority_disagreement(left, primary_right, selected_weights)
        saved[f"{method}_subclass_disagreement"] = subclass_disagreement
        saved[f"{method}_primary_disagreement"] = primary_disagreement
        saved[f"{method}_disagreement"] = primary_disagreement
    metrics_path = project_path("results", "metrics", "external.json")
    write_json(metrics_path, results)
    np.savez_compressed(project_path("results", "metrics", "external-map-data.npz"), **saved)
    write_json(project_path("results", "manifests", "evaluate.json"), manifest(
        "evaluate", config_hash(config), [frozen_path, reference_path], metrics=str(metrics_path),
        valid_cells=int(valid.sum()),
    ))
    return metrics_path
