from __future__ import annotations

import json
from pathlib import Path

from .artifacts import manifest, write_json
from .config import config_hash, project_path


def freeze(config: dict, force: bool = False) -> Path:
    inputs = [
        project_path("data", "processed", "climate_features.npz"),
        project_path("results", "metrics", "internal.json"),
        project_path("results", "models", "scaler.joblib"),
        project_path("results", "models", "pca.joblib"),
        project_path("results", "models", "pca_features.npy"),
        project_path("results", "models", "kmeans.joblib"),
        project_path("results", "models", "kmeans_labels.npy"),
        project_path("results", "models", "gmm.joblib"),
        project_path("results", "models", "gmm_labels.npy"),
        project_path("results", "models", "gmm_confidence.npy"),
    ]
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze; missing artifacts: {missing}")
    internal = json.loads(inputs[1].read_text(encoding="utf-8"))
    output = project_path("results", "manifests", "frozen-model.json")
    payload = manifest(
        "freeze", config_hash(config), inputs,
        selected_kmeans_k=internal["kmeans"]["selected"]["k"],
        selected_gmm_k=internal["gmm"]["selected"]["k"],
        preprocessing={"precipitation_transform": "log1p", "scaling": "standard",
                       "pca_variance_target": config["features"]["pca_variance_target"]},
    )
    write_json(output, payload, overwrite=force)
    return output
