from __future__ import annotations

import json
from pathlib import Path

from .artifacts import manifest, write_json
from .config import config_hash, project_path, run_path


def freeze(config: dict, force: bool = False) -> Path:
    inputs = [
        run_path(config, "data", "processed", "climate_features.npz"),
        run_path(config, "results", "metrics", "internal.json"),
        run_path(config, "results", "models", "scaler.joblib"),
        run_path(config, "results", "models", "pca.joblib"),
        run_path(config, "results", "models", "pca_features.npy"),
        run_path(config, "results", "models", "kmeans.joblib"),
        run_path(config, "results", "models", "kmeans_labels.npy"),
        run_path(config, "results", "models", "gmm.joblib"),
        run_path(config, "results", "models", "gmm_labels.npy"),
        run_path(config, "results", "models", "gmm_confidence.npy"),
    ]
    if config.get("region"):
        inputs.extend([
            run_path(config, "results", "manifests", "prepare-region.json"),
            project_path(*Path(config["region"]["boundary_path"]).parts),
        ])
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot freeze; missing artifacts: {missing}")
    internal = json.loads(inputs[1].read_text(encoding="utf-8"))
    output = run_path(config, "results", "manifests", "frozen-model.json")
    payload = manifest(
        "freeze", config_hash(config), inputs,
        selected_kmeans_k=internal["kmeans"]["selected"]["k"],
        selected_gmm_k=internal["gmm"]["selected"]["k"],
        selected_gmm_covariance=internal["gmm"]["selected"].get("covariance_type", "diag"),
        preprocessing={"precipitation_transform": "log1p", "scaling": "standard",
                       "scaling_weights": "exact_cell_area",
                       "pca_sampling": internal.get("training_sampling", "area_sample"),
                       "pca_variance_target": config["features"]["pca_variance_target"]},
        region=config.get("region", {}).get("name"),
        selection_policy={"external_labels_used": False,
                          "stability": internal["kmeans"]["selected"].get("stability_kind", "initialization"),
                          "gmm_primary_metric": "silhouette_after_stability_and_minimum_area_safeguards"},
    )
    write_json(output, payload, overwrite=force)
    return output
