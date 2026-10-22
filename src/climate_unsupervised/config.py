from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config["_path"] = str(config_path)
    return config


def config_hash(config: dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def run_path(config: dict[str, Any], *parts: str) -> Path:
    """Resolve an artifact inside the configured run without escaping the project."""
    raw_root = str(config.get("run", {}).get("output_root", "."))
    root = Path(raw_root)
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("run.output_root must be a project-relative path without '..'")
    return PROJECT_ROOT.joinpath(root, *parts)
