from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config as config_module


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any], overwrite: bool = True) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def manifest(stage: str, config_digest: str, inputs: Iterable[Path], **extra: Any) -> dict[str, Any]:
    input_hashes = {}
    for path in inputs:
        resolved = path.resolve()
        try:
            key = str(resolved.relative_to(config_module.PROJECT_ROOT.resolve()))
        except ValueError:
            key = str(resolved)
        input_hashes[key] = sha256_file(resolved)
    return {
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_digest,
        "inputs": input_hashes,
        **extra,
    }


def verify_manifest_inputs(payload: dict[str, Any]) -> None:
    for raw_path, expected in payload.get("inputs", {}).items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = config_module.PROJECT_ROOT / path
        if not path.exists() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen input changed or is missing: {path}")
