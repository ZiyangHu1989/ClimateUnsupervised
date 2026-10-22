from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as PolygonPath

from .artifacts import manifest, sha256_file, write_json
from .config import config_hash, project_path, run_path


def _ring_area(ring: list[list[float]]) -> float:
    points = np.asarray(ring, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def polygon_components(payload: dict[str, Any], country: str) -> list[list[list[list[float]]]]:
    matches = [feature for feature in payload.get("features", [])
               if feature.get("properties", {}).get("ADMIN") == country]
    if len(matches) != 1:
        raise ValueError(f"Expected one boundary feature for {country!r}; found {len(matches)}")
    geometry = matches[0]["geometry"]
    if geometry["type"] == "Polygon":
        components = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        components = geometry["coordinates"]
    else:
        raise ValueError(f"Unsupported boundary geometry: {geometry['type']}")
    return sorted(components, key=lambda polygon: _ring_area(polygon[0]), reverse=True)


def points_in_components(
    longitude: np.ndarray, latitude: np.ndarray,
    components: list[list[list[list[float]]]], component_count: int,
) -> np.ndarray:
    if component_count < 1 or component_count > len(components):
        raise ValueError("component_count is outside the available polygon range")
    points = np.column_stack([longitude, latitude])
    selected = np.zeros(len(points), dtype=bool)
    for polygon in components[:component_count]:
        inside = PolygonPath(np.asarray(polygon[0])).contains_points(points, radius=1e-9)
        for hole in polygon[1:]:
            inside &= ~PolygonPath(np.asarray(hole)).contains_points(points, radius=1e-9)
        selected |= inside
    return selected


def _ensure_boundary(config: dict[str, Any], force: bool) -> Path:
    region = config["region"]
    target = project_path(*Path(region["boundary_path"]).parts)
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(region["boundary_url"], headers={"User-Agent": "ClimateUnsupervised/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size == 0:
        raise RuntimeError("Regional boundary download was empty")
    temporary.replace(target)
    return target


def prepare_region(config: dict[str, Any], force: bool = False) -> Path:
    if "region" not in config:
        raise ValueError("The configuration has no region section")
    region = config["region"]
    source = project_path(*Path(region.get("source_features", "data/processed/climate_features.npz")).parts)
    if not source.exists():
        raise FileNotFoundError("Run the global preprocessing stage before preparing a region")
    output = run_path(config, "data", "processed", "climate_features.npz")
    if output.exists() and not force:
        return output

    boundary = _ensure_boundary(config, force=False)
    payload = json.loads(boundary.read_text(encoding="utf-8"))
    components = polygon_components(payload, str(region["country"]))
    component_count = int(region.get("component_count", 1))
    with np.load(source) as data:
        longitude, latitude = data["longitude"], data["latitude"]
        selected = points_in_components(longitude, latitude, components, component_count)
        if not selected.any():
            raise RuntimeError("The regional boundary retained no climate cells")
        arrays = {name: data[name][selected] if data[name].shape[:1] == (len(selected),) else data[name]
                  for name in data.files}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output)
    manifest_path = run_path(config, "results", "manifests", "prepare-region.json")
    write_json(manifest_path, manifest(
        "prepare-region", config_hash(config), [source, boundary], output=str(output),
        region_name=region.get("name", region["country"]), country=region["country"],
        boundary_url=region["boundary_url"], boundary_sha256=sha256_file(boundary),
        polygon_components_retained=component_count, locations=int(selected.sum()),
        latitude_range=[float(latitude[selected].min()), float(latitude[selected].max())],
        longitude_range=[float(longitude[selected].min()), float(longitude[selected].max())],
    ))
    return output
