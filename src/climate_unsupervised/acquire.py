from __future__ import annotations

import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .artifacts import sha256_file, write_json
from .config import config_hash, project_path
from .preprocess import aggregate_year


def download_era5(config: dict, force: bool = False) -> list[Path]:
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to use CDS downloads") from exc

    era5 = config["era5"]
    first_year, last_year = map(int, era5["years"])
    destination = project_path("data", "raw", "era5")
    destination.mkdir(parents=True, exist_ok=True)
    workers = max(1, int(era5.get("download_workers", 1)))

    def process_year(year: int) -> Path:
        target = destination / f"era5_land_{year}.nc"
        interim = project_path("data", "interim", f"era5_land_{year}_0p5.npz")
        if interim.exists() and interim.stat().st_size > 0 and not force:
            return interim
        if not target.exists() or force:
            request = {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": era5["variables"],
                "year": [str(year)],
                "month": [f"{month:02d}" for month in range(1, 13)],
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
            temporary = target.with_suffix(".nc.part")
            client = cdsapi.Client()
            client.retrieve(era5["dataset"], request, str(temporary))
            if not temporary.exists() or temporary.stat().st_size == 0:
                raise RuntimeError(f"CDS returned an empty file for {year}")
            temporary.replace(target)
        if era5.get("delete_native_after_aggregation", True):
            arrays = aggregate_year(
                target, year, float(era5["source_resolution_degrees"]),
                float(era5["target_resolution_degrees"]),
            )
            interim.parent.mkdir(parents=True, exist_ok=True)
            temporary_interim = interim.with_suffix(".npz.part")
            with temporary_interim.open("wb") as handle:
                import numpy as np
                np.savez_compressed(handle, **arrays)
            temporary_interim.replace(interim)
            target.unlink()
            result = interim
        else:
            result = target
        print(f"era5_year_complete year={year} artifact={result.name}", flush=True)
        return result

    outputs_by_year: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_year, year): year for year in range(first_year, last_year + 1)}
        for future in as_completed(futures):
            year = futures[future]
            outputs_by_year[year] = future.result()
    outputs = [outputs_by_year[year] for year in range(first_year, last_year + 1)]
    write_json(project_path("results", "manifests", "download-era5.json"), {
        "stage": "download-era5", "config_sha256": config_hash(config),
        "files": {str(path): sha256_file(path) for path in outputs},
        "native_files_retained": not era5.get("delete_native_after_aggregation", True),
        "download_workers": workers,
    })
    return outputs


def download_koppen(config: dict, force: bool = False) -> Path:
    url = config["koppen"]["url"]
    target = project_path("data", "external", "koppen_geiger_1991_2020.tif")
    if target.exists() and target.stat().st_size > 0 and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tif.part")
    request = urllib.request.Request(url, headers={"User-Agent": "ClimateUnsupervised/0.1"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if temporary.stat().st_size == 0:
        raise RuntimeError("Koppen-Geiger download was empty")
    temporary.replace(target)
    write_json(project_path("results", "manifests", "download-koppen.json"), {
        "stage": "download-koppen", "config_sha256": config_hash(config),
        "url": url, "file": str(target), "sha256": sha256_file(target),
    })
    return target
