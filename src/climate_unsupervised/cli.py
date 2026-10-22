from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT, load_config


COMMANDS = ("download-era5", "preprocess", "prepare-region", "fit", "visualize", "freeze",
            "download-koppen", "evaluate", "evaluate-full", "regional-all", "multiscale-fit",
            "multiscale-freeze", "multiscale-evaluate", "multiscale-all",
            "feature-prepare", "feature-fit", "feature-evaluate", "feature-all", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover global climate regions without supervised labels")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--force", action="store_true", help="Recreate existing stage artifacts where supported")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "feature-prepare":
        from .feature_engineering import prepare_feature_series
        print(prepare_feature_series(config, args.force))
    if args.command in ("feature-fit", "feature-all"):
        from .feature_experiment import run_feature_experiment
        print(run_feature_experiment(config, args.force))
    if args.command in ("feature-evaluate", "feature-all"):
        from .feature_experiment import evaluate_feature_experiment
        print(evaluate_feature_experiment(config))
    if args.command in ("prepare-region", "regional-all", "multiscale-all"):
        from .region import prepare_region
        path = prepare_region(config, args.force)
        print(f"region_preparation_complete path={path}")
    if args.command in ("download-era5", "all"):
        from .acquire import download_era5
        download_era5(config, args.force)
    if args.command in ("preprocess", "all"):
        from .preprocess import preprocess
        preprocess(config, args.force)
    if args.command in ("fit", "regional-all", "all"):
        from .modeling import fit_models
        fit_models(config, args.force)
    if args.command in ("visualize", "regional-all", "all"):
        from .visualize import visualize
        paths = visualize(config)
        print(f"visualizations_complete count={len(paths)}")
    if args.command in ("freeze", "regional-all", "all"):
        from .freeze import freeze
        path = freeze(config, args.force)
        print(f"freeze_complete path={path}")
    if args.command in ("download-koppen", "all"):
        from .acquire import download_koppen
        download_koppen(config, args.force)
    if args.command in ("evaluate", "regional-all", "all"):
        from .evaluate import evaluate
        path = evaluate(config)
        print(f"evaluation_complete path={path}")
        if args.command in ("all", "regional-all"):
            from .visualize import visualize
            visualize(config)
    if args.command in ("evaluate-full", "regional-all"):
        from .comprehensive import run_comprehensive_evaluation
        path = run_comprehensive_evaluation(config)
        print(f"comprehensive_evaluation_complete report={path}")
    if args.command in ("multiscale-fit", "multiscale-all"):
        from .multiscale import run_multiscale_benchmark
        path = run_multiscale_benchmark(config, args.force)
        print(f"multiscale_fit_complete metrics={path}")
    if args.command in ("multiscale-freeze", "multiscale-all"):
        from .multiscale import freeze_multiscale
        path = freeze_multiscale(config, args.force)
        print(f"multiscale_freeze_complete manifest={path}")
    if args.command in ("multiscale-evaluate", "multiscale-all"):
        from .multiscale import evaluate_multiscale
        path = evaluate_multiscale(config)
        print(f"multiscale_evaluation_complete report={path}")


if __name__ == "__main__":
    main()
