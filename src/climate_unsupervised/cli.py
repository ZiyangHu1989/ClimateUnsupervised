from __future__ import annotations

import argparse
from pathlib import Path

from .config import PROJECT_ROOT, load_config


COMMANDS = ("download-era5", "preprocess", "fit", "visualize", "freeze",
            "download-koppen", "evaluate", "all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover global climate regions without supervised labels")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--force", action="store_true", help="Recreate existing stage artifacts where supported")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command in ("download-era5", "all"):
        from .acquire import download_era5
        download_era5(config, args.force)
    if args.command in ("preprocess", "all"):
        from .preprocess import preprocess
        preprocess(config, args.force)
    if args.command in ("fit", "all"):
        from .modeling import fit_models
        fit_models(config, args.force)
    if args.command in ("visualize", "all"):
        from .visualize import visualize
        paths = visualize(config)
        print(f"visualizations_complete count={len(paths)}")
    if args.command in ("freeze", "all"):
        from .freeze import freeze
        path = freeze(config, args.force)
        print(f"freeze_complete path={path}")
    if args.command in ("download-koppen", "all"):
        from .acquire import download_koppen
        download_koppen(config, args.force)
    if args.command in ("evaluate", "all"):
        from .evaluate import evaluate
        path = evaluate(config)
        print(f"evaluation_complete path={path}")
        if args.command == "all":
            from .visualize import visualize
            visualize(config)


if __name__ == "__main__":
    main()
