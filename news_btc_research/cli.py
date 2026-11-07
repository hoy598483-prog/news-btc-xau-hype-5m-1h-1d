from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl news metadata, measure BTC volume shocks, rank tokens, and run Fourier analysis."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to JSON configuration")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    result = run_pipeline(config)
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

