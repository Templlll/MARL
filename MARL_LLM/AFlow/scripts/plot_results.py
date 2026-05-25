# -*- coding: utf-8 -*-
"""Plot AFlow evaluation results.

This script reads the JSON summary produced under
workspace/<dataset>/workflows/results.json and writes a simple
round-vs-score figure for reporting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AFlow results")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. HumanEval")
    parser.add_argument(
        "--workspace",
        default="workspace",
        help="AFlow workspace root directory (default: workspace)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to workspace/<dataset>/workflows/results.png",
    )
    return parser.parse_args()


def load_results(results_path: Path) -> List[dict]:
    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {results_path}, got {type(data).__name__}")
    return data


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.workspace) / args.dataset / "workflows"
    results_path = dataset_root / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Cannot find {results_path}")

    results = load_results(results_path)
    if not results:
        raise ValueError(f"No records found in {results_path}")

    rounds = [item.get("round") for item in results]
    scores = [item.get("score") for item in results]

    output_path = Path(args.output) if args.output else dataset_root / "results.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(rounds, scores, marker="o", linewidth=2)
    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.title(f"AFlow Results on {args.dataset}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()