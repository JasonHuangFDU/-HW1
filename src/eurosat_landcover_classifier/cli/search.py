from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..search import run_hyperparameter_search


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyper-parameter search for the NumPy-only MLP.")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="search_experiments")
    parser.add_argument("--mode", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--num-trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--learning-rates", type=str, default="0.01,0.005,0.002")
    parser.add_argument("--hidden-dims", type=str, default="128,256,512")
    parser.add_argument("--weight-decays", type=str, default="0.0,1e-4,1e-3")
    parser.add_argument("--activations", type=str, default="relu,tanh")
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--decay-every", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable live training progress display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_hyperparameter_search(args)
    for trial_result in result["results"]:
        print(json.dumps(trial_result, ensure_ascii=False))
    print(f"Best result saved to: {Path(args.output_dir) / 'search_results.json'}")


if __name__ == "__main__":
    main()
