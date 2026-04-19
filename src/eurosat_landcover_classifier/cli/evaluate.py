from __future__ import annotations

import argparse
from pathlib import Path

from ..evaluation import evaluate_checkpoint, format_confusion_matrix
from ..evaluation.visualization import plot_confusion_matrix
from ..utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the EuroSAT test split.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_checkpoint(args.checkpoint, data_dir=args.data_dir, batch_size=args.batch_size)
    checkpoint = result["checkpoint"]
    metrics = result["metrics"]
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(format_confusion_matrix(metrics["confusion_matrix"], checkpoint["class_names"]))

    output_dir = args.output_dir or str(Path(args.checkpoint).resolve().parent / "evaluation")
    ensure_dir(output_dir)
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        checkpoint["class_names"],
        str(Path(output_dir) / "confusion_matrix.png"),
    )


if __name__ == "__main__":
    main()

