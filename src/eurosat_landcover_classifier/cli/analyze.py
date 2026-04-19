from __future__ import annotations

import argparse
from typing import List, Tuple

from ..analysis import analyze_checkpoint


def _parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_pair_csv(raw: str) -> List[Tuple[str, str]]:
    pairs = []
    for chunk in raw.split(";"):
        items = [item.strip() for item in chunk.split(",") if item.strip()]
        if not items:
            continue
        if len(items) != 2:
            raise ValueError(f"Invalid pair specification '{chunk}'. Use 'ClassA,ClassB;ClassC,ClassD'.")
        pairs.append((items[0], items[1]))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run weight visualization and misclassification analysis.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--max-errors", type=int, default=16)
    parser.add_argument("--max-units", type=int, default=64)
    parser.add_argument(
        "--weight-classes",
        "--focus-classes",
        dest="weight_classes",
        type=str,
        default="",
        help="Comma-separated class names for class-specific first-layer weight visualization. Leave empty or use 'all' to generate all classes.",
    )
    parser.add_argument(
        "--top-units-per-class",
        type=int,
        default=16,
        help="Number of hidden units to visualize for each focus class.",
    )
    parser.add_argument(
        "--error-pairs",
        type=str,
        default="",
        help="Semicolon-separated class pairs for pairwise misclassification analysis, e.g. 'Highway,River;AnnualCrop,PermanentCrop'.",
    )
    parser.add_argument(
        "--top-error-pairs",
        type=int,
        default=3,
        help="When --error-pairs is empty, analyze the top-k class pairs with the highest pairwise confusion rate.",
    )
    parser.add_argument(
        "--max-errors-per-pair",
        type=int,
        default=12,
        help="Maximum number of misclassified samples to visualize for each analyzed class pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_checkpoint(
        args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        max_errors=args.max_errors,
        max_units=args.max_units,
        weight_classes=_parse_csv(args.weight_classes),
        top_units_per_class=args.top_units_per_class,
        error_pairs=_parse_pair_csv(args.error_pairs),
        top_error_pairs=args.top_error_pairs,
        max_errors_per_pair=args.max_errors_per_pair,
    )

    print(f"Analysis saved to: {result['output_dir']}")
    for item in result["selected_error_pairs"]:
        print(
            f"{item['class_a_name']} <-> {item['class_b_name']}: "
            f"mutual_confusions={item['mutual_confusion_count']} "
            f"samples={item['sample_count']} pair_confusion_rate={item['pair_confusion_rate']:.4f}"
        )


if __name__ == "__main__":
    main()
