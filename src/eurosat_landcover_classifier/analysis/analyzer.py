from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

from ..data import DataLoader
from ..evaluation import evaluate_checkpoint
from ..evaluation.visualization import plot_confusion_matrix
from ..utils import ensure_dir, save_json
from .errors import (
    build_pairwise_error_summary,
    collect_misclassified_samples,
    collect_pairwise_misclassified_samples,
    resolve_error_pairs,
)
from .visualization import plot_first_layer_weights, plot_misclassified_samples
from .weights import top_hidden_units_for_class


def _slugify_class_name(class_name: str) -> str:
    return class_name.lower().replace(" ", "_")


def _slugify_pair(left_name: str, right_name: str) -> str:
    return f"{_slugify_class_name(left_name)}_vs_{_slugify_class_name(right_name)}"


def _resolve_weight_classes(class_names: Sequence[str], weight_classes: Sequence[str] | None) -> list[str]:
    if not weight_classes:
        return list(class_names)

    normalized = [name.strip() for name in weight_classes if name.strip()]
    if not normalized or any(name.lower() == "all" for name in normalized):
        return list(class_names)

    missing = [name for name in normalized if name not in class_names]
    if missing:
        raise ValueError(f"Unknown class names: {', '.join(missing)}. Available classes: {', '.join(class_names)}")

    return list(dict.fromkeys(normalized))


def analyze_checkpoint(
    checkpoint_path: str,
    *,
    data_dir: str = "",
    batch_size: int = 128,
    output_dir: str = "",
    max_errors: int = 16,
    max_units: int = 64,
    weight_classes: Sequence[str] | None = None,
    top_units_per_class: int = 16,
    error_pairs: Sequence[Tuple[str, str]] | None = None,
    top_error_pairs: int = 3,
    max_errors_per_pair: int = 12,
) -> Dict[str, object]:
    result = evaluate_checkpoint(checkpoint_path, data_dir=data_dir, batch_size=batch_size)
    checkpoint = result["checkpoint"]
    config = checkpoint["config"]
    bundle = result["bundle"]
    model = result["model"]
    metrics = result["metrics"]

    resolved_output_dir = output_dir or str(Path(checkpoint_path).resolve().parent / "analysis")
    output_root = ensure_dir(resolved_output_dir)
    weights_dir = ensure_dir(str(output_root / "weights"))
    errors_dir = ensure_dir(str(output_root / "errors"))

    loader = DataLoader(
        bundle.test_samples,
        batch_size=batch_size,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        shuffle=False,
        return_images=True,
        seed=config["seed"],
    )

    global_errors = collect_misclassified_samples(model, loader, max_errors=max_errors)
    pairwise_errors = collect_pairwise_misclassified_samples(
        model,
        loader,
        num_classes=len(checkpoint["class_names"]),
        max_errors_per_pair=max_errors_per_pair,
    )
    matrix = metrics["confusion_matrix"]
    resolved_weight_classes = _resolve_weight_classes(checkpoint["class_names"], weight_classes)

    plot_first_layer_weights(
        model.fc1.weight.data,
        bundle.image_shape,
        str(weights_dir / "first_layer_weights.png"),
        max_units=max_units,
    )
    for class_name in resolved_weight_classes:
        unit_indices, titles = top_hidden_units_for_class(
            model,
            checkpoint["class_names"],
            class_name,
            top_k=top_units_per_class,
        )
        plot_first_layer_weights(
            model.fc1.weight.data,
            bundle.image_shape,
            str(weights_dir / f"{_slugify_class_name(class_name)}_first_layer_weights.png"),
            max_units=top_units_per_class,
            unit_indices=unit_indices,
            titles=titles,
            figure_title=f"{class_name} Class: Top Hidden Units",
        )

    plot_confusion_matrix(matrix, checkpoint["class_names"], str(errors_dir / "confusion_matrix.png"))
    plot_misclassified_samples(
        global_errors["wrong_images"],
        global_errors["wrong_true"],
        global_errors["wrong_pred"],
        checkpoint["class_names"],
        str(errors_dir / "misclassified_examples.png"),
        max_items=max_errors,
    )

    selected_error_pairs = resolve_error_pairs(
        checkpoint["class_names"],
        matrix,
        focus_pairs=error_pairs,
        top_error_pairs=top_error_pairs,
    )
    pairwise_summary = build_pairwise_error_summary(
        checkpoint["class_names"],
        matrix,
        selected_error_pairs,
    )

    for item in pairwise_summary:
        pair_key = (item["class_a_index"], item["class_b_index"])
        bucket = pairwise_errors["per_pair"][pair_key]
        plot_misclassified_samples(
            bucket["wrong_images"],
            bucket["wrong_true"],
            bucket["wrong_pred"],
            checkpoint["class_names"],
            str(errors_dir / f"{_slugify_pair(item['class_a_name'], item['class_b_name'])}_misclassified_examples.png"),
            max_items=max_errors_per_pair,
        )

    all_pair_indices = [
        (left, right)
        for left in range(len(checkpoint["class_names"]))
        for right in range(left + 1, len(checkpoint["class_names"]))
    ]
    all_pair_summary = build_pairwise_error_summary(
        checkpoint["class_names"],
        matrix,
        all_pair_indices,
    )
    all_pair_summary.sort(
        key=lambda item: (item["pair_confusion_rate"], item["mutual_confusion_count"]),
        reverse=True,
    )

    save_json(
        str(errors_dir / "pairwise_misclassification_summary.json"),
        {
            "definition": "(A->B + B->A) / (samples of A + samples of B)",
            "weights_dir": str(weights_dir),
            "errors_dir": str(errors_dir),
            "selected_pairs": [
                {
                    "class_a_name": item["class_a_name"],
                    "class_b_name": item["class_b_name"],
                    "sample_count": item["sample_count"],
                    "correct_count": item["correct_count"],
                    "mutual_confusion_count": item["mutual_confusion_count"],
                    "other_error_count": item["other_error_count"],
                    "pair_confusion_rate": item["pair_confusion_rate"],
                    "a_to_b_count": item["a_to_b_count"],
                    "b_to_a_count": item["b_to_a_count"],
                }
                for item in pairwise_summary
            ],
            "all_pairs": [
                {
                    "class_a_name": item["class_a_name"],
                    "class_b_name": item["class_b_name"],
                    "sample_count": item["sample_count"],
                    "correct_count": item["correct_count"],
                    "mutual_confusion_count": item["mutual_confusion_count"],
                    "other_error_count": item["other_error_count"],
                    "pair_confusion_rate": item["pair_confusion_rate"],
                    "a_to_b_count": item["a_to_b_count"],
                    "b_to_a_count": item["b_to_a_count"],
                }
                for item in all_pair_summary
            ],
        },
    )

    return {
        "output_dir": str(output_root),
        "weights_dir": str(weights_dir),
        "errors_dir": str(errors_dir),
        "weight_classes": resolved_weight_classes,
        "selected_error_pairs": pairwise_summary,
        "class_names": checkpoint["class_names"],
    }
