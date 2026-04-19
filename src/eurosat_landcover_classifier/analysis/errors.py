from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from ..data import DataLoader
from ..models import MLPClassifier


def collect_misclassified_samples(model: MLPClassifier, loader: DataLoader, max_errors: int = 16) -> Dict[str, object]:
    true_labels = []
    pred_labels = []
    wrong_images = []
    wrong_true = []
    wrong_pred = []

    for x_batch, y_batch, raw_images in loader:
        predictions = model.predict(x_batch)
        true_labels.append(y_batch)
        pred_labels.append(predictions)

        mismatches = np.where(predictions != y_batch)[0]
        for index in mismatches:
            if len(wrong_images) >= max_errors:
                break
            wrong_images.append(raw_images[index])
            wrong_true.append(int(y_batch[index]))
            wrong_pred.append(int(predictions[index]))

    return {
        "y_true": np.concatenate(true_labels, axis=0),
        "y_pred": np.concatenate(pred_labels, axis=0),
        "wrong_images": wrong_images,
        "wrong_true": wrong_true,
        "wrong_pred": wrong_pred,
    }

def collect_pairwise_misclassified_samples(
    model: MLPClassifier,
    loader: DataLoader,
    num_classes: int,
    max_errors_per_pair: int = 16,
) -> Dict[str, object]:
    per_pair = {
        (left, right): {
            "confusion_count": 0,
            "wrong_images": [],
            "wrong_true": [],
            "wrong_pred": [],
        }
        for left in range(num_classes)
        for right in range(left + 1, num_classes)
    }

    for x_batch, y_batch, raw_images in loader:
        predictions = model.predict(x_batch)
        mismatches = np.where(predictions != y_batch)[0]

        for index in mismatches:
            true_class = int(y_batch[index])
            pred_class = int(predictions[index])
            pair = tuple(sorted((true_class, pred_class)))
            if pair[0] == pair[1]:
                continue
            bucket = per_pair[pair]
            bucket["confusion_count"] += 1
            if len(bucket["wrong_images"]) < max_errors_per_pair:
                bucket["wrong_images"].append(raw_images[index])
                bucket["wrong_true"].append(true_class)
                bucket["wrong_pred"].append(pred_class)

    return {"per_pair": per_pair}

def resolve_error_pairs(
    class_names: Sequence[str],
    matrix: np.ndarray,
    focus_pairs: Sequence[Tuple[str, str]] | None = None,
    top_error_pairs: int = 3,
) -> List[Tuple[int, int]]:
    if focus_pairs:
        resolved_pairs = []
        for left_name, right_name in focus_pairs:
            if left_name not in class_names or right_name not in class_names:
                raise ValueError(
                    f"Unknown class pair '{left_name}, {right_name}'. Available classes: {', '.join(class_names)}"
                )
            left_index = class_names.index(left_name)
            right_index = class_names.index(right_name)
            if left_index == right_index:
                raise ValueError(f"Pair must contain two different classes: '{left_name}'")
            resolved_pairs.append(tuple(sorted((left_index, right_index))))
        return list(dict.fromkeys(resolved_pairs))

    pair_scores = []
    row_sums = matrix.sum(axis=1)
    for left in range(len(class_names)):
        for right in range(left + 1, len(class_names)):
            sample_count = int(row_sums[left] + row_sums[right])
            confusion_count = int(matrix[left, right] + matrix[right, left])
            confusion_rate = confusion_count / max(sample_count, 1)
            pair_scores.append((confusion_rate, confusion_count, left, right))

    pair_scores.sort(reverse=True)
    return [
        (left, right)
        for confusion_rate, confusion_count, left, right in pair_scores
        if confusion_count > 0
    ][:top_error_pairs]


def build_pairwise_error_summary(
    class_names: Sequence[str],
    matrix: np.ndarray,
    selected_pairs: Sequence[Tuple[int, int]],
) -> List[Dict[str, object]]:
    summary = []
    row_sums = matrix.sum(axis=1)

    for left, right in selected_pairs:
        sample_count = int(row_sums[left] + row_sums[right])
        correct_count = int(matrix[left, left] + matrix[right, right])
        confusion_count = int(matrix[left, right] + matrix[right, left])
        other_error_count = sample_count - correct_count - confusion_count
        confusion_rate = confusion_count / max(sample_count, 1)

        summary.append(
            {
                "class_a_index": int(left),
                "class_a_name": class_names[left],
                "class_b_index": int(right),
                "class_b_name": class_names[right],
                "sample_count": sample_count,
                "correct_count": correct_count,
                "mutual_confusion_count": confusion_count,
                "other_error_count": int(other_error_count),
                "pair_confusion_rate": float(confusion_rate),
                "a_to_b_count": int(matrix[left, right]),
                "b_to_a_count": int(matrix[right, left]),
            }
        )

    return summary
