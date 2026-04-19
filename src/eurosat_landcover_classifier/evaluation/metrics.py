from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def format_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str]) -> str:
    header = ["true/pred"] + list(class_names)
    rows = ["\t".join(header)]
    for index, class_name in enumerate(class_names):
        values = "\t".join(str(int(value)) for value in matrix[index])
        rows.append(f"{class_name}\t{values}")
    return "\n".join(rows)
