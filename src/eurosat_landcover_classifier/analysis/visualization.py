from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_first_layer_weights(
    weights: np.ndarray,
    image_shape,
    output_path: str,
    max_units: int = 64,
    unit_indices: Sequence[int] | None = None,
    titles: Sequence[str] | None = None,
    figure_title: str = "First Layer Weights",
) -> None:
    if unit_indices is None:
        unit_indices = list(range(min(weights.shape[1], max_units)))
    else:
        unit_indices = list(unit_indices)[:max_units]

    num_units = len(unit_indices)
    if num_units == 0:
        return

    num_cols = min(8, num_units)
    num_rows = math.ceil(num_units / num_cols)

    figure, axes = plt.subplots(num_rows, num_cols, figsize=(2.2 * num_cols, 2.2 * num_rows))
    axes = np.atleast_1d(axes).reshape(num_rows, num_cols)

    for index in range(num_rows * num_cols):
        axis = axes[index // num_cols, index % num_cols]
        axis.axis("off")

        if index >= num_units:
            continue

        unit_index = int(unit_indices[index])
        weight_image = weights[:, unit_index].reshape(image_shape)
        minimum = weight_image.min()
        maximum = weight_image.max()
        normalized = (weight_image - minimum) / (maximum - minimum + 1e-8)
        axis.imshow(normalized)
        if titles is not None and index < len(titles):
            axis.set_title(str(titles[index]), fontsize=8)
        else:
            axis.set_title(f"Unit {unit_index}", fontsize=8)

    figure.suptitle(figure_title, fontsize=14)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_misclassified_samples(
    images: Sequence[np.ndarray],
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    class_names: Sequence[str],
    output_path: str,
    max_items: int = 16,
) -> None:
    num_items = min(len(images), max_items)
    if num_items == 0:
        return

    num_cols = min(4, num_items)
    num_rows = math.ceil(num_items / num_cols)
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(3.5 * num_cols, 3.5 * num_rows))
    axes = np.atleast_1d(axes).reshape(num_rows, num_cols)

    for index in range(num_rows * num_cols):
        axis = axes[index // num_cols, index % num_cols]
        axis.axis("off")
        if index >= num_items:
            continue
        axis.imshow(images[index])
        axis.set_title(
            f"T: {class_names[int(true_labels[index])]}\nP: {class_names[int(pred_labels[index])]}",
            fontsize=9,
        )

    figure.suptitle("Misclassified Test Samples", fontsize=14)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
