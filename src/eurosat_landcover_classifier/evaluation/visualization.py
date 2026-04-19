from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(history: Dict[str, List[float]], output_path: str) -> None:
    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(epochs, history["lr"], label="lr")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], output_path: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black", fontsize=8)

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
