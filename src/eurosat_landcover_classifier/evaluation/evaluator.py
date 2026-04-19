from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from ..data import DataLoader, prepare_eurosat
from ..models import MLPClassifier, softmax
from ..utils import load_checkpoint
from .metrics import accuracy_score, confusion_matrix


def evaluate_model(model: MLPClassifier, loader: DataLoader, num_classes: int) -> Dict[str, np.ndarray]:
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in loader:
        logits = model.forward_array(x_batch)
        probabilities = softmax(logits)
        batch_loss = -np.log(probabilities[np.arange(y_batch.shape[0]), y_batch] + 1e-12).mean()
        predictions = np.argmax(probabilities, axis=1)

        total_loss += float(batch_loss) * y_batch.shape[0]
        total_samples += y_batch.shape[0]
        all_predictions.append(predictions)
        all_targets.append(y_batch)

    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_predictions, axis=0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_true, y_pred, num_classes=num_classes),
    }


def evaluate_checkpoint(checkpoint_path: str, data_dir: str = "", batch_size: int = 128) -> Dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    resolved_data_dir = data_dir or config["data_dir"]

    bundle = prepare_eurosat(
        data_dir=resolved_data_dir,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        seed=config["seed"],
    )

    model = MLPClassifier(
        input_dim=bundle.input_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=len(checkpoint["class_names"]),
        activation=config["activation"],
        seed=config["seed"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    test_loader = DataLoader(
        bundle.test_samples,
        batch_size=batch_size,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        shuffle=False,
        seed=config["seed"],
    )

    metrics = evaluate_model(model, test_loader, num_classes=len(checkpoint["class_names"]))
    return {
        "checkpoint": checkpoint,
        "bundle": bundle,
        "model": model,
        "metrics": metrics,
        "output_root": str(Path(checkpoint_path).resolve().parent),
    }
