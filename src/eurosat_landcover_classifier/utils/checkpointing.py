from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from ..models.mlp import MLPClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_checkpoint(
    path: str,
    model: MLPClassifier,
    config: Dict[str, Any],
    class_names,
    mean: np.ndarray,
    std: np.ndarray,
    best_val_accuracy: float,
) -> None:
    payload = {f"model::{key}": value for key, value in model.state_dict().items()}
    payload["meta::config"] = np.array(json.dumps(config, ensure_ascii=False))
    payload["meta::class_names"] = np.array(json.dumps(list(class_names), ensure_ascii=False))
    payload["meta::mean"] = mean.astype(np.float32)
    payload["meta::std"] = std.astype(np.float32)
    payload["meta::best_val_accuracy"] = np.array(best_val_accuracy, dtype=np.float32)
    np.savez(path, **payload)


def load_checkpoint(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as checkpoint:
        model_state = {
            key.replace("model::", ""): checkpoint[key].astype(np.float32)
            for key in checkpoint.files
            if key.startswith("model::")
        }
        config = json.loads(checkpoint["meta::config"].item())
        class_names = json.loads(checkpoint["meta::class_names"].item())
        mean = checkpoint["meta::mean"].astype(np.float32)
        std = checkpoint["meta::std"].astype(np.float32)
        best_val_accuracy = float(checkpoint["meta::best_val_accuracy"].item())

    return {
        "model_state": model_state,
        "config": config,
        "class_names": class_names,
        "mean": mean,
        "std": std,
        "best_val_accuracy": best_val_accuracy,
    }
