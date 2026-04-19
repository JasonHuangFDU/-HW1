from __future__ import annotations

from typing import List, Sequence

import numpy as np


def top_hidden_units_for_class(model, class_names: Sequence[str], class_name: str, top_k: int) -> tuple[List[int], List[str]]:
    if class_name not in class_names:
        raise ValueError(f"Unknown class '{class_name}'. Available classes: {', '.join(class_names)}")

    class_index = class_names.index(class_name)
    class_weights = model.fc2.weight.data[:, class_index]
    ranked_units = np.argsort(class_weights)[::-1][:top_k]
    titles = [f"Unit {int(unit)}\nfc2={class_weights[int(unit)]:.3f}" for unit in ranked_units]
    return ranked_units.tolist(), titles
