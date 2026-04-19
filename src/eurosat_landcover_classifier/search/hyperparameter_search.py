from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..training import TrainConfig, train_model
from ..utils import ensure_dir, save_json


def parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def build_trials(
    learning_rates: str,
    hidden_dims: str,
    weight_decays: str,
    activations: str,
    mode: str = "grid",
    num_trials: int = 6,
    seed: int = 42,
):
    learning_rate_values = parse_list(learning_rates, float)
    hidden_dim_values = parse_list(hidden_dims, int)
    weight_decay_values = parse_list(weight_decays, float)
    activation_values = parse_list(activations, str)

    search_space = list(itertools.product(learning_rate_values, hidden_dim_values, weight_decay_values, activation_values))
    if mode == "grid":
        return search_space

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(search_space), size=min(num_trials, len(search_space)), replace=False)
    return [search_space[index] for index in indices]


def run_hyperparameter_search(args) -> Dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    results: List[Dict[str, Any]] = []
    best_result = None

    trials = build_trials(
        learning_rates=args.learning_rates,
        hidden_dims=args.hidden_dims,
        weight_decays=args.weight_decays,
        activations=args.activations,
        mode=args.mode,
        num_trials=args.num_trials,
        seed=args.seed,
    )

    for trial_index, (lr, hidden_dim, weight_decay, activation) in enumerate(trials, start=1):
        run_name = f"trial_{trial_index:02d}_lr{lr}_hd{hidden_dim}_wd{weight_decay}_act{activation}".replace(".", "p")
        config = TrainConfig(
            data_dir=args.data_dir,
            output_dir=str(output_dir),
            run_name=run_name,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=lr,
            hidden_dim=hidden_dim,
            activation=activation,
            weight_decay=weight_decay,
            lr_decay=args.lr_decay,
            decay_every=args.decay_every,
            min_lr=args.min_lr,
            max_grad_norm=args.max_grad_norm,
            show_progress=not args.no_progress,
            evaluate_on_test=False,
        )
        result = train_model(config)
        trial_result = {
            "run_name": run_name,
            "lr": lr,
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "activation": activation,
            "best_val_accuracy": float(result["best_val_accuracy"]),
            "best_checkpoint": result["best_checkpoint"],
        }
        results.append(trial_result)
        if best_result is None or trial_result["best_val_accuracy"] > best_result["best_val_accuracy"]:
            best_result = trial_result

    payload = {"results": results, "best_result": best_result}
    save_json(str(Path(output_dir) / "search_results.json"), payload)
    return payload
