from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from tqdm.auto import tqdm


from ..data import DataLoader, DatasetBundle, prepare_eurosat
from ..evaluation import evaluate_model
from ..evaluation.visualization import plot_training_history
from ..models import MLPClassifier, Tensor, cross_entropy_loss
from ..utils import ensure_dir, save_checkpoint, save_json, set_seed, timestamp
from .optimizers import SGD, StepLRScheduler


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str = "experiments"
    run_name: str = ""
    seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    epochs: int = 20
    lr: float = 0.01
    hidden_dim: int = 256
    activation: str = "relu"
    weight_decay: float = 1e-4
    lr_decay: float = 0.5
    decay_every: int = 5
    min_lr: float = 1e-5
    max_grad_norm: float = 5.0
    show_progress: bool = True
    evaluate_on_test: bool = True


def _build_loaders(bundle: DatasetBundle, config: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(
        bundle.train_samples,
        batch_size=config.batch_size,
        mean=bundle.mean,
        std=bundle.std,
        shuffle=True,
        seed=config.seed,
    )
    val_loader = DataLoader(
        bundle.val_samples,
        batch_size=config.batch_size,
        mean=bundle.mean,
        std=bundle.std,
        shuffle=False,
        seed=config.seed,
    )
    test_loader = DataLoader(
        bundle.test_samples,
        batch_size=config.batch_size,
        mean=bundle.mean,
        std=bundle.std,
        shuffle=False,
        seed=config.seed,
    )
    return train_loader, val_loader, test_loader


def _write_progress(message: str, enabled: bool) -> None:
    if not enabled:
        return
    if tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def train_model(config: TrainConfig) -> Dict[str, object]:
    set_seed(config.seed)

    bundle = prepare_eurosat(
        data_dir=config.data_dir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    train_loader, val_loader, test_loader = _build_loaders(bundle, config)

    run_name = config.run_name or timestamp()
    run_dir = ensure_dir(str(Path(config.output_dir) / run_name))

    model = MLPClassifier(
        input_dim=bundle.input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=len(bundle.class_names),
        activation=config.activation,
        seed=config.seed,
    )
    optimizer = SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = StepLRScheduler(
        optimizer,
        decay=config.lr_decay,
        decay_every=config.decay_every,
        min_lr=config.min_lr,
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "lr": [],
    }

    best_val_accuracy = -1.0
    best_model_state = None
    best_checkpoint_path = Path(run_dir) / "best_model.npz"
    last_checkpoint_path = Path(run_dir) / "last_model.npz"

    _write_progress(f"Training run: {run_dir}", config.show_progress)

    for epoch in range(1, config.epochs + 1):
        epoch_lr = optimizer.lr
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        batch_iterator = train_loader
        if config.show_progress and tqdm is not None:
            batch_iterator = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch:02d}/{config.epochs:02d}",
                leave=False,
            )

        for x_batch, y_batch in batch_iterator:
            optimizer.zero_grad()
            logits = model(Tensor(x_batch, requires_grad=False))
            loss = cross_entropy_loss(logits, y_batch)
            loss.backward()
            optimizer.clip_grad_norm(config.max_grad_norm)
            optimizer.step()

            predictions = np.argmax(logits.data, axis=1)
            train_loss_sum += float(loss.data) * y_batch.shape[0]
            train_correct += int((predictions == y_batch).sum())
            train_total += y_batch.shape[0]

            if config.show_progress and tqdm is not None:
                batch_iterator.set_postfix(
                    loss=f"{float(loss.data):.4f}",
                    acc=f"{train_correct / max(train_total, 1):.4f}",
                    lr=f"{epoch_lr:.5f}",
                )

        train_loss = train_loss_sum / max(train_total, 1)
        train_accuracy = train_correct / max(train_total, 1)

        model.eval()
        val_metrics = evaluate_model(model, val_loader, num_classes=len(bundle.class_names))
        scheduler.step(epoch)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["lr"].append(epoch_lr)

        if float(val_metrics["accuracy"]) > best_val_accuracy:
            best_val_accuracy = float(val_metrics["accuracy"])
            best_model_state = model.state_dict()
            save_checkpoint(
                str(best_checkpoint_path),
                model=model,
                config=asdict(config),
                class_names=bundle.class_names,
                mean=bundle.mean,
                std=bundle.std,
                best_val_accuracy=best_val_accuracy,
            )

        save_json(str(Path(run_dir) / "history.json"), history)
        _write_progress(
            (
                f"[Epoch {epoch:02d}/{config.epochs:02d}] "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} "
                f"val_loss={float(val_metrics['loss']):.4f} "
                f"val_acc={float(val_metrics['accuracy']):.4f} "
                f"best_val_acc={best_val_accuracy:.4f} "
                f"lr={epoch_lr:.5f}"
            ),
            config.show_progress,
        )

    save_checkpoint(
        str(last_checkpoint_path),
        model=model,
        config=asdict(config),
        class_names=bundle.class_names,
        mean=bundle.mean,
        std=bundle.std,
        best_val_accuracy=best_val_accuracy,
    )
    plot_training_history(history, str(Path(run_dir) / "curves.png"))

    final_test_accuracy = None
    if config.evaluate_on_test:
        best_model = MLPClassifier(
            input_dim=bundle.input_dim,
            hidden_dim=config.hidden_dim,
            num_classes=len(bundle.class_names),
            activation=config.activation,
            seed=config.seed,
        )
        best_model.load_state_dict(best_model_state if best_model_state is not None else model.state_dict())
        best_model.eval()
        test_metrics = evaluate_model(best_model, test_loader, num_classes=len(bundle.class_names))
        final_test_accuracy = float(test_metrics["accuracy"])

    summary = {
        "run_dir": str(run_dir),
        "config": asdict(config),
        "num_classes": len(bundle.class_names),
        "image_shape": list(bundle.image_shape),
        "split_sizes": {
            "train": len(bundle.train_samples),
            "val": len(bundle.val_samples),
            "test": len(bundle.test_samples),
        },
        "best_val_accuracy": best_val_accuracy,
        "final_test_accuracy": final_test_accuracy,
    }
    save_json(str(Path(run_dir) / "summary.json"), summary)
    if final_test_accuracy is not None:
        _write_progress(
            (
                f"Training finished. best_val_acc={best_val_accuracy:.4f} "
                f"final_test_acc={final_test_accuracy:.4f}"
            ),
            config.show_progress,
        )
    else:
        _write_progress(
            f"Training finished. best_val_acc={best_val_accuracy:.4f}",
            config.show_progress,
        )

    return {
        "run_dir": str(run_dir),
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "final_test_accuracy": final_test_accuracy,
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "class_names": bundle.class_names,
        "image_shape": bundle.image_shape,
    }
