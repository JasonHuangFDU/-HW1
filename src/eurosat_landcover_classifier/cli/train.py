from __future__ import annotations

import argparse

from ..training import TrainConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy-only MLP classifier on EuroSAT.")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--decay-every", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--no-progress", action="store_true", help="Disable live training progress display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        activation=args.activation,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        decay_every=args.decay_every,
        min_lr=args.min_lr,
        max_grad_norm=args.max_grad_norm,
        show_progress=not args.no_progress,
    )
    result = train_model(config)
    print(f"Run directory: {result['run_dir']}")
    print(f"Best checkpoint: {result['best_checkpoint']}")
    print(f"Best val accuracy: {result['best_val_accuracy']:.4f}")
    if result["final_test_accuracy"] is not None:
        print(f"Final test accuracy: {result['final_test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
