"""EuroSAT land-cover classification package built with NumPy only."""

from .analysis import (
    analyze_checkpoint,
    build_pairwise_error_summary,
    collect_misclassified_samples,
    collect_pairwise_misclassified_samples,
    plot_first_layer_weights,
    plot_misclassified_samples,
    resolve_error_pairs,
    top_hidden_units_for_class,
)
from .data import DataLoader, DatasetBundle, Sample, prepare_eurosat
from .evaluation import (
    accuracy_score,
    confusion_matrix,
    evaluate_checkpoint,
    evaluate_model,
    format_confusion_matrix,
)
from .models import MLPClassifier, Parameter, Tensor, cross_entropy_loss, softmax
from .search import run_hyperparameter_search
from .training import SGD, StepLRScheduler, TrainConfig, train_model
from .utils import ensure_dir, load_checkpoint, save_checkpoint, save_json, set_seed, timestamp

__all__ = [
    "DataLoader",
    "DatasetBundle",
    "Sample",
    "prepare_eurosat",
    "analyze_checkpoint",
    "accuracy_score",
    "build_pairwise_error_summary",
    "collect_misclassified_samples",
    "collect_pairwise_misclassified_samples",
    "confusion_matrix",
    "evaluate_checkpoint",
    "evaluate_model",
    "format_confusion_matrix",
    "MLPClassifier",
    "Parameter",
    "Tensor",
    "cross_entropy_loss",
    "softmax",
    "plot_first_layer_weights",
    "plot_misclassified_samples",
    "resolve_error_pairs",
    "run_hyperparameter_search",
    "SGD",
    "StepLRScheduler",
    "top_hidden_units_for_class",
    "TrainConfig",
    "train_model",
    "ensure_dir",
    "load_checkpoint",
    "save_checkpoint",
    "save_json",
    "set_seed",
    "timestamp",
]
