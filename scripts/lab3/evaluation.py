from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from scripts.lab3.config import Lab3Config
from scripts.lab3.data import (
    SplitIndices,
    load_split_indices,
    load_full_data,
    prepare_streaming_preprocessor,
    make_full_loader,
    make_streaming_loader,
)
from scripts.lab3.training import TrainingResult


@dataclass
class EvaluationResult:
    metrics: dict[str, float]
    confidence_intervals: dict[str, tuple[float, float]]
    y_true: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    precision_at_fraction: float,
) -> dict[str, float]:
    """Вычисляет основные метрики бинарной классификации."""

    predictions = (probabilities >= 0.5).astype(np.int64)

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "recall_minority": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
    }

    k = max(
        1,
        int(np.ceil(len(probabilities) * precision_at_fraction)),
    )

    top_indices = np.argsort(probabilities)[-k:]
    precision_at_k = float(y_true[top_indices].mean())

    metrics["precision_at_k"] = precision_at_k

    return metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: Lab3Config,
) -> dict[str, tuple[float, float]]:
    """Рассчитывает стратифицированные bootstrap-доверительные интервалы."""

    rng = np.random.default_rng(config.random_state)

    class_indices = {
        label: np.flatnonzero(y_true == label) for label in np.unique(y_true)
    }

    if len(class_indices) < 2:
        raise ValueError("Для bootstrap-оценки необходимы оба класса")

    metric_samples: dict[str, list[float]] = {}

    for _ in range(config.evaluation.bootstrap_iterations):
        sampled_indices = np.concatenate(
            [
                rng.choice(
                    indices,
                    size=len(indices),
                    replace=True,
                )
                for indices in class_indices.values()
            ]
        )

        rng.shuffle(sampled_indices)

        sample_metrics = calculate_metrics(
            y_true=y_true[sampled_indices],
            probabilities=probabilities[sampled_indices],
            precision_at_fraction=config.evaluation.precision_at_fraction,
        )

        for name, value in sample_metrics.items():
            metric_samples.setdefault(name, []).append(value)

    alpha = 1.0 - config.evaluation.confidence_level

    intervals = {}

    for name, values in metric_samples.items():
        lower, upper = np.quantile(
            values,
            [alpha / 2, 1 - alpha / 2],
        )

        intervals[name] = (
            float(lower),
            float(upper),
        )

    return intervals


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    config: Lab3Config,
) -> EvaluationResult:
    """Вычисляет метрики и доверительные интервалы."""

    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(y_true) != len(probabilities):
        raise ValueError("Количество меток и предсказаний не совпадает")

    if len(y_true) == 0:
        raise ValueError("Тестовая выборка пуста")

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Предсказания содержат NaN или Inf")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Вероятности должны находиться в диапазоне [0, 1]")

    metrics = calculate_metrics(
        y_true,
        probabilities,
        config.evaluation.precision_at_fraction,
    )

    intervals = bootstrap_confidence_intervals(
        y_true,
        probabilities,
        config,
    )

    predictions = (probabilities >= 0.5).astype(np.int64)

    return EvaluationResult(
        metrics=metrics,
        confidence_intervals=intervals,
        y_true=y_true,
        probabilities=probabilities,
        predictions=predictions,
    )


def predict_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Получает вероятности и метки из DataLoader."""

    model = model.to(device)
    model.eval()

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)

            logits = model(features)
            probabilities = torch.sigmoid(logits)

            all_probabilities.append(probabilities.cpu().numpy())
            all_targets.append(targets.numpy())

    if not all_probabilities:
        raise ValueError("DataLoader не вернул тестовые объекты")

    return (
        np.concatenate(all_targets).astype(np.int64),
        np.concatenate(all_probabilities),
    )


def evaluate_full(
    result: TrainingResult,
    config: Lab3Config,
    splits: SplitIndices | None = None,
) -> EvaluationResult:
    """Оценивает модель полного режима на тестовой выборке."""

    if splits is None:
        splits = load_split_indices(config)

    data = load_full_data(config, splits)

    loader = make_full_loader(
        X=data.X_test,
        y=splits.y_test,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.random_state,
        num_workers=config.runtime.num_workers,
    )

    device = torch.device(result.device)

    y_true, probabilities = predict_loader(
        result.model,
        loader,
        device,
    )

    return evaluate_predictions(
        y_true,
        probabilities,
        config,
    )


def evaluate_streaming(
    result: TrainingResult,
    config: Lab3Config,
    splits: SplitIndices | None = None,
) -> EvaluationResult:
    """Оценивает потоковую модель на тестовой выборке."""

    if splits is None:
        splits = load_split_indices(config)

    preprocessor = prepare_streaming_preprocessor(
        config,
        splits,
    )

    loader = make_streaming_loader(
        config=config,
        indices=splits.test,
        preprocessor=preprocessor,
        shuffle=False,
    )

    device = torch.device(result.device)

    y_true, probabilities = predict_loader(
        result.model,
        loader,
        device,
    )

    return evaluate_predictions(
        y_true,
        probabilities,
        config,
    )


def format_evaluation(
    name: str,
    result: EvaluationResult,
) -> pd.DataFrame:
    """Формирует таблицу метрик и доверительных интервалов."""

    rows = []

    for metric, value in result.metrics.items():
        lower, upper = result.confidence_intervals[metric]

        rows.append(
            {
                "metric": metric,
                "value": value,
                "ci_lower": lower,
                "ci_upper": upper,
            }
        )

    return pd.DataFrame(rows).set_index("metric")
