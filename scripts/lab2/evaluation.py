import math

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from scripts.lab2.config import Lab2Config
from scripts.lab2.data import iter_csv_split, load_split_indices


def calculate_metrics(
    y_true,
    y_pred,
    y_score,
    precision_at_fraction: float,
):
    """Вычисляет основные метрики классификации."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    k = max(
        1,
        math.ceil(len(y_true) * precision_at_fraction),
    )

    # Учитываем объекты с одинаковыми оценками пропорционально.
    threshold = np.sort(y_score)[-k]

    above_threshold = y_score > threshold
    at_threshold = y_score == threshold

    selected_count = k - np.sum(above_threshold)

    precision_at_k = (
        np.sum(y_true[above_threshold]) + selected_count * np.mean(y_true[at_threshold])
    ) / k

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, y_score),
        "recall_minority": recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        ),
        "precision_at_k": precision_at_k,
    }


def bootstrap_confidence_intervals(
    y_true,
    y_pred,
    y_score,
    config: Lab2Config,
):
    """Вычисляет стратифицированные bootstrap CI для метрик."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    rng = np.random.default_rng(config.random_state)

    classes = np.unique(y_true)

    class_indices = {label: np.flatnonzero(y_true == label) for label in classes}

    metric_samples = {}

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

        metrics = calculate_metrics(
            y_true[sampled_indices],
            y_pred[sampled_indices],
            y_score[sampled_indices],
            config.evaluation.precision_at_fraction,
        )

        for name, value in metrics.items():
            metric_samples.setdefault(name, []).append(value)

    alpha = 1 - config.evaluation.confidence_level

    results = {}

    point_metrics = calculate_metrics(
        y_true,
        y_pred,
        y_score,
        config.evaluation.precision_at_fraction,
    )

    for name, point_value in point_metrics.items():
        values = metric_samples[name]

        lower = np.quantile(values, alpha / 2)
        upper = np.quantile(values, 1 - alpha / 2)

        results[name] = {
            "value": float(point_value),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
        }

    return results


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Lab2Config,
):
    """Оценивает обученную модель на тестовой выборке."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return bootstrap_confidence_intervals(
        y_test,
        predictions,
        probabilities,
        config,
    )


def evaluate_model_streaming(
    model,
    config: Lab2Config,
):
    """Оценивает модель, последовательно обрабатывая тестовые чанки."""

    split = load_split_indices(config)
    test_indices = split["test_indices"]

    y_true_parts = []
    y_pred_parts = []
    y_score_parts = []

    # Определяем индекс положительного класса.
    classes = model.classes_
    positive_class_index = np.flatnonzero(classes == 1)

    if len(positive_class_index) == 0:
        raise ValueError("Модель не содержит положительный класс 1")

    positive_class_index = positive_class_index[0]

    for X_chunk, y_chunk in iter_csv_split(
        config,
        test_indices,
    ):
        predictions = model.predict(X_chunk)

        probabilities = model.predict_proba(X_chunk)[:, positive_class_index]

        y_true_parts.append(y_chunk.to_numpy())
        y_pred_parts.append(predictions)
        y_score_parts.append(probabilities)

    if not y_true_parts:
        raise ValueError("Тестовая выборка не содержит данных")

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    y_score = np.concatenate(y_score_parts)

    return bootstrap_confidence_intervals(
        y_true,
        y_pred,
        y_score,
        config,
    )


def evaluate_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Lab2Config,
):
    """Обучает и оценивает наивный baseline."""

    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(X_train, y_train)

    return evaluate_model(
        baseline,
        X_test,
        y_test,
        config,
    )


def print_metrics(title: str, metrics: dict):
    """Выводит метрики с доверительными интервалами."""

    print(f"\n{title}")

    for name, values in metrics.items():
        print(
            f"{name:20s}: "
            f"{values['value']:.4f} "
            f"[{values['ci_lower']:.4f}; "
            f"{values['ci_upper']:.4f}]"
        )


if __name__ == "__main__":
    from scripts.lab2.config import load_config
    from scripts.lab2.data import load_and_split_data
    from scripts.lab2.tuning import tune_hyperparameters
    from scripts.lab2.training import train_full, train_chunked

    config = load_config()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split_data(config)

    best_params, _ = tune_hyperparameters(
        config,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print("Лучшие параметры:", best_params)

    full_result = train_full(
        config,
        X_train,
        y_train,
        best_params,
    )

    chunk_result = train_chunked(
        config,
        X_train,
        y_train,
        best_params,
    )

    full_metrics = evaluate_model(
        full_result["model"],
        X_test,
        y_test,
        config,
    )

    chunk_metrics = evaluate_model_streaming(
        chunk_result["model"],
        config,
    )

    baseline_metrics = evaluate_baseline(
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )

    print_metrics("Полное обучение", full_metrics)
    print_metrics("Обучение на чанках", chunk_metrics)
    print_metrics("Наивный baseline", baseline_metrics)
