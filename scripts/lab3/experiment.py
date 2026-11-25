import hashlib
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.lab3.config import PROJECT_ROOT, load_config
from scripts.lab3.evaluation import (
    EvaluationResult,
    evaluate_full,
    evaluate_streaming,
)
from scripts.lab3.model import TitanicMLP
from scripts.lab3.training import (
    TrainingResult,
    train_full,
    train_streaming,
)


def get_input_dim(model: TitanicMLP) -> int:
    """Определяет размер входного слоя MLP."""

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            return layer.in_features

    raise ValueError("В модели не найден линейный слой")


def save_checkpoint(
    path: Path,
    config,
    full_result: TrainingResult,
    streaming_result: TrainingResult,
) -> str:
    """Сохраняет обе модели в одном checkpoint и возвращает SHA256."""

    checkpoint = {
        "format_version": 1,
        "lab_number": config.lab_number,
        "variant": config.variant,
        "random_state": config.random_state,
        "architecture": config.model.model_dump(),
        "full": {
            "input_dim": get_input_dim(full_result.model),
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in full_result.model.state_dict().items()
            },
            "best_epoch": full_result.best_epoch,
            "best_validation_loss": full_result.best_validation_loss,
        },
        "streaming": {
            "input_dim": get_input_dim(streaming_result.model),
            "state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in streaming_result.model.state_dict().items()
            },
            "best_epoch": streaming_result.best_epoch,
            "best_validation_loss": streaming_result.best_validation_loss,
        },
    }

    torch.save(checkpoint, path)

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    return sha256


def load_checkpoint_model(
    path: Path,
    mode: str,
    config,
    device: str,
) -> TitanicMLP:
    """Загружает выбранную модель из общего checkpoint."""

    if mode not in {"full", "streaming"}:
        raise ValueError(f"Неизвестный режим: {mode}")

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model_info = checkpoint[mode]

    model = TitanicMLP(
        input_dim=model_info["input_dim"],
        config=config.model,
    )

    model.load_state_dict(model_info["state_dict"])
    model.to(device)
    model.eval()

    return model


def save_history(
    result: TrainingResult,
    path: Path,
) -> None:
    """Сохраняет историю обучения в CSV."""

    pd.DataFrame(result.history).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def save_loss_curve(
    result: TrainingResult,
    mode: str,
    path: Path,
) -> None:
    """Строит и сохраняет график функции потерь."""

    history = pd.DataFrame(result.history)

    plt.figure(figsize=(9, 5))

    plt.plot(
        history["epoch"],
        history["train_loss"],
        label="Train loss",
    )
    plt.plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation loss",
    )

    plt.axvline(
        result.best_epoch,
        linestyle="--",
        label=f"Best epoch: {result.best_epoch}",
    )

    plt.title(f"Loss curves — {mode}")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_metrics_rows(
    mode: str,
    evaluation: EvaluationResult,
    run_id: str,
) -> list[dict]:
    """Формирует строки итоговой таблицы метрик."""

    rows = []

    for metric, value in evaluation.metrics.items():
        lower, upper = evaluation.confidence_intervals[metric]

        rows.append(
            {
                "mode": mode,
                "metric": metric,
                "value": value,
                "ci_lower": lower,
                "ci_upper": upper,
                "run_id": run_id,
            }
        )

    return rows


def get_mlflow_params(config, result: TrainingResult, mode: str) -> dict:
    """Формирует параметры эксперимента для MLflow."""

    params = {
        "mode": mode,
        "variant": config.variant,
        "random_state": config.random_state,
        "architecture": config.model.architecture,
        "hidden_dims": config.model.hidden_dims,
        "dropout": config.model.dropout,
        "batch_size": config.training.batch_size,
        "max_epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
        "early_stopping_patience": config.training.early_stopping_patience,
        "early_stopping_min_delta": config.training.early_stopping_min_delta,
        "chunk_size": config.data.chunk_size,
        "device": result.device,
        "parameter_count": result.parameter_count,
        "epochs_trained": result.epochs_trained,
        "best_epoch": result.best_epoch,
    }

    return {key: str(value) for key, value in params.items()}


def get_mlflow_metrics(
    result: TrainingResult,
    evaluation: EvaluationResult,
) -> dict[str, float]:
    """Формирует метрики эксперимента для MLflow."""

    metrics = {
        f"test_{name}": float(value) for name, value in evaluation.metrics.items()
    }

    for name, (lower, upper) in evaluation.confidence_intervals.items():
        metrics[f"test_{name}_ci_lower"] = float(lower)
        metrics[f"test_{name}_ci_upper"] = float(upper)

    metrics.update(
        {
            "best_validation_loss": result.best_validation_loss,
            "best_epoch": float(result.best_epoch),
            "epochs_trained": float(result.epochs_trained),
            "training_time_seconds": result.elapsed_seconds,
            "parameter_count": float(result.parameter_count),
        }
    )

    return metrics


def log_experiment(
    mode: str,
    config,
    result: TrainingResult,
    evaluation: EvaluationResult,
    checkpoint_path: Path,
    history_path: Path,
    curve_path: Path,
) -> str:
    """Создаёт MLflow Run и записывает параметры, метрики и артефакты."""

    with mlflow.start_run(run_name=f"Lab3_{mode}") as run:
        mlflow.set_tags(
            {
                "lab": "3",
                "variant": config.variant,
                "loading_mode": mode,
                "framework": "PyTorch",
            }
        )

        mlflow.log_params(get_mlflow_params(config, result, mode))

        mlflow.log_metrics(get_mlflow_metrics(result, evaluation))

        mlflow.log_artifact(str(checkpoint_path), artifact_path="model")
        mlflow.log_artifact(str(history_path), artifact_path="training")
        mlflow.log_artifact(str(curve_path), artifact_path="plots")

        return run.info.run_id


def verify_checkpoint(
    checkpoint_path: Path,
    config,
    full_result: TrainingResult,
    streaming_result: TrainingResult,
    full_evaluation: EvaluationResult,
    streaming_evaluation: EvaluationResult,
) -> dict[str, float]:
    """Проверяет совпадение предсказаний после загрузки checkpoint."""

    full_model = load_checkpoint_model(
        checkpoint_path,
        "full",
        config,
        full_result.device,
    )

    streaming_model = load_checkpoint_model(
        checkpoint_path,
        "streaming",
        config,
        streaming_result.device,
    )

    loaded_full_result = replace(
        full_result,
        model=full_model,
    )

    loaded_streaming_result = replace(
        streaming_result,
        model=streaming_model,
    )

    loaded_full_evaluation = evaluate_full(
        loaded_full_result,
        config,
    )

    loaded_streaming_evaluation = evaluate_streaming(
        loaded_streaming_result,
        config,
    )

    full_diff = float(
        np.max(
            np.abs(full_evaluation.probabilities - loaded_full_evaluation.probabilities)
        )
    )

    streaming_diff = float(
        np.max(
            np.abs(
                streaming_evaluation.probabilities
                - loaded_streaming_evaluation.probabilities
            )
        )
    )

    return {
        "full_max_probability_difference": full_diff,
        "streaming_max_probability_difference": streaming_diff,
    }


def run_experiment() -> None:
    """Запускает полный эксперимент ЛР3."""

    config = load_config()

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ FULL")
    print("=" * 60)

    full_result = train_full(config)

    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ STREAMING")
    print("=" * 60)

    streaming_result = train_streaming(config)

    print("\n" + "=" * 60)
    print("ОЦЕНКА FULL")
    print("=" * 60)

    full_evaluation = evaluate_full(
        full_result,
        config,
    )

    print(full_evaluation.metrics)

    print("\n" + "=" * 60)
    print("ОЦЕНКА STREAMING")
    print("=" * 60)

    streaming_evaluation = evaluate_streaming(
        streaming_result,
        config,
    )

    print(streaming_evaluation.metrics)

    checkpoint_path = output_dir / "dl_model.pt"

    sha256 = save_checkpoint(
        checkpoint_path,
        config,
        full_result,
        streaming_result,
    )

    full_history_path = output_dir / "training_history_full.csv"
    streaming_history_path = output_dir / "training_history_streaming.csv"

    full_curve_path = output_dir / "loss_curve_full.png"
    streaming_curve_path = output_dir / "loss_curve_streaming.png"

    save_history(full_result, full_history_path)
    save_history(streaming_result, streaming_history_path)

    save_loss_curve(
        full_result,
        "full",
        full_curve_path,
    )

    save_loss_curve(
        streaming_result,
        "streaming",
        streaming_curve_path,
    )

    print("\n" + "=" * 60)
    print("ПРОВЕРКА CHECKPOINT")
    print("=" * 60)

    checkpoint_differences = verify_checkpoint(
        checkpoint_path,
        config,
        full_result,
        streaming_result,
        full_evaluation,
        streaming_evaluation,
    )

    for name, value in checkpoint_differences.items():
        print(f"{name}: {value:.12f}")

    tolerance = config.reproducibility.device_parity_tolerance

    if any(value > tolerance for value in checkpoint_differences.values()):
        raise RuntimeError(
            "Предсказания после загрузки checkpoint отличаются "
            "от исходных больше допустимого значения"
        )

    print("Checkpoint verification: PASS")
    print(f"SHA256: {sha256}")

    print("\n" + "=" * 60)
    print("ЛОГИРОВАНИЕ MLFLOW")
    print("=" * 60)

    full_run_id = log_experiment(
        "full",
        config,
        full_result,
        full_evaluation,
        checkpoint_path,
        full_history_path,
        full_curve_path,
    )

    streaming_run_id = log_experiment(
        "streaming",
        config,
        streaming_result,
        streaming_evaluation,
        checkpoint_path,
        streaming_history_path,
        streaming_curve_path,
    )

    metrics_rows = make_metrics_rows(
        "full",
        full_evaluation,
        full_run_id,
    ) + make_metrics_rows(
        "streaming",
        streaming_evaluation,
        streaming_run_id,
    )

    metrics_path = output_dir / "dl_metrics.csv"

    pd.DataFrame(metrics_rows).to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    run_info = {
        "experiment_name": config.mlflow.experiment_name,
        "tracking_uri": config.mlflow.tracking_uri,
        "full_run_id": full_run_id,
        "streaming_run_id": streaming_run_id,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256,
        "checkpoint_verification": checkpoint_differences,
    }

    run_info_path = output_dir / "run_info.json"

    run_info_path.write_text(
        json.dumps(
            run_info,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    # Добавляем общие таблицу метрик и сведения о запусках
    # в оба MLflow Run.
    for run_id in (full_run_id, streaming_run_id):
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(str(metrics_path), artifact_path="results")
            mlflow.log_artifact(str(run_info_path), artifact_path="results")

    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЁН")
    print("=" * 60)
    print(f"Full Run ID:      {full_run_id}")
    print(f"Streaming Run ID: {streaming_run_id}")
    print(f"Checkpoint SHA256: {sha256}")
    print(f"Результаты: {output_dir}")


if __name__ == "__main__":
    run_experiment()
