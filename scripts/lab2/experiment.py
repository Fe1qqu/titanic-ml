import hashlib
import json
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from scripts.lab2.config import load_config
from scripts.lab2.data import load_and_split_data
from scripts.lab2.evaluation import (
    evaluate_baseline,
    evaluate_model,
    evaluate_model_streaming,
)
from scripts.lab2.training import train_full, train_chunked
from scripts.lab2.tuning import (
    tune_hyperparameters,
    tune_hyperparameters_streaming,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def calculate_sha256(path: Path) -> str:
    """Вычисляет SHA-256 файла."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def flatten_metrics(metrics: dict) -> dict:
    """Преобразует метрики и доверительные интервалы в плоский словарь."""

    result = {}

    for name, values in metrics.items():
        result[name] = values["value"]
        result[f"{name}_ci_lower"] = values["ci_lower"]
        result[f"{name}_ci_upper"] = values["ci_upper"]

    return result


def log_experiment(
    mode: str,
    metrics: dict,
    training_result: dict,
    config,
    best_params: dict,
    model_path: Path,
):
    """Регистрирует один эксперимент в MLflow."""

    with mlflow.start_run(run_name=mode) as run:
        params = {
            "mode": mode,
            "model": config.model.name,
            "chunk_size": config.chunk_size,
            "epochs": training_result["epochs"],
            "random_state": config.random_state,
            "test_size": config.test_size,
            "validation_size": config.validation_size,
            "numeric_features": ", ".join(config.numeric_features),
            "categorical_features": ", ".join(config.categorical_features),
        }

        params.update(best_params)
        mlflow.log_params(params)

        mlflow.log_metrics(
            {
                **flatten_metrics(metrics),
                "training_time_seconds": training_result["time"],
                "peak_memory_mb": (training_result["peak_memory"] / 1024**2),
                "peak_memory_delta_mb": (
                    training_result["peak_memory_delta"] / 1024**2
                ),
            }
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="models",
        )

        mlflow.log_dict(
            metrics,
            "metrics_with_confidence_intervals.json",
        )

        return run.info.run_id


def run_worker(mode: str, config):
    """Выполняет один эксперимент в отдельном процессе."""

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_path = output_dir / f".worker_{mode}.json"

    if mode == "full":
        print("Загрузка данных для полного режима...")

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = load_and_split_data(config)

        print("Подбор гиперпараметров полного режима...")

        best_params, tuning_results = tune_hyperparameters(
            config,
            X_train,
            y_train,
            X_val,
            y_val,
        )

        print("Лучшие параметры полного режима:", best_params)

        training_result = train_full(
            config,
            X_train,
            y_train,
            best_params,
        )

        metrics = evaluate_model(
            training_result["model"],
            X_test,
            y_test,
            config,
        )

        baseline_metrics = evaluate_baseline(
            X_train,
            y_train,
            X_test,
            y_test,
            config,
        )

        model_path = output_dir / "ml_model.joblib"

        joblib.dump(
            training_result["model"],
            model_path,
        )

        result = {
            "mode": mode,
            "best_params": best_params,
            "tuning_results": tuning_results,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "training_result": {
                key: value for key, value in training_result.items() if key != "model"
            },
            "model_path": str(model_path),
        }

    elif mode == "chunked":
        print("Подбор гиперпараметров потокового режима...")

        best_params = tune_hyperparameters_streaming(config)

        print("Лучшие параметры потокового режима:", best_params)

        training_result = train_chunked(
            config,
            best_params=best_params,
        )

        metrics = evaluate_model_streaming(
            training_result["model"],
            config,
        )

        model_path = output_dir / "ml_model_chunked.joblib"

        joblib.dump(
            training_result["model"],
            model_path,
        )

        result = {
            "mode": mode,
            "best_params": best_params,
            "metrics": metrics,
            "training_result": {
                key: value for key, value in training_result.items() if key != "model"
            },
            "model_path": str(model_path),
        }

    else:
        raise ValueError(f"Неизвестный режим: {mode}")

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(f"Результаты режима {mode} сохранены.")


def create_metrics_dataframe(
    full_metrics: dict,
    chunk_metrics: dict,
    baseline_metrics: dict,
    full_run_id: str,
    chunk_run_id: str,
):
    """Формирует таблицу метрик для CSV."""

    rows = []

    experiments = [
        ("full", full_metrics, full_run_id),
        ("chunked", chunk_metrics, chunk_run_id),
        ("baseline", baseline_metrics, ""),
    ]

    for mode, metrics, run_id in experiments:
        for name, values in metrics.items():
            rows.append(
                {
                    "mode": mode,
                    "metric": name,
                    "value": values["value"],
                    "ci_lower": values["ci_lower"],
                    "ci_upper": values["ci_upper"],
                    "run_id": run_id,
                }
            )

    return pd.DataFrame(rows)


def create_report(
    config,
    full_result,
    chunk_result,
    full_run_id,
    chunk_run_id,
    model_hash,
):
    """Формирует итоговый Markdown-отчёт."""

    full_metrics = full_result["metrics"]
    chunk_metrics = chunk_result["metrics"]
    baseline_metrics = full_result["baseline_metrics"]

    full_training = full_result["training_result"]
    chunk_training = chunk_result["training_result"]

    lines = [
        "# Лабораторная работа № 2",
        "",
        "## 1. Конфигурация эксперимента",
        "",
        f"- Вариант: {config.variant}",
        f"- Датасет: `{config.dataset_path}`",
        f"- Модель: {config.model.name}",
        f"- Размер чанка: {config.chunk_size}",
        f"- Seed: {config.random_state}",
        f"- Уровень доверия: {config.evaluation.confidence_level:.0%}",
        "",
        "## 2. Подбор гиперпараметров",
        "",
        "Подбор выполнялся отдельно для каждого режима.",
        "",
        "### Полный режим",
        "",
        "```json",
        json.dumps(
            full_result["best_params"],
            ensure_ascii=False,
            indent=4,
        ),
        "```",
        "",
        "### Потоковый режим",
        "",
        "```json",
        json.dumps(
            chunk_result["best_params"],
            ensure_ascii=False,
            indent=4,
        ),
        "```",
        "",
        "## 3. Сравнение моделей",
        "",
        "| Метрика | Полное обучение | Обучение на чанках | Baseline |",
        "|---|---:|---:|---:|",
    ]

    for name in full_metrics:
        full = full_metrics[name]
        chunk = chunk_metrics[name]
        baseline = baseline_metrics[name]

        lines.append(
            f"| {name} "
            f"| {full['value']:.4f} "
            f"[{full['ci_lower']:.4f}; {full['ci_upper']:.4f}] "
            f"| {chunk['value']:.4f} "
            f"[{chunk['ci_lower']:.4f}; {chunk['ci_upper']:.4f}] "
            f"| {baseline['value']:.4f} "
            f"[{baseline['ci_lower']:.4f}; {baseline['ci_upper']:.4f}] |"
        )

    lines.extend(
        [
            "",
            "## 4. Время и использование памяти",
            "",
            "| Показатель | Полное обучение | Обучение на чанках |",
            "|---|---:|---:|",
            (
                f"| Время обучения, с | "
                f"{full_training['time']:.4f} | "
                f"{chunk_training['time']:.4f} |"
            ),
            (
                f"| Пиковый RSS, МБ | "
                f"{full_training['peak_memory'] / 1024**2:.2f} | "
                f"{chunk_training['peak_memory'] / 1024**2:.2f} |"
            ),
            (
                f"| Прирост RSS, МБ | "
                f"{full_training['peak_memory_delta'] / 1024**2:.2f} | "
                f"{chunk_training['peak_memory_delta'] / 1024**2:.2f} |"
            ),
            "",
            (
                "Измерения обучения проводились в отдельных процессах. "
                "Пиковое RSS измерялось периодическим опросом процесса "
                "с интервалом 5 мс."
            ),
            "",
            "## 5. MLflow",
            "",
            f"- Experiment: `{config.mlflow.experiment_name}`",
            f"- Full run ID: `{full_run_id}`",
            f"- Chunked run ID: `{chunk_run_id}`",
            "",
            "## 6. Сохранение моделей",
            "",
            "- Основная модель: `ml_model.joblib`",
            "- Потоковая модель: `ml_model_chunked.joblib`",
            f"- SHA-256 основной модели: `{model_hash}`",
            "",
            "## 7. Выводы",
            "",
            (
                "В работе реализованы два режима обучения SGDClassifier: "
                "на полной обучающей выборке и последовательная обработка "
                "данных небольшими порциями."
            ),
            "",
            (
                "Гиперпараметры подбирались независимо для каждого режима "
                "по F1 на validation-выборке. Тестовая выборка при подборе "
                "не использовалась."
            ),
            "",
            (
                "Потоковый режим на этапах подбора параметров, обучения "
                "и оценки загружает признаки последовательно. Для "
                "стратифицированного разбиения в память загружается "
                "целевой столбец и индексы строк."
            ),
            "",
            (
                "Потоковый режим может требовать больше времени из-за "
                "повторного чтения CSV и последовательной обработки "
                "чанков. Различия метрик могут быть связаны с отдельным "
                "подбором гиперпараметров и способом обновления "
                "классификатора."
            ),
            "",
            (
                "Измерения памяти в данном эксперименте относятся "
                "к этапу обучения модели, а не ко всему циклу эксперимента."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def run_subprocess_worker(mode: str):
    """Запускает режим эксперимента в отдельном процессе."""

    command = [
        sys.executable,
        "-m",
        "scripts.lab2.experiment",
        "--worker",
        mode,
    ]

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    """Запускает эксперименты и сохраняет итоговые артефакты."""

    config = load_config()

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Каждый режим запускается в отдельном процессе.
    print("Запуск полного режима...")
    run_subprocess_worker("full")

    print("\nЗапуск потокового режима...")
    run_subprocess_worker("chunked")

    full_result_path = output_dir / ".worker_full.json"
    chunk_result_path = output_dir / ".worker_chunked.json"

    full_result = json.loads(full_result_path.read_text(encoding="utf-8"))

    chunk_result = json.loads(chunk_result_path.read_text(encoding="utf-8"))

    full_model_path = Path(full_result["model_path"])
    chunk_model_path = Path(chunk_result["model_path"])

    model_hash = calculate_sha256(full_model_path)

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    print("\nРегистрация экспериментов в MLflow...")

    full_run_id = log_experiment(
        "full",
        full_result["metrics"],
        full_result["training_result"],
        config,
        full_result["best_params"],
        full_model_path,
    )

    chunk_run_id = log_experiment(
        "chunked",
        chunk_result["metrics"],
        chunk_result["training_result"],
        config,
        chunk_result["best_params"],
        chunk_model_path,
    )

    metrics_df = create_metrics_dataframe(
        full_result["metrics"],
        chunk_result["metrics"],
        full_result["baseline_metrics"],
        full_run_id,
        chunk_run_id,
    )

    metrics_path = output_dir / "ml_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False,
        encoding="utf-8-sig",
    )

    report = create_report(
        config,
        full_result,
        chunk_result,
        full_run_id,
        chunk_run_id,
        model_hash,
    )

    report_path = output_dir / "lab2_report.md"

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    # Временные файлы результатов больше не нужны.
    full_result_path.unlink(missing_ok=True)
    chunk_result_path.unlink(missing_ok=True)

    print("\nЭксперимент завершён.")
    print(f"MLflow full run ID: {full_run_id}")
    print(f"MLflow chunked run ID: {chunk_run_id}")
    print(f"SHA-256: {model_hash}")

    print("\nСозданные файлы:")

    for path in [
        full_model_path,
        chunk_model_path,
        metrics_path,
        report_path,
    ]:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker_config = load_config()
        run_worker(sys.argv[2], worker_config)
    else:
        main()
