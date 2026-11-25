import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import psutil

from scripts.lab3.config import PROJECT_ROOT, load_config
from scripts.lab3.data import (
    load_split_indices,
    load_full_data,
    prepare_streaming_preprocessor,
    make_full_loader,
    make_streaming_loader,
)


def measure_worker(mode: str, dataset_path: str) -> dict:
    """Измеряет подготовку данных и проход по обучающим батчам."""
    config = load_config()
    config = config.model_copy(
        update={"dataset_path": str(Path(dataset_path).resolve())}
    )

    process = psutil.Process()
    baseline_rss = process.memory_info().rss
    peak_rss = [baseline_rss]
    stop_event = threading.Event()

    def monitor_memory():
        while not stop_event.is_set():
            try:
                current_rss = process.memory_info().rss
                peak_rss[0] = max(peak_rss[0], current_rss)
            except psutil.Error:
                break
            stop_event.wait(0.005)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()

    start_time = time.perf_counter()

    try:
        splits = load_split_indices(config)

        if mode == "full":
            bundle = load_full_data(config, splits)

            loader = make_full_loader(
                X=bundle.X_train,
                y=splits.y_train,
                batch_size=config.training.batch_size,
                shuffle=False,
                seed=config.random_state,
                num_workers=config.runtime.num_workers,
            )

        elif mode == "streaming":
            preprocessor = prepare_streaming_preprocessor(config, splits)

            loader = make_streaming_loader(
                config=config,
                indices=splits.train,
                preprocessor=preprocessor,
                shuffle=False,
            )

        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        rows_processed = 0
        batches_processed = 0

        for features, labels in loader:
            rows_processed += len(labels)
            batches_processed += 1

        elapsed_seconds = time.perf_counter() - start_time

    finally:
        stop_event.set()
        monitor.join()

    final_peak = max(peak_rss[0], process.memory_info().rss)

    return {
        "mode": mode,
        "rows_total": len(splits.train) + len(splits.validation) + len(splits.test),
        "rows_train": rows_processed,
        "batches": batches_processed,
        "elapsed_seconds": elapsed_seconds,
        "baseline_rss_mb": baseline_rss / (1024**2),
        "peak_rss_mb": final_peak / (1024**2),
        "peak_increment_mb": max(0, final_peak - baseline_rss) / (1024**2),
    }


def create_expanded_csv(source: Path, destination: Path, factor: int) -> None:
    """Создаёт временный CSV повторением строк исходного файла."""
    content = source.read_bytes()
    header, separator, body = content.partition(b"\n")

    if not separator:
        raise ValueError("Не удалось прочитать заголовок CSV.")

    with destination.open("wb") as file:
        file.write(header + separator)

        for _ in range(factor):
            file.write(body)


def run_worker(mode: str, dataset_path: Path) -> dict:
    """Запускает измерение в отдельном процессе."""
    command = [
        sys.executable,
        "-m",
        "scripts.lab3.memory_benchmark",
        "--worker",
        mode,
        "--dataset",
        str(dataset_path),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Ошибка измерения режима {mode}:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue

    raise RuntimeError(
        f"Не удалось получить JSON-результат режима {mode}:\n{result.stdout}"
    )


def run_benchmark() -> None:
    config = load_config()
    source = PROJECT_ROOT / config.dataset_path
    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    factors = [1, 10, 100]
    rows = []

    with tempfile.TemporaryDirectory(prefix="lab3_memory_") as temp:
        temp_dir = Path(temp)

        for factor in factors:
            if factor == 1:
                dataset_path = source
            else:
                dataset_path = temp_dir / f"titanic_x{factor}.csv"
                create_expanded_csv(source, dataset_path, factor)

            print("\n" + "=" * 60)
            print(f"РАЗМЕР ДАННЫХ: x{factor}")
            print(f"Файл: {dataset_path}")
            print("=" * 60)

            for mode in ["full", "streaming"]:
                print(f"\nИзмерение режима: {mode}")

                result = run_worker(mode, dataset_path)
                result["scale_factor"] = factor
                result["dataset_file_mb"] = dataset_path.stat().st_size / (1024**2)

                rows.append(result)

                print(
                    f"Строк в датасете: {result['rows_total']}\n"
                    f"Обработано обучающих строк: {result['rows_train']}\n"
                    f"Батчей: {result['batches']}\n"
                    f"Время: {result['elapsed_seconds']:.4f} с\n"
                    f"Базовая RAM: {result['baseline_rss_mb']:.2f} MB\n"
                    f"Пиковая RAM: {result['peak_rss_mb']:.2f} MB\n"
                    f"Прирост RAM: {result['peak_increment_mb']:.2f} MB"
                )

    result_frame = pd.DataFrame(rows)

    columns = [
        "mode",
        "scale_factor",
        "rows_total",
        "rows_train",
        "batches",
        "dataset_file_mb",
        "elapsed_seconds",
        "baseline_rss_mb",
        "peak_rss_mb",
        "peak_increment_mb",
    ]

    result_frame = result_frame[columns]

    output_path = output_dir / "memory_compare.csv"
    result_frame.to_csv(output_path, index=False, float_format="%.6f")

    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ ПАМЯТИ И ВРЕМЕНИ")
    print("=" * 60)
    print(result_frame.to_string(index=False))
    print(f"\nРезультаты сохранены: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=["full", "streaming"])
    parser.add_argument("--dataset")
    args = parser.parse_args()

    if args.worker:
        if not args.dataset:
            parser.error("Для worker необходимо указать --dataset.")

        result = measure_worker(args.worker, args.dataset)
        print(json.dumps(result, ensure_ascii=False))
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
