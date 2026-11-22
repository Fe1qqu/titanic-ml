import threading
import time

import numpy as np
import pandas as pd
import psutil

from sklearn.pipeline import Pipeline

from scripts.lab2.config import Lab2Config
from scripts.lab2.data import (
    iter_csv_split,
    load_and_split_data,
    load_split_indices,
)
from scripts.lab2.pipeline import (
    build_classifier,
    build_preprocessor,
)


class PeakMemoryMonitor:
    """Измеряет пиковое RSS процесса с периодическим опросом."""

    def __init__(self, interval: float = 0.005):
        self.process = psutil.Process()
        self.interval = interval
        self.peak_memory = self.process.memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True,
        )

    def _monitor(self):
        while not self.stop_event.is_set():
            current = self.process.memory_info().rss
            self.peak_memory = max(
                self.peak_memory,
                current,
            )
            self.stop_event.wait(self.interval)

    def start(self):
        self.memory_before = self.process.memory_info().rss
        self.peak_memory = self.memory_before
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join()

        self.memory_after = self.process.memory_info().rss
        self.peak_memory = max(
            self.peak_memory,
            self.memory_after,
        )


def _prepare_classifier(config, best_params):
    """Создаёт классификатор с выбранными гиперпараметрами."""

    classifier = build_classifier(config)

    classifier_params = {
        key.removeprefix("classifier__"): value for key, value in best_params.items()
    }

    classifier.set_params(**classifier_params)

    classifier.set_params(
        tol=None,
        shuffle=False,
    )

    return classifier


def _memory_result(monitor):
    """Формирует информацию об использовании памяти."""

    return {
        "memory_before": monitor.memory_before,
        "memory_after": monitor.memory_after,
        "peak_memory": monitor.peak_memory,
        "peak_memory_delta": max(
            0,
            monitor.peak_memory - monitor.memory_before,
        ),
    }


def train_full(
    config: Lab2Config,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    epochs: int = 10,
):
    """Обучает модель на полной обучающей выборке."""

    X_train = X_train.sort_index()
    y_train = y_train.loc[X_train.index]

    preprocessor = build_preprocessor(config)
    classifier = _prepare_classifier(config, best_params)

    monitor = PeakMemoryMonitor()
    monitor.start()

    start_time = time.perf_counter()

    try:
        preprocessor.fit(X_train)

        X_transformed = preprocessor.transform(X_train)

        for epoch in range(epochs):
            if epoch == 0:
                classifier.partial_fit(
                    X_transformed,
                    y_train,
                    classes=np.unique(y_train),
                )
            else:
                classifier.partial_fit(
                    X_transformed,
                    y_train,
                )
    finally:
        elapsed_time = time.perf_counter() - start_time
        monitor.stop()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return {
        "model": model,
        "time": elapsed_time,
        "epochs": epochs,
        **_memory_result(monitor),
    }


def train_chunked(
    config: Lab2Config,
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None,
    best_params: dict = None,
    epochs: int = 10,
):
    """Обучает модель потоковой обработкой CSV-чанков.

    X_train и y_train оставлены для совместимости со старыми вызовами,
    но в потоковом режиме не используются.
    """

    if best_params is None:
        raise ValueError("Необходимо передать best_params")

    preprocessor = build_preprocessor(config)
    classifier = _prepare_classifier(config, best_params)

    monitor = PeakMemoryMonitor()
    monitor.start()

    start_time = time.perf_counter()

    try:
        # Разбиение формируется по целевому столбцу.
        # Полная таблица признаков не загружается.
        split = load_split_indices(config)

        train_indices = split["train_indices"]
        classes = np.unique(split["y_train"])

        # Первый проход: накопление статистик предобработки.
        for X_chunk, _ in iter_csv_split(
            config,
            train_indices,
        ):
            preprocessor.partial_fit(X_chunk)

        preprocessor._finalize()

        # Второй проход: последовательное обучение классификатора.
        first_batch = True

        for epoch in range(epochs):
            for X_chunk, y_chunk in iter_csv_split(
                config,
                train_indices,
            ):
                X_transformed = preprocessor.transform(X_chunk)

                if first_batch:
                    classifier.partial_fit(
                        X_transformed,
                        y_chunk,
                        classes=classes,
                    )
                    first_batch = False
                else:
                    classifier.partial_fit(
                        X_transformed,
                        y_chunk,
                    )

        if first_batch:
            raise ValueError("Обучающая выборка не содержит данных")

    finally:
        elapsed_time = time.perf_counter() - start_time
        monitor.stop()

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return {
        "model": model,
        "time": elapsed_time,
        "epochs": epochs,
        "train_rows": len(train_indices),
        **_memory_result(monitor),
    }


if __name__ == "__main__":
    from scripts.lab2.config import load_config
    from scripts.lab2.tuning import tune_hyperparameters

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

    print("\nПолное обучение:")
    print(f"Эпох: {full_result['epochs']}")
    print(f"Время: {full_result['time']:.4f} сек.")
    print(f"Пиковая память: {full_result['peak_memory'] / 1024**2:.2f} MB")
    print(f"Прирост RSS: {full_result['peak_memory_delta'] / 1024**2:.2f} MB")

    # X_train и y_train передаются только для совместимости.
    # train_chunked самостоятельно получает индексы и читает CSV.
    chunk_result = train_chunked(
        config,
        X_train,
        y_train,
        best_params,
    )

    print("\nОбучение на чанках:")
    print(f"Эпох: {chunk_result['epochs']}")
    print(f"Размер чанка: {config.chunk_size}")
    print(f"Обработано строк: {chunk_result['train_rows']}")
    print(f"Время: {chunk_result['time']:.4f} сек.")
    print(f"Пиковая память: {chunk_result['peak_memory'] / 1024**2:.2f} MB")
    print(f"Прирост RSS: {chunk_result['peak_memory_delta'] / 1024**2:.2f} MB")
