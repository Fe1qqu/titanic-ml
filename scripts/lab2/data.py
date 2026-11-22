from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.lab2.config import Lab2Config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_and_split_data(config: Lab2Config):
    """Загружает датасет и разделяет его на train, validation и test."""

    dataset_path = PROJECT_ROOT / config.dataset_path
    data = pd.read_csv(dataset_path)

    features = config.numeric_features + config.categorical_features
    required_columns = features + [config.target_column]

    missing_columns = set(required_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(f"В датасете отсутствуют столбцы: {sorted(missing_columns)}")

    data = data[required_columns].copy()

    if data[config.target_column].isna().any():
        raise ValueError("Целевая переменная содержит пропуски")

    X = data[features]
    y = data[config.target_column].astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y_train_val,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )


def load_split_indices(config: Lab2Config) -> dict:
    """Формирует разбиение по индексам, загружая только целевой столбец."""

    dataset_path = PROJECT_ROOT / config.dataset_path

    y = pd.read_csv(
        dataset_path,
        usecols=[config.target_column],
    )[config.target_column]

    if y.isna().any():
        raise ValueError("Целевая переменная содержит пропуски")

    y = y.astype(int)
    indices = np.arange(len(y))

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    train_indices, validation_indices = train_test_split(
        train_val_indices,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y.iloc[train_val_indices],
    )

    train_indices = np.sort(train_indices)
    validation_indices = np.sort(validation_indices)
    test_indices = np.sort(test_indices)

    return {
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "y_train": y.iloc[train_indices].reset_index(drop=True),
        "y_validation": y.iloc[validation_indices].reset_index(drop=True),
        "y_test": y.iloc[test_indices].reset_index(drop=True),
    }


def iter_csv_split(
    config: Lab2Config,
    indices: np.ndarray,
) -> Iterator[tuple[pd.DataFrame, pd.Series]]:
    """Последовательно возвращает признаки и целевые значения выбранных строк."""

    dataset_path = PROJECT_ROOT / config.dataset_path
    features = config.numeric_features + config.categorical_features

    index_set = set(indices)

    for chunk in pd.read_csv(
        dataset_path,
        chunksize=config.chunk_size,
    ):
        selected = chunk.loc[chunk.index.isin(index_set)]

        if selected.empty:
            continue

        X_chunk = selected[features].copy()
        y_chunk = selected[config.target_column].astype(int).copy()

        yield X_chunk, y_chunk


if __name__ == "__main__":
    from scripts.lab2.config import load_config

    config = load_config()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split_data(config)

    print("Полное разбиение данных:")
    print(f"Обучающая выборка: {len(X_train)}")
    print(f"Проверочная выборка: {len(X_val)}")
    print(f"Тестовая выборка: {len(X_test)}")

    split = load_split_indices(config)

    print("\nПотоковое разбиение данных:")
    print(f"Обучающая выборка: {len(split['train_indices'])}")
    print(f"Проверочная выборка: {len(split['validation_indices'])}")
    print(f"Тестовая выборка: {len(split['test_indices'])}")
