from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

from scripts.lab3.config import Lab3Config, PROJECT_ROOT


@dataclass
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


@dataclass
class FullDataBundle:
    splits: SplitIndices
    preprocessor: "TabularPreprocessor"
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray


class TabularPreprocessor:
    """Предобработка табличных признаков с обучением статистик на train."""

    def __init__(self, numeric_features: list[str], categorical_features: list[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        self.numeric_means: dict[str, float] = {}
        self.numeric_scales: dict[str, float] = {}
        self.categorical_modes: dict[str, str] = {}
        self.categorical_categories: dict[str, list[str]] = {}

        self.is_fitted = False

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric_features)

        for column in self.categorical_features:
            names.extend(
                f"{column}_{category}"
                for category in self.categorical_categories[column]
            )

        return names

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        """Обучает предобработку на переданном обучающем наборе."""

        for column in self.numeric_features:
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).to_numpy(dtype=np.float64, copy=True)

            valid_values = values[np.isfinite(values)]

            if len(valid_values) == 0:
                raise ValueError(
                    f"Числовой признак {column} не содержит корректных значений"
                )

            mean = float(valid_values.mean())
            scale = float(valid_values.std(ddof=0))

            self.numeric_means[column] = mean
            self.numeric_scales[column] = scale if scale > 0 else 1.0

        for column in self.categorical_features:
            values = frame[column].astype("string").dropna()
            counts = Counter(values.astype(str).tolist())

            if not counts:
                raise ValueError(
                    f"Категориальный признак {column} не содержит значений"
                )

            self._set_categories(column, counts)

        self.is_fitted = True
        return self

    def fit_streaming(
        self,
        dataset_path: Path,
        train_indices: np.ndarray,
        chunk_size: int,
    ) -> "TabularPreprocessor":
        """Обучает статистики, последовательно читая обучающие строки CSV."""

        numeric_sums = {column: 0.0 for column in self.numeric_features}
        numeric_squared_sums = {column: 0.0 for column in self.numeric_features}
        numeric_counts = {column: 0 for column in self.numeric_features}
        categorical_counts = {column: Counter() for column in self.categorical_features}

        selected_indices = set(train_indices.tolist())
        offset = 0

        columns = self.numeric_features + self.categorical_features

        for chunk in pd.read_csv(
            dataset_path,
            usecols=columns,
            chunksize=chunk_size,
        ):
            global_indices = np.arange(offset, offset + len(chunk))
            mask = np.isin(global_indices, train_indices)
            selected = chunk.iloc[mask]

            offset += len(chunk)

            if selected.empty:
                continue

            for column in self.numeric_features:
                values = pd.to_numeric(
                    selected[column],
                    errors="coerce",
                ).to_numpy(dtype=np.float64, copy=True)

                valid_values = values[np.isfinite(values)]

                numeric_sums[column] += float(valid_values.sum())
                numeric_squared_sums[column] += float(np.square(valid_values).sum())
                numeric_counts[column] += len(valid_values)

            for column in self.categorical_features:
                values = selected[column].astype("string").dropna()
                categorical_counts[column].update(values.astype(str).tolist())

        if not selected_indices:
            raise ValueError("Обучающая выборка не содержит индексов")

        for column in self.numeric_features:
            count = numeric_counts[column]

            if count == 0:
                raise ValueError(
                    f"Числовой признак {column} не содержит корректных значений"
                )

            mean = numeric_sums[column] / count
            variance = numeric_squared_sums[column] / count - mean**2

            scale = float(np.sqrt(max(variance, 0.0)))

            self.numeric_means[column] = float(mean)
            self.numeric_scales[column] = scale if scale > 0 else 1.0

        for column in self.categorical_features:
            counts = categorical_counts[column]

            if not counts:
                raise ValueError(
                    f"Категориальный признак {column} не содержит значений"
                )

            self._set_categories(column, counts)

        self.is_fitted = True
        return self

    def _set_categories(
        self,
        column: str,
        counts: Counter,
    ) -> None:
        """Фиксирует моду и категории в детерминированном порядке."""

        mode = min(
            counts,
            key=lambda value: (-counts[value], value),
        )

        self.categorical_modes[column] = mode
        self.categorical_categories[column] = sorted(counts.keys())

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Преобразует признаки в числовую матрицу float32."""

        if not self.is_fitted:
            raise RuntimeError(
                "Предобработчик не обучен. Сначала вызовите fit или fit_streaming"
            )

        parts = []

        for column in self.numeric_features:
            values = pd.to_numeric(
                frame[column],
                errors="coerce",
            ).to_numpy(dtype=np.float64, copy=True)

            values[~np.isfinite(values)] = self.numeric_means[column]

            scaled = (values - self.numeric_means[column]) / self.numeric_scales[column]

            parts.append(scaled.reshape(-1, 1))

        for column in self.categorical_features:
            values = (
                frame[column]
                .astype("string")
                .fillna(self.categorical_modes[column])
                .astype(str)
                .to_numpy()
            )

            categories = self.categorical_categories[column]

            encoded = np.zeros(
                (len(frame), len(categories)),
                dtype=np.float32,
            )

            for index, category in enumerate(categories):
                encoded[:, index] = (values == category).astype(np.float32)

            parts.append(encoded)

        return np.column_stack(parts).astype(np.float32)


def load_split_indices(config: Lab3Config) -> SplitIndices:
    """Создаёт единое разбиение по индексам строк и целевой переменной."""

    dataset_path = PROJECT_ROOT / config.dataset_path

    target = pd.read_csv(
        dataset_path,
        usecols=[config.target_column],
    )[config.target_column]

    if target.isna().any():
        raise ValueError("Целевая переменная содержит пропущенные значения")

    y = pd.to_numeric(target, errors="raise").to_numpy(dtype=np.int64)

    indices = np.arange(len(y))

    train_validation_indices, test_indices = train_test_split(
        indices,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y[train_validation_indices],
    )

    train_indices = np.sort(train_indices)
    validation_indices = np.sort(validation_indices)
    test_indices = np.sort(test_indices)

    return SplitIndices(
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
        y_train=y[train_indices],
        y_validation=y[validation_indices],
        y_test=y[test_indices],
    )


def load_full_data(
    config: Lab3Config,
    splits: SplitIndices | None = None,
) -> FullDataBundle:
    """Загружает таблицу целиком и формирует обучающие выборки."""

    dataset_path = PROJECT_ROOT / config.dataset_path

    if splits is None:
        splits = load_split_indices(config)

    columns = (
        config.numeric_features + config.categorical_features + [config.target_column]
    )

    frame = pd.read_csv(dataset_path, usecols=columns)

    preprocessor = TabularPreprocessor(
        config.numeric_features,
        config.categorical_features,
    )

    preprocessor.fit(frame.iloc[splits.train])

    X_train = preprocessor.transform(frame.iloc[splits.train])
    X_validation = preprocessor.transform(frame.iloc[splits.validation])
    X_test = preprocessor.transform(frame.iloc[splits.test])

    return FullDataBundle(
        splits=splits,
        preprocessor=preprocessor,
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
    )


class StreamingTitanicDataset(IterableDataset):
    """Потоковый набор данных: читает CSV и выдаёт отдельные строки."""

    def __init__(
        self,
        config: Lab3Config,
        indices: np.ndarray,
        preprocessor: TabularPreprocessor,
        shuffle: bool = False,
    ):
        super().__init__()

        self.dataset_path = PROJECT_ROOT / config.dataset_path
        self.numeric_features = config.numeric_features
        self.categorical_features = config.categorical_features
        self.target_column = config.target_column

        self.indices = indices
        self.preprocessor = preprocessor
        self.chunk_size = config.data.chunk_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator:
        columns = (
            self.numeric_features + self.categorical_features + [self.target_column]
        )

        offset = 0

        for chunk in pd.read_csv(
            self.dataset_path,
            usecols=columns,
            chunksize=self.chunk_size,
        ):
            global_indices = np.arange(offset, offset + len(chunk))
            mask = np.isin(global_indices, self.indices)
            selected = chunk.iloc[mask]

            offset += len(chunk)

            if selected.empty:
                continue

            features = self.preprocessor.transform(selected)
            labels = pd.to_numeric(
                selected[self.target_column],
                errors="raise",
            ).to_numpy(dtype=np.float32)

            X_tensor = torch.from_numpy(features)
            y_tensor = torch.from_numpy(labels)

            if self.shuffle:
                order = torch.randperm(len(y_tensor))
            else:
                order = torch.arange(len(y_tensor))

            for index in order:
                yield X_tensor[index], y_tensor[index]


def make_full_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """Создаёт DataLoader для данных, загруженных в память."""

    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def make_streaming_loader(
    config: Lab3Config,
    indices: np.ndarray,
    preprocessor: TabularPreprocessor,
    shuffle: bool,
) -> DataLoader:
    """Создаёт DataLoader для потокового чтения CSV."""

    dataset = StreamingTitanicDataset(
        config=config,
        indices=indices,
        preprocessor=preprocessor,
        shuffle=shuffle,
    )

    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=config.runtime.num_workers,
    )


def prepare_streaming_preprocessor(
    config: Lab3Config,
    splits: SplitIndices,
) -> TabularPreprocessor:
    """Обучает предобработчик потоковым проходом только по train."""

    preprocessor = TabularPreprocessor(
        config.numeric_features,
        config.categorical_features,
    )

    preprocessor.fit_streaming(
        dataset_path=PROJECT_ROOT / config.dataset_path,
        train_indices=splits.train,
        chunk_size=config.data.chunk_size,
    )

    return preprocessor
