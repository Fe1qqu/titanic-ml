import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "lab3.json"


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictConfig):
    chunk_size: int = Field(gt=0)
    loading_modes: list[Literal["full", "streaming"]] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_loading_modes(self):
        if set(self.loading_modes) != {"full", "streaming"}:
            raise ValueError(
                "Для лабораторной работы обязательны режимы full и streaming"
            )

        if len(self.loading_modes) != len(set(self.loading_modes)):
            raise ValueError("Режимы загрузки не должны повторяться")

        return self


class ModelConfig(StrictConfig):
    architecture: Literal["MLP"]
    hidden_dims: list[int] = Field(min_length=1)
    dropout: float = Field(ge=0, lt=1)

    @model_validator(mode="after")
    def validate_hidden_dims(self):
        if any(size <= 0 for size in self.hidden_dims):
            raise ValueError("Размеры скрытых слоёв должны быть положительными")

        return self


class TrainingConfig(StrictConfig):
    batch_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    early_stopping_patience: int = Field(gt=0)
    early_stopping_min_delta: float = Field(ge=0)


class EvaluationConfig(StrictConfig):
    bootstrap_iterations: int = Field(ge=100)
    confidence_level: float = Field(gt=0, lt=1)
    precision_at_fraction: float = Field(gt=0, le=1)


class ReproducibilityConfig(StrictConfig):
    deterministic: bool
    same_seed_tolerance: float = Field(ge=0)
    device_parity_tolerance: float = Field(gt=0)


class RuntimeConfig(StrictConfig):
    device: Literal["auto", "cpu", "cuda"]
    num_workers: int = Field(ge=0)


class MLflowConfig(StrictConfig):
    experiment_name: str = Field(min_length=1)
    tracking_uri: str = Field(min_length=1)


class Lab3Config(StrictConfig):
    lab_number: Literal[3]
    variant: str = Field(pattern=r"^\d{2}$")
    dataset_path: str = Field(min_length=1)
    target_column: str = Field(min_length=1)

    random_state: int = Field(ge=0)
    test_size: float = Field(gt=0, lt=1)
    validation_size: float = Field(gt=0, lt=1)

    numeric_features: list[str] = Field(min_length=1)
    categorical_features: list[str] = Field(min_length=1)

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    reproducibility: ReproducibilityConfig
    runtime: RuntimeConfig
    mlflow: MLflowConfig

    output_dir: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_features(self):
        numeric = set(self.numeric_features)
        categorical = set(self.categorical_features)

        if numeric & categorical:
            raise ValueError(
                "Числовые и категориальные признаки не должны пересекаться"
            )

        if self.target_column in numeric | categorical:
            raise ValueError("Целевая переменная не должна входить в признаки")

        if len(numeric) != len(self.numeric_features):
            raise ValueError("Числовые признаки содержат дубликаты")

        if len(categorical) != len(self.categorical_features):
            raise ValueError("Категориальные признаки содержат дубликаты")

        if self.test_size + self.validation_size >= 1:
            raise ValueError("Сумма test_size и validation_size должна быть меньше 1")

        return self


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> Lab3Config:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    config = Lab3Config.model_validate(data)

    dataset_path = PROJECT_ROOT / config.dataset_path

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

    return config


if __name__ == "__main__":
    config = load_config()

    print("Конфигурация ЛР3 успешно проверена")
    print(f"Вариант: {config.variant}")
    print(f"Архитектура: {config.model.architecture}")
    print(f"Скрытые слои: {config.model.hidden_dims}")
    print(f"Режимы загрузки: {config.data.loading_modes}")
    print(f"Размер чанка: {config.data.chunk_size}")
    print(f"Устройство: {config.runtime.device}")
    print(f"Датасет: {config.dataset_path}")
