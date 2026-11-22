import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "lab2.json"


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(StrictConfig):
    name: Literal["SGDClassifier"]
    loss: Literal["log_loss"]
    alpha_candidates: list[float] = Field(min_length=1)
    penalty_candidates: list[Literal["l1", "l2", "elasticnet"]] = Field(min_length=1)
    max_iter: int = Field(gt=0)
    tol: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_alpha_candidates(self):
        if any(alpha <= 0 for alpha in self.alpha_candidates):
            raise ValueError("Все значения alpha должны быть положительными")
        return self


class EvaluationConfig(StrictConfig):
    bootstrap_iterations: int = Field(ge=100)
    confidence_level: float = Field(gt=0, lt=1)
    precision_at_fraction: float = Field(gt=0, le=1)


class MLflowConfig(StrictConfig):
    experiment_name: str = Field(min_length=1)
    tracking_uri: str = Field(min_length=1)


class Lab2Config(StrictConfig):
    lab_number: Literal[2]
    variant: str = Field(pattern=r"^\d{2}$")
    dataset_path: str = Field(min_length=1)
    target_column: str = Field(min_length=1)

    random_state: int = Field(ge=0)
    test_size: float = Field(gt=0, lt=1)
    validation_size: float = Field(gt=0, lt=1)
    chunk_size: int = Field(gt=0)

    numeric_features: list[str] = Field(min_length=1)
    categorical_features: list[str] = Field(min_length=1)

    model: ModelConfig
    evaluation: EvaluationConfig
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
) -> Lab2Config:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    config = Lab2Config.model_validate(data)

    dataset_path = PROJECT_ROOT / config.dataset_path

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

    return config


if __name__ == "__main__":
    config = load_config()

    print("Конфигурация успешно проверена")
    print(f"Вариант: {config.variant}")
    print(f"Размер чанка: {config.chunk_size}")
    print(f"Модель: {config.model.name}")
    print(f"Датасет: {config.dataset_path}")
