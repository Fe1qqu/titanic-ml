import copy
import json

import numpy as np
import pandas as pd
from pydantic import ValidationError
from sklearn.model_selection import train_test_split

from scripts.lab2.config import (
    DEFAULT_CONFIG_PATH,
    Lab2Config,
    PROJECT_ROOT,
)
from scripts.lab2.pipeline import build_preprocessor


def check_invalid_configs(config_data: dict, log_lines: list[str]) -> None:
    """Проверяет отклонение трёх некорректных конфигураций."""

    test_cases = [
        (
            "Пересечение числовых и категориальных признаков",
            lambda data: data["categorical_features"].__setitem__(
                0, data["numeric_features"][0]
            ),
        ),
        (
            "Целевая переменная включена в признаки",
            lambda data: data["numeric_features"].append(data["target_column"]),
        ),
        (
            "Нулевой размер чанка",
            lambda data: data.__setitem__("chunk_size", 0),
        ),
    ]

    print("Проверка невалидных конфигураций...")

    for name, modify in test_cases:
        invalid_data = copy.deepcopy(config_data)
        modify(invalid_data)

        try:
            Lab2Config.model_validate(invalid_data)
        except ValidationError as error:
            log_lines.append(f"[PASS] {name}")
            log_lines.append(str(error))
            log_lines.append("")
            print(f"[PASS] {name}")
        else:
            log_lines.append(f"[FAIL] {name}")
            print(f"[FAIL] {name}")
            raise AssertionError(f"Конфигурация не была отклонена: {name}")


def check_preprocessing_leakage(
    config: Lab2Config,
    log_lines: list[str],
) -> None:
    """Проверяет, что статистики предобработки не зависят от тестовых данных."""

    print("\nПроверка утечки данных...")

    dataset_path = PROJECT_ROOT / config.dataset_path
    data = pd.read_csv(dataset_path)

    features = config.numeric_features + config.categorical_features

    X = data[features]
    y = data[config.target_column]

    X_train_validation, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    X_train, _, _, _ = train_test_split(
        X_train_validation,
        y.loc[X_train_validation.index],
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y.loc[X_train_validation.index],
    )

    # Обучаем предобработчик исключительно на обучающей выборке.
    preprocessor = build_preprocessor(config)
    preprocessor.fit(X_train)

    original_mean = preprocessor.scaler_.mean_.copy()
    original_categories = copy.deepcopy(preprocessor.categories_)

    # Искусственно изменяем тестовые данные.
    modified_test = X_test.copy()

    numeric_feature = config.numeric_features[0]
    categorical_feature = config.categorical_features[0]

    modified_test[numeric_feature] = 1_000_000_000
    modified_test[categorical_feature] = modified_test[categorical_feature].astype(
        object
    )

    modified_test[categorical_feature] = "__leakage_test__"

    # Преобразование тестовых данных не должно менять статистики.
    preprocessor.transform(modified_test)

    np.testing.assert_array_equal(
        preprocessor.scaler_.mean_,
        original_mean,
    )

    assert preprocessor.categories_ == original_categories

    log_lines.append(
        "[PASS] Статистики предобработки не изменились после "
        "преобразования изменённой тестовой выборки."
    )
    print("[PASS] Тестовые данные не изменили статистики предобработки.")

    # Демонстрируем, что намеренное использование теста приводит к утечке.
    leaky_preprocessor = build_preprocessor(config)
    leaky_preprocessor.fit(pd.concat([X_train, modified_test], axis=0))

    mean_changed = not np.allclose(
        leaky_preprocessor.scaler_.mean_,
        original_mean,
    )

    category_leaked = (
        "__leakage_test__" in leaky_preprocessor.categories_[categorical_feature]
    )

    assert mean_changed, (
        "Тест не обнаружил изменение числовых статистик "
        "при использовании тестовой выборки."
    )

    assert category_leaked, (
        "Тест не обнаружил попадание тестовой категории в обученные категории."
    )

    log_lines.append(
        "[PASS] Контрольная утечка обнаружена: "
        "изменились числовые статистики и появился "
        "искусственный тестовый признак."
    )
    print("[PASS] Контрольная утечка успешно обнаружена.")

    log_lines.append(f"Проверенный числовой признак: {numeric_feature}")
    log_lines.append(f"Проверенный категориальный признак: {categorical_feature}")


def main() -> None:
    with DEFAULT_CONFIG_PATH.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        config_data = json.load(file)

    config = Lab2Config.model_validate(config_data)

    log_lines = [
        "Проверки корректности конфигурации и отсутствия утечки данных",
        f"Вариант: {config.variant}",
        "",
    ]

    check_invalid_configs(config_data, log_lines)
    check_preprocessing_leakage(config, log_lines)

    output_dir = PROJECT_ROOT / config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "config_negative_test.log"
    log_path.write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )

    print("\nВсе проверки пройдены.")
    print(f"Результаты сохранены: {log_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
