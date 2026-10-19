from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

TARGET = "Survived"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Отделяем целевую переменную от признаков
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("=" * 60)
    print("1. ПРОВЕРКА ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
    print("=" * 60)
    print(f"Целевая переменная: {TARGET}")
    print(f"Размер X: {X.shape}")
    print(f"Размер y: {y.shape}")

    if TARGET in X.columns:
        raise ValueError("Утечка: целевая переменная находится в X")

    print("Проверка пройдена: Survived отсутствует в X")

    # Разделяем данные до обучения преобразований
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 60)
    print("2. РАЗДЕЛЕНИЕ НА ВЫБОРКИ")
    print("=" * 60)
    print(f"Train: {X_train.shape[0]} объектов")
    print(f"Test:  {X_test.shape[0]} объектов")

    print("\nРаспределение классов:")
    print("Train:")
    print(y_train.value_counts(normalize=True).sort_index().round(4))
    print("Test:")
    print(y_test.value_counts(normalize=True).sort_index().round(4))

    # Проверяем отсутствие пересечения индексов
    overlap = X_train.index.intersection(X_test.index)

    if len(overlap) > 0:
        raise ValueError("Обнаружено пересечение train и test")

    print("\nПроверка пройдена: train и test не пересекаются")

    # Фиксируем признаки, требующие отдельного решения
    print("\n" + "=" * 60)
    print("3. ПРИЗНАКИ ДЛЯ ДАЛЬНЕЙШЕГО АНАЛИЗА")
    print("=" * 60)

    for column in ["PassengerId", "Name", "Ticket", "Cabin"]:
        if column in X.columns:
            print(
                f"{column}: присутствует в X; необходимо оценить способ использования"
            )

    print("\nПроверка завершена.")


if __name__ == "__main__":
    main()
