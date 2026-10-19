from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[["Age", "Fare"]]
    y = df["Survived"]

    X_train, X_test, _, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("=" * 60)
    print("СРАВНЕНИЕ ПАРАМЕТРОВ ЗАПОЛНЕНИЯ")
    print("=" * 60)

    for column in X.columns:
        full_median = X[column].median()
        train_median = X_train[column].median()

        print(f"\nПризнак: {column}")
        print(f"Медиана по всему набору: {full_median:.4f}")
        print(f"Медиана только по train: {train_median:.4f}")
        print(f"Разница: {full_median - train_median:.4f}")

    print("\n" + "=" * 60)
    print("ПРОВЕРКА ПРАВИЛЬНОГО ПОРЯДКА")
    print("=" * 60)
    print("1. Разделить данные на train и test.")
    print("2. Вычислить медианы по train.")
    print("3. Заполнить пропуски в train и test")
    print("   медианами, вычисленными по train.")
    print("4. Не вычислять параметры по test.")


if __name__ == "__main__":
    main()
