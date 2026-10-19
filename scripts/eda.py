import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports" / "LAB1"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET_COLUMN = "Survived"
RANDOM_STATE = 42
TEST_SIZE = 0.2

sns.set_theme(style="whitegrid")


def calculate_sha256(file_path: Path) -> str:
    """Вычислить SHA-256 файла."""
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def add_stat(
    rows: list[dict],
    metric: str,
    feature: str,
    value: object,
    description: str,
) -> None:
    """Добавить запись в таблицу сводной статистики."""
    rows.append(
        {
            "metric": metric,
            "feature": feature,
            "value": value,
            "description": description,
        }
    )


def save_eda_statistics(df: pd.DataFrame) -> None:
    """Сохранить сводную статистику в CSV."""
    rows = []
    row_count, column_count = df.shape

    add_stat(rows, "row_count", "dataset", row_count, "Количество строк")
    add_stat(
        rows,
        "column_count",
        "dataset",
        column_count,
        "Количество столбцов",
    )
    add_stat(
        rows,
        "duplicate_rows",
        "dataset",
        int(df.duplicated().sum()),
        "Количество полных дубликатов строк",
    )
    add_stat(
        rows,
        "duplicate_passenger_ids",
        "PassengerId",
        int(df["PassengerId"].duplicated().sum()),
        "Количество повторных PassengerId",
    )

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percent = missing_count / row_count * 100

        add_stat(
            rows,
            "dtype",
            column,
            str(df[column].dtype),
            "Тип данных столбца",
        )
        add_stat(
            rows,
            "missing_count",
            column,
            missing_count,
            "Количество пропусков",
        )
        add_stat(
            rows,
            "missing_percent",
            column,
            round(missing_percent, 4),
            "Доля пропусков в процентах",
        )
        add_stat(
            rows,
            "unique_count",
            column,
            int(df[column].nunique()),
            "Количество уникальных непустых значений",
        )

    for class_value, count in df[TARGET_COLUMN].value_counts().items():
        add_stat(
            rows,
            "class_count",
            str(class_value),
            int(count),
            "Количество объектов целевого класса",
        )
        add_stat(
            rows,
            "class_percent",
            str(class_value),
            round(count / row_count * 100, 4),
            "Доля объектов целевого класса в процентах",
        )

    numeric_df = df.select_dtypes(include="number")

    for column in numeric_df.columns:
        series = numeric_df[column]
        stats = series.describe()

        for stat_name, value in stats.items():
            add_stat(
                rows,
                f"numeric_{stat_name}",
                column,
                round(float(value), 6),
                f"Числовая статистика: {stat_name}",
            )

        # Выбросы по правилу 1,5 межквартильного размаха.
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = int(((series < lower_bound) | (series > upper_bound)).sum())

        add_stat(
            rows,
            "iqr_outlier_count",
            column,
            outlier_count,
            "Количество потенциальных выбросов по правилу 1,5 IQR",
        )

    # Корреляции числовых признаков с целевым признаком.
    correlations = numeric_df.corr()[TARGET_COLUMN]

    for column, value in correlations.items():
        add_stat(
            rows,
            "correlation_with_survived",
            column,
            round(float(value), 6),
            "Корреляция Пирсона с Survived",
        )

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(
        REPORTS_DIR / "eda_stats.csv",
        index=False,
        encoding="utf-8",
    )

    # Сохраняем прежний отдельный файл пропусков.
    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_percent": (df.isna().mean().values * 100).round(4),
        }
    )
    missing_df.to_csv(
        REPORTS_DIR / "missing_values.csv",
        index=False,
        encoding="utf-8",
    )


def save_missingness_plot(df: pd.DataFrame) -> None:
    """Построить обязательный график пропусков."""
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))

    if missing.empty:
        ax.text(
            0.5,
            0.5,
            "Пропуски не обнаружены",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    else:
        bars = ax.barh(missing.index, missing.values)
        ax.bar_label(
            bars,
            labels=[
                f"{count} ({count / len(df) * 100:.2f}%)" for count in missing.values
            ],
            padding=4,
        )
        ax.set_xlabel("Количество пропусков")
        ax.set_ylabel("Признак")
        ax.set_title("Пропуски в наборе Titanic")
        ax.set_xlim(0, max(missing.values) * 1.25)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "missing_values.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_class_distribution(df: pd.DataFrame) -> None:
    """Построить распределение целевого класса."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    labels = {0: "Погиб", 1: "Выжил"}

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        [labels.get(value, str(value)) for value in counts.index],
        counts.values,
    )

    ax.bar_label(
        bars,
        labels=[f"{count} ({count / len(df) * 100:.2f}%)" for count in counts.values],
        padding=4,
    )
    ax.set_title("Распределение целевого класса Survived")
    ax.set_ylabel("Количество пассажиров")
    ax.set_ylim(0, max(counts.values) * 1.2)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def save_numeric_distributions(df: pd.DataFrame) -> None:
    """Построить распределения возраста и стоимости билета."""
    for column, filename, title in [
        ("Age", "age_distribution.png", "Распределение возраста"),
        ("Fare", "fare_distribution.png", "Распределение стоимости билета"),
    ]:
        values = df[column].dropna()

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(values, bins=30, kde=True, ax=ax)

        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Количество пассажиров")

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / filename, dpi=150)
        plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    """Построить тепловую карту корреляций числовых признаков."""
    numeric_df = df.select_dtypes(include="number")
    correlation_matrix = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )

    ax.set_title("Корреляции числовых признаков")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)


def save_missingness_by_survival(df: pd.DataFrame) -> None:
    """Сравнить доли пропусков по целевым классам."""
    columns = [
        column for column in ["Age", "Cabin", "Embarked"] if column in df.columns
    ]

    proportions = df.groupby(TARGET_COLUMN)[columns].apply(
        lambda group: group.isna().mean() * 100
    )
    proportions.index = proportions.index.map({0: "Погиб", 1: "Выжил"})

    fig, ax = plt.subplots(figsize=(9, 5))
    proportions.T.plot(kind="bar", ax=ax)

    ax.set_title("Доля пропусков по классам Survived")
    ax.set_xlabel("Признак")
    ax.set_ylabel("Пропуски, %")
    ax.legend(title="Исход")
    ax.tick_params(axis="x", rotation=0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", padding=2)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "missingness_by_survival.png",
        dpi=150,
    )
    plt.close(fig)


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Исходный набор данных не найден: {DATA_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"В наборе отсутствует целевой столбец {TARGET_COLUMN}")

    print("=" * 60)
    print("РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ TITANIC")
    print("=" * 60)
    print(f"Размер набора: {df.shape}")
    print(f"SHA-256: {calculate_sha256(DATA_PATH)}")
    print(f"Полных дубликатов: {df.duplicated().sum()}")
    print("\nТипы данных:")
    print(df.dtypes.to_string())

    print("\nПропуски:")
    print(
        df.isna()
        .sum()
        .to_frame("count")
        .assign(percent=(df.isna().mean() * 100).round(2))
        .to_string()
    )

    print("\nРаспределение целевого класса:")
    print(df[TARGET_COLUMN].value_counts().sort_index().to_string())

    print("\nЧисловые описательные статистики:")
    print(df.describe().T.to_string())

    save_eda_statistics(df)
    save_missingness_plot(df)
    save_class_distribution(df)
    save_numeric_distributions(df)
    save_correlation_heatmap(df)
    save_missingness_by_survival(df)

    print("\nСозданные артефакты:")
    print(f"Сводка: {REPORTS_DIR / 'eda_stats.csv'}")
    print(f"Пропуски: {REPORTS_DIR / 'missing_values.csv'}")
    print(f"Графики: {FIGURES_DIR}")
    print("\nEDA завершён.")


if __name__ == "__main__":
    main()
