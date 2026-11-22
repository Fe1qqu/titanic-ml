import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.lab2.config import Lab2Config


class StreamingPreprocessor(BaseEstimator, TransformerMixin):
    """Предобработка с возможностью последовательного накопления статистик."""

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def _reset(self):
        self.scaler_ = StandardScaler()
        self.category_counts_ = {feature: {} for feature in self.categorical_features}
        self.is_fitted_ = False

    def partial_fit(self, X, y=None):
        """Накопление статистик по очередной порции данных."""

        if not hasattr(self, "scaler_"):
            self._reset()

        numeric = X[self.numeric_features].astype(float)
        self.scaler_.partial_fit(numeric)

        for feature in self.categorical_features:
            values = X[feature].dropna()

            counts = values.value_counts()

            for value, count in counts.items():
                current = self.category_counts_[feature].get(value, 0)
                self.category_counts_[feature][value] = current + int(count)

        return self

    def fit(self, X, y=None):
        """Обучение предобработчика на полной выборке."""

        self._reset()
        self.partial_fit(X)
        self._finalize()
        return self

    def _finalize(self):
        """Фиксация категорий и значений для заполнения пропусков."""

        self.categories_ = {}
        self.fill_values_ = {}

        for feature in self.categorical_features:
            counts = self.category_counts_[feature]

            if not counts:
                categories = ["__missing__"]
                fill_value = "__missing__"
            else:
                categories = sorted(counts, key=str)
                fill_value = sorted(
                    counts,
                    key=lambda value: (-counts[value], str(value)),
                )[0]

            self.categories_[feature] = categories
            self.fill_values_[feature] = fill_value

        self.numeric_fill_values_ = np.where(
            np.isfinite(self.scaler_.mean_),
            self.scaler_.mean_,
            0.0,
        )

        self.is_fitted_ = True
        return self

    def transform(self, X):
        """Преобразование числовых и категориальных признаков."""

        if not getattr(self, "is_fitted_", False):
            self._finalize()

        numeric = X[self.numeric_features].astype(float)
        numeric = numeric.fillna(
            dict(
                zip(
                    self.numeric_features,
                    self.numeric_fill_values_,
                )
            )
        )

        numeric_transformed = self.scaler_.transform(numeric)

        categorical_parts = []

        for feature in self.categorical_features:
            values = X[feature].fillna(self.fill_values_[feature])

            categories = self.categories_[feature]

            encoded = np.column_stack(
                [(values == category).to_numpy(dtype=float) for category in categories]
            )

            categorical_parts.append(encoded)

        if categorical_parts:
            categorical_transformed = np.hstack(categorical_parts)

            return np.hstack(
                [
                    numeric_transformed,
                    categorical_transformed,
                ]
            )

        return numeric_transformed

    def get_feature_names_out(self, input_features=None):
        """Возвращает названия преобразованных признаков."""

        names = list(self.numeric_features)

        for feature in self.categorical_features:
            names.extend(
                [f"{feature}_{category}" for category in self.categories_[feature]]
            )

        return np.asarray(names, dtype=object)


def build_preprocessor(config: Lab2Config) -> StreamingPreprocessor:
    """Создаёт потоковый предобработчик."""

    return StreamingPreprocessor(
        numeric_features=config.numeric_features,
        categorical_features=config.categorical_features,
    )


def build_classifier(
    config: Lab2Config,
    random_state: int | None = None,
) -> SGDClassifier:
    """Создаёт классификатор с параметрами из конфигурации."""

    return SGDClassifier(
        loss=config.model.loss,
        alpha=config.model.alpha_candidates[0],
        penalty=config.model.penalty_candidates[0],
        max_iter=config.model.max_iter,
        tol=config.model.tol,
        shuffle=False,
        random_state=(config.random_state if random_state is None else random_state),
    )


def build_pipeline(config: Lab2Config) -> Pipeline:
    """Создаёт полный конвейер предобработки и классификации."""

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            ("classifier", build_classifier(config)),
        ]
    )


def build_param_grid(config: Lab2Config) -> dict:
    """Возвращает пространство гиперпараметров для подбора."""

    return {
        "classifier__alpha": config.model.alpha_candidates,
        "classifier__penalty": config.model.penalty_candidates,
    }


if __name__ == "__main__":
    from scripts.lab2.config import load_config

    config = load_config()
    pipeline = build_pipeline(config)

    print("ML-пайплайн успешно создан")
    print(pipeline)
    print("\nСетка гиперпараметров:")
    print(build_param_grid(config))
