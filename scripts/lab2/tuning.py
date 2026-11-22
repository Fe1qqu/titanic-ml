import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid

from scripts.lab2.config import Lab2Config
from scripts.lab2.pipeline import (
    build_preprocessor,
    build_classifier,
    build_param_grid,
    build_pipeline,
)
from scripts.lab2.data import iter_csv_split, load_split_indices


def tune_hyperparameters(
    config: Lab2Config,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
):
    """Подбирает гиперпараметры по F1 на проверочной выборке."""

    X_train = X_train.sort_index()
    y_train = y_train.loc[X_train.index]

    X_val = X_val.sort_index()
    y_val = y_val.loc[X_val.index]

    results = []

    for params in ParameterGrid(build_param_grid(config)):
        pipeline = build_pipeline(config)
        pipeline.set_params(**params)

        pipeline.set_params(
            classifier__max_iter=10,
            classifier__tol=None,
            classifier__shuffle=False,
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_val)
        score = f1_score(y_val, predictions, zero_division=0)

        results.append(
            {
                "params": params,
                "validation_f1": score,
            }
        )

    results.sort(
        key=lambda item: item["validation_f1"],
        reverse=True,
    )

    best_result = results[0]

    return best_result["params"], results


def tune_hyperparameters_streaming(config):
    """Подбирает гиперпараметры SGDClassifier без загрузки всех признаков."""

    split = load_split_indices(config)

    # Накопление статистик предобработки только на обучающей выборке
    preprocessor = build_preprocessor(config)

    for X_chunk, _ in iter_csv_split(
        config,
        split["train_indices"],
    ):
        preprocessor.partial_fit(X_chunk)

    preprocessor._finalize()

    param_grid = ParameterGrid(build_param_grid(config))

    best_params = None
    best_score = -1.0

    tuning_epochs = 10
    classes = np.unique(split["y_train"])

    for params in param_grid:
        classifier = build_classifier(
            config,
            random_state=config.random_state,
        )

        classifier_params = {
            name.removeprefix("classifier__"): value for name, value in params.items()
        }

        classifier.set_params(
            **classifier_params,
            max_iter=tuning_epochs,
            tol=None,
            shuffle=False,
        )

        # Обучение классификатора по чанкам
        first_batch = True

        for epoch in range(tuning_epochs):
            for X_chunk, y_chunk in iter_csv_split(
                config,
                split["train_indices"],
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

        # Оценка на validation-выборке
        y_true_parts = []
        y_pred_parts = []

        for X_chunk, y_chunk in iter_csv_split(
            config,
            split["validation_indices"],
        ):
            X_transformed = preprocessor.transform(X_chunk)

            predictions = classifier.predict(X_transformed)

            y_true_parts.append(y_chunk.to_numpy())
            y_pred_parts.append(predictions)

        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)

        score = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        print(f"Параметры: {params} | Validation F1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_params = params.copy()

    print(f"\nЛучшие потоковые параметры: {best_params}")
    print(f"Лучший Validation F1: {best_score:.4f}")

    return best_params


if __name__ == "__main__":
    from scripts.lab2.config import load_config
    from scripts.lab2.data import load_and_split_data

    config = load_config()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = load_and_split_data(config)

    best_params, results = tune_hyperparameters(
        config,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print("Результаты подбора гиперпараметров:\n")

    for result in results:
        print(
            f"Параметры: {result['params']}, "
            f"Validation F1: {result['validation_f1']:.4f}"
        )

    print("\nЛучшие параметры:")
    print(best_params)
