import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from scripts.lab3.config import Lab3Config
from scripts.lab3.data import (
    SplitIndices,
    load_split_indices,
    load_full_data,
    prepare_streaming_preprocessor,
    make_full_loader,
    make_streaming_loader,
)
from scripts.lab3.model import TitanicMLP, count_parameters


@dataclass
class TrainingResult:
    model: TitanicMLP
    history: list[dict[str, float]]
    best_epoch: int
    epochs_trained: int
    best_validation_loss: float
    elapsed_seconds: float
    device: str
    parameter_count: int


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Устанавливает seed для Python, NumPy и PyTorch."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False


def resolve_device(config: Lab3Config) -> torch.device:
    """Определяет устройство обучения согласно конфигурации."""

    requested_device = config.runtime.device

    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("В конфигурации указано CUDA, но CUDA недоступна")

    return torch.device(requested_device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> float:
    """Выполняет одну эпоху обучения или валидации."""

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            logits = model(features)
            loss = criterion(logits, targets)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = len(targets)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader не вернул ни одного объекта")

    return total_loss / total_samples


def train_model(
    model: TitanicMLP,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: Lab3Config,
    device: torch.device,
) -> TrainingResult:
    """Обучает модель с ранней остановкой по validation loss."""

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    history: list[dict[str, float]] = []

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        validation_loss = _run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
            }
        )

        improved = (
            validation_loss
            < best_validation_loss - config.training.early_stopping_min_delta
        )

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch

            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{config.training.epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
        )

        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(
                "Early stopping: "
                f"валидационная ошибка не улучшалась "
                f"{config.training.early_stopping_patience} эпох."
            )
            break

    elapsed_seconds = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("Не удалось получить состояние обученной модели")

    model.load_state_dict(best_state)
    model.eval()

    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        epochs_trained=len(history),
        best_validation_loss=best_validation_loss,
        elapsed_seconds=elapsed_seconds,
        device=str(device),
        parameter_count=count_parameters(model),
    )


def train_full(
    config: Lab3Config,
    splits: SplitIndices | None = None,
) -> TrainingResult:
    """Обучает MLP с предварительной загрузкой данных в память."""

    seed_everything(
        config.random_state,
        config.reproducibility.deterministic,
    )

    device = resolve_device(config)

    if splits is None:
        splits = load_split_indices(config)

    data = load_full_data(config, splits)

    train_loader = make_full_loader(
        X=data.X_train,
        y=splits.y_train,
        batch_size=config.training.batch_size,
        shuffle=True,
        seed=config.random_state,
        num_workers=config.runtime.num_workers,
    )

    validation_loader = make_full_loader(
        X=data.X_validation,
        y=splits.y_validation,
        batch_size=config.training.batch_size,
        shuffle=False,
        seed=config.random_state,
        num_workers=config.runtime.num_workers,
    )

    model = TitanicMLP(
        input_dim=data.X_train.shape[1],
        config=config.model,
    )

    print("Режим: full")
    print(f"Устройство: {device}")
    print(f"Размер обучающей выборки: {len(splits.train)}")

    return train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        device=device,
    )


def train_streaming(
    config: Lab3Config,
    splits: SplitIndices | None = None,
) -> TrainingResult:
    """Обучает MLP с потоковым чтением CSV."""

    seed_everything(
        config.random_state,
        config.reproducibility.deterministic,
    )

    device = resolve_device(config)

    if splits is None:
        splits = load_split_indices(config)

    preprocessor = prepare_streaming_preprocessor(
        config,
        splits,
    )

    train_loader = make_streaming_loader(
        config=config,
        indices=splits.train,
        preprocessor=preprocessor,
        shuffle=True,
    )

    validation_loader = make_streaming_loader(
        config=config,
        indices=splits.validation,
        preprocessor=preprocessor,
        shuffle=False,
    )

    model = TitanicMLP(
        input_dim=len(preprocessor.feature_names),
        config=config.model,
    )

    print("Режим: streaming")
    print(f"Устройство: {device}")
    print(f"Размер обучающей выборки: {len(splits.train)}")
    print(f"Размер чанка CSV: {config.data.chunk_size}")

    return train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        device=device,
    )
