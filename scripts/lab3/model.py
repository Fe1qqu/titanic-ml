import torch
from torch import nn

from scripts.lab3.config import ModelConfig


class TitanicMLP(nn.Module):
    """Многослойный перцептрон для бинарной классификации Titanic."""

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("Количество входных признаков должно быть положительным")

        layers = []
        previous_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Возвращает логиты для объектов входного батча."""

        return self.network(features).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Возвращает количество обучаемых параметров модели."""

    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def predict_proba(
    model: nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    """Вычисляет вероятности положительного класса."""

    model.eval()

    with torch.no_grad():
        logits = model(features)
        probabilities = torch.sigmoid(logits)

    return probabilities
