import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "lab1.json"


class LabConfig(BaseModel):
    """Конфигурация лабораторной работы."""

    model_config = ConfigDict(extra="forbid")

    lab_number: int = Field(ge=1, le=8)
    variant: str = Field(pattern=r"^\d{2}$")
    dataset_path: str = Field(min_length=1)
    target_column: str = Field(min_length=1)
    random_state: int = Field(ge=0)
    test_size: float = Field(gt=0, lt=1)
    expected_rows: int = Field(gt=0)
    expected_columns: int = Field(gt=0)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> LabConfig:
    """Загрузить JSON-конфигурацию и проверить её по схеме."""
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    return LabConfig.model_validate(data)


if __name__ == "__main__":
    config = load_config()
    print("Конфигурация успешно проверена.")
    print(config.model_dump_json(indent=2))
