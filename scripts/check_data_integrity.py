import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "hash_manifest.json"


def calculate_sha256(path: Path) -> str:
    """Вычисляет SHA-256 файла."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def collect_metadata(path: Path) -> dict:
    """Собирает метаданные исходного набора данных."""
    df = pd.read_csv(path)

    return {
        "file": path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": calculate_sha256(path),
        "size_bytes": path.stat().st_size,
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }


def create_manifest() -> None:
    """Создаёт манифест, не перезаписывая существующий."""
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"Манифест уже существует: {MANIFEST_PATH}")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Файл не найден: {DATA_PATH}")

    metadata = collect_metadata(DATA_PATH)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    MANIFEST_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"Манифест создан: {MANIFEST_PATH}")
    print(f"SHA-256: {metadata['sha256']}")
    print(f"Размер: {metadata['size_bytes']} байт")
    print(f"Строк: {metadata['rows']}")
    print(f"Столбцов: {metadata['columns_count']}")


def check_integrity() -> bool:
    """Сравнивает текущий файл с сохранённым манифестом."""
    if not MANIFEST_PATH.is_file():
        print("Ошибка: манифест отсутствует. Сначала создайте его.")
        return False

    if not DATA_PATH.is_file():
        print(f"Ошибка: файл данных не найден: {DATA_PATH}")
        return False

    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual = collect_metadata(DATA_PATH)

    fields = (
        "file",
        "sha256",
        "size_bytes",
        "rows",
        "columns_count",
        "columns",
    )

    success = True

    for field in fields:
        if actual[field] != expected[field]:
            success = False
            print(f"НЕ СОВПАДАЕТ: {field}")
            print(f"  Ожидалось: {expected[field]}")
            print(f"  Получено:  {actual[field]}")

    if success:
        print("Проверка целостности пройдена.")
        print(f"SHA-256: {actual['sha256']}")
        print(
            f"Размер таблицы: {actual['rows']} строк, "
            f"{actual['columns_count']} столбцов"
        )

    return success


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка целостности исходных данных")
    parser.add_argument(
        "--create-manifest",
        action="store_true",
        help="Создать манифест, если он ещё не существует",
    )
    args = parser.parse_args()

    try:
        if args.create_manifest:
            create_manifest()
            return 0

        return 0 if check_integrity() else 1

    except (OSError, ValueError, pd.errors.ParserError) as error:
        print(f"Ошибка: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
