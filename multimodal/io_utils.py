from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Iterable, TypeVar

from .schema import dataclass_from_dict, dataclass_to_dict

T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def load_dataclasses(path: str | Path, cls: type[T]) -> list[T]:
    return [dataclass_from_dict(cls, item) for item in read_json(path)]


def save_dataclasses(items: Iterable, path: str | Path) -> None:
    write_json([dataclass_to_dict(item) for item in items], path)


def _jsonable(value):
    if is_dataclass(value):
        return dataclass_to_dict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
