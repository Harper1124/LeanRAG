from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from typing import Iterable, TypeVar

from .schema import dataclass_from_dict, dataclass_to_dict

T = TypeVar("T")


# 创建目录并返回 Path，供构建脚本统一处理输出路径。
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path):
    # 使用 utf-8-sig 兼容带 BOM 的 JSON 文件。
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(data, path: str | Path) -> None:
    # 写 JSON 前先递归转换 dataclass/list/dict，并自动创建父目录。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict]:
    # 跳过空行，返回 JSONL 中的对象列表。
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    # 逐行写 JSON，适合预测结果和 entity/relation 文件。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def load_dataclasses(path: str | Path, cls: type[T]) -> list[T]:
    # 从 JSON 数组恢复 dataclass 列表。
    return [dataclass_from_dict(cls, item) for item in read_json(path)]


def save_dataclasses(items: Iterable, path: str | Path) -> None:
    write_json([dataclass_to_dict(item) for item in items], path)


def _jsonable(value):
    # 递归处理 dataclass 和容器类型，保证 json.dump 可以直接序列化。
    if is_dataclass(value):
        return dataclass_to_dict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
