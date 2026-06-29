from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, TypeVar, get_args, get_origin


@dataclass
class MMChunk:
    chunk_id: str
    hash_code: str
    doc_id: str
    text: str
    modality: str = "text"
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    source_path: str | None = None
    bbox: list[float] | None = None
    order: int = 0
    attached_media_ids: list[str] = field(default_factory=list)


@dataclass
class MMMedia:
    media_id: str
    doc_id: str
    modality: Literal["image", "table"]
    page: int | None
    path: str
    caption: str = ""
    ocr_text: str = ""
    summary: str = ""
    table_html: str = ""
    table_markdown: str = ""
    bbox: list[float] | None = None
    nearby_chunk_ids: list[str] = field(default_factory=list)
    attached_entity_names: list[str] = field(default_factory=list)
    attach_scores: dict[str, float] = field(default_factory=dict)


T = TypeVar("T")


def dataclass_to_dict(item: Any) -> dict[str, Any]:
    if not is_dataclass(item):
        raise TypeError(f"Expected dataclass instance, got {type(item)!r}")
    return asdict(item)


def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    valid_fields = {item.name: item for item in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in valid_fields:
            continue
        kwargs[key] = _coerce_value(valid_fields[key].type, value)
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin is list and not isinstance(value, list):
        return [value]
    if origin is dict and not isinstance(value, dict):
        return {}
    if origin is Literal:
        allowed = get_args(annotation)
        return value if value in allowed else allowed[0]
    return value
