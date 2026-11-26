from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Literal, TypeVar, get_args, get_origin


MediaType = Literal["image", "chart", "table", "noise", "generic"]
MEDIA_TYPES = {"image", "chart", "table", "noise", "generic"}


@dataclass
class MMChunk:
    # 文本证据块：保留页码、位置、章节和附着媒体，查询时可回溯到原 PDF。
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
    # 图片/表格证据：保存路径、OCR/摘要、附近 chunk 和关联实体等检索辅助信息。
    media_id: str
    doc_id: str
    modality: MediaType
    page: int | None
    path: str
    original_type: str = ""
    mapped_type: MediaType | str = ""
    type: MediaType | str = ""
    caption: str = ""
    footnote: str = ""
    ocr_text: str = ""
    summary: str = ""
    table_html: str = ""
    table_markdown: str = ""
    bbox: list[float] | None = None
    nearby_chunk_ids: list[str] = field(default_factory=list)
    attached_entity_names: list[str] = field(default_factory=list)
    attach_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # mapped_type/type are the canonical Phase 1 fields. Keep modality in
        # sync so Phase 2-5 code that still reads modality remains compatible.
        candidate = str(self.mapped_type or self.type or self.modality or "generic").lower()
        canonical = candidate if candidate in MEDIA_TYPES else "generic"
        self.mapped_type = canonical
        self.type = canonical
        self.modality = canonical  # type: ignore[assignment]
        if not self.original_type:
            self.original_type = candidate

    @property
    def indexable(self) -> bool:
        return self.mapped_type != "noise"


def is_indexable_media(item: MMMedia | dict[str, Any]) -> bool:
    """Return False for decorative/layout noise while accepting legacy records."""
    if isinstance(item, MMMedia):
        return item.indexable
    mapped = str(item.get("mapped_type") or item.get("type") or item.get("modality") or "generic").lower()
    return mapped != "noise"


@dataclass
class MMNode:
    node_id: str
    doc_id: str
    node_type: Literal["document", "text", "entity", "media", "page", "aggregate"]
    page_id: int | None
    text_for_embedding: str
    raw_ref: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    bbox: list[float] | None = None
    caption: str = ""
    ocr_text: str = ""
    summary: str = ""
    source: str = ""


@dataclass
class MMEdge:
    edge_id: str
    src: str
    dst: str
    src_type: str
    dst_type: str
    edge_type: Literal[
        "document_contains_page",
        "page_contains_node",
        "text_mentions_entity",
        "entity_relation_entity",
        "text_caption_of_media",
        "text_refers_to_media",
        "entity_link_media",
        "media_near_text",
        "media_same_page_media",
        "node_semantic_similar_node",
        "node_aggregate_parent",
        "page_next_page",
        "page_prev_page",
    ]
    weight: float = 1.0
    direction: Literal["directed", "undirected"] = "directed"
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


T = TypeVar("T")


def dataclass_to_dict(item: Any) -> dict[str, Any]:
    # 统一 dataclass -> dict，便于写 JSON。
    if not is_dataclass(item):
        raise TypeError(f"Expected dataclass instance, got {type(item)!r}")
    return asdict(item)


def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    # 从 JSON 字典恢复 dataclass，忽略历史文件中多余字段以保持兼容。
    valid_fields = {item.name: item for item in fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in valid_fields:
            continue
        kwargs[key] = _coerce_value(valid_fields[key].type, value)
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    # 对常见类型做轻量纠正，避免旧 JSON 中单值/list/dict 形态不一致导致加载失败。
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
