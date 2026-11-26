from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None

from .io_utils import save_dataclasses, write_json
from .schema import MEDIA_TYPES, MMChunk, MMMedia, dataclass_to_dict


logger = logging.getLogger(__name__)

TEXT_BLOCK_TYPES = {
    "text", "title", "paragraph", "list", "equation", "formula", "interline_equation",
    "display_formula", "inline_formula", "equation_inline", "equation_interline", "aside_text", "page_aside_text",
    "algorithm", "code", "reference",
}
NOISE_TYPE_TERMS = {
    "page_number", "page_num", "page_header", "page_footer", "page_footnote", "header", "footer",
    "logo", "watermark", "decoration", "decorative", "ornament", "background", "seal",
}
CHART_TYPE_TERMS = {
    "chart", "plot", "bar_chart", "bar_graph", "line_chart", "line_graph", "pie_chart",
    "scatter_chart", "scatter_plot", "histogram", "box_plot", "heatmap", "radar_chart",
}
IMAGE_TYPE_TERMS = {
    "image", "img", "figure", "photo", "photograph", "diagram", "illustration", "schematic", "drawing",
}


# 将 MinerU 的解析产物统一转换为本项目内部使用的 chunk/media 数据结构。
def build_mm_chunks_from_mineru(
    mineru_output_dir: str,
    doc_id: str,
    source_pdf: str,
    max_token_size: int = 1024,
    overlap_token_size: int = 128,
) -> tuple[list[MMChunk], list[MMMedia]]:
    """Convert MinerU output into text chunks and normalized media records."""
    out = Path(mineru_output_dir)
    content_file = _find_content_list(out)
    if content_file:
        # 优先使用 MinerU 的 content_list，它保留了页码、bbox、图片和表格等结构化信息。
        content_items = _load_content_items(content_file)
        asset_root = content_file.parent
    else:
        # 某些解析结果只有 markdown；此时退化为纯文本文档。
        markdown = _find_markdown(out)
        if not markdown:
            raise FileNotFoundError(f"No MinerU content_list JSON or markdown found under {mineru_output_dir}")
        content_items = [{"type": "text", "text": markdown.read_text(encoding="utf-8", errors="ignore")}]
        asset_root = markdown.parent

    chunks: list[MMChunk] = []
    media_items: list[MMMedia] = []
    text_order = 0
    media_order = 0
    section_title = None
    for item in content_items:
        # MinerU 不同版本字段名略有差异，先统一抽取类型、页码和位置框。
        item_type = _item_type(item)
        page = _extract_page(item)
        bbox = _extract_bbox(item)
        if item_type == "text":
            text = _clean_text(_first(item, ["text", "content", "md", "markdown"], ""))
            if _looks_like_heading(text):
                section_title = text[:200]
            for part in _split_text(text, max_token_size, overlap_token_size):
                # hash_code 是 LeanRAG 后续图构建和证据回填的稳定键。
                chunk_id = f"{doc_id}_chunk_{text_order:06d}"
                hash_code = _hash(f"{doc_id}|{text_order}|{part}")
                chunks.append(
                    MMChunk(
                        chunk_id=chunk_id,
                        hash_code=hash_code,
                        doc_id=doc_id,
                        text=part,
                        page_start=page,
                        page_end=page,
                        section_title=section_title,
                        source_path=source_pdf,
                        bbox=bbox,
                        order=text_order,
                    )
                )
                text_order += 1
        elif item_type in MEDIA_TYPES:
            # 媒体本身不直接进入 LeanRAG 文本图，而是通过 summary/nearby_chunk_ids 与文本证据关联。
            path = _resolve_media_path(item, asset_root)
            original_type = _original_type(item)
            media_id = f"{doc_id}_{item_type}_{media_order:06d}"
            caption = _clean_text(_first(item, [f"{item_type}_caption", "caption", "img_caption", "image_caption", "chart_caption", "table_caption"], ""))
            footnote = _clean_text(_first(item, [f"{item_type}_footnote", "footnote", "img_footnote", "image_footnote", "chart_footnote", "table_footnote"], ""))
            table_html = _clean_text(_first(item, ["table_html", "html", "table_body"], ""))
            table_markdown = _clean_text(_first(item, ["table_markdown", "markdown", "md"], ""))
            ocr_text = _clean_text(_first(item, ["ocr_text", "ocr", "text"], ""))
            summary = _make_media_summary(item_type, caption, ocr_text, table_markdown or table_html, footnote)
            media_items.append(
                MMMedia(
                    media_id=media_id,
                    doc_id=doc_id,
                    modality=item_type,
                    page=page,
                    path=str(path) if path else "",
                    original_type=original_type,
                    mapped_type=item_type,
                    type=item_type,
                    caption=caption,
                    ocr_text=ocr_text,
                    summary=summary,
                    table_html=table_html if item_type == "table" else "",
                    table_markdown=table_markdown if item_type == "table" else "",
                    bbox=bbox,
                )
            )
            media_order += 1

    stats = media_type_statistics(media_items)
    logger.info("MinerU media type statistics: %s", " ".join(f"{name}={stats[name]}" for name in sorted(MEDIA_TYPES)))
    return chunks, media_items


def export_leanrag_text_chunks(mm_chunks: list[MMChunk], output_path: str) -> None:
    """Export text chunks in the format consumed by LeanRAG KG extractors."""
    write_json(
        [{"hash_code": chunk.hash_code, "text": chunk.text} for chunk in mm_chunks if chunk.modality == "text" and chunk.text],
        output_path,
    )


def save_mm_artifacts(
    chunks: list[MMChunk],
    media_items: list[MMMedia],
    working_dir: str,
) -> dict[str, str]:
    # 同时保存多模态完整结构和 LeanRAG 原生文本 chunk 结构。
    working = Path(working_dir)
    mm_chunk_file = working / "mm_chunk.json"
    mm_media_file = working / "mm_media.json"
    leanrag_chunk_file = working / "leanrag_chunk.json"
    save_dataclasses(chunks, mm_chunk_file)
    save_dataclasses(media_items, mm_media_file)
    media_stats_file = working / "media_type_stats.json"
    write_json(media_type_statistics(media_items), media_stats_file)
    export_leanrag_text_chunks(chunks, str(leanrag_chunk_file))
    try:
        from .mm_node_builder import build_phase1_mm_graph

        build_phase1_mm_graph(working_dir, validate=False)
    except Exception:
        # Node/edge generation can be rerun with multimodal.mm_node_builder;
        # keep the legacy chunk export path available if optional artifacts fail.
        pass
    return {
        "mm_chunk_file": str(mm_chunk_file),
        "mm_media_file": str(mm_media_file),
        "leanrag_chunk_file": str(leanrag_chunk_file),
        "media_type_stats_file": str(media_stats_file),
        "mm_nodes_file": str(working / "mm_nodes.jsonl"),
        "mm_edges_seed_file": str(working / "mm_edges_seed.jsonl"),
    }


def mm_chunks_to_leanrag_records(mm_chunks: list[MMChunk]) -> list[dict[str, str]]:
    return [{"hash_code": chunk.hash_code, "text": chunk.text} for chunk in mm_chunks if chunk.modality == "text"]


def mm_chunks_as_dicts(chunks: list[MMChunk]) -> list[dict[str, Any]]:
    return [dataclass_to_dict(chunk) for chunk in chunks]


def mm_media_as_dicts(media_items: list[MMMedia]) -> list[dict[str, Any]]:
    return [dataclass_to_dict(item) for item in media_items]


def _find_content_list(out: Path) -> Path | None:
    candidates = sorted(out.rglob("*content_list*.json"))
    if candidates:
        return candidates[0]
    candidates = sorted(out.rglob("*.json"))
    return candidates[0] if candidates else None


def _find_markdown(out: Path) -> Path | None:
    candidates = sorted(out.rglob("*.md"))
    return candidates[0] if candidates else None


def _load_content_items(path: Path) -> list[dict[str, Any]]:
    # 兼容 list、带 content/items/blocks 字段、以及按 pages 嵌套的多种 MinerU 输出。
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        if all(isinstance(item, list) for item in data):
            flattened = []
            for page_idx, page_items in enumerate(data):
                for item in page_items:
                    if isinstance(item, dict):
                        normalized = _normalize_content_item(item)
                        normalized.setdefault("page_idx", page_idx)
                        flattened.append(normalized)
            return flattened
        return [_normalize_content_item(item) for item in data if isinstance(item, dict)]
    for key in ("content_list", "content", "items", "blocks"):
        if isinstance(data.get(key), list):
            return [_normalize_content_item(item) for item in data[key] if isinstance(item, dict)]
    if isinstance(data.get("pages"), list):
        flattened = []
        for page in data["pages"]:
            if not isinstance(page, dict):
                continue
            page_no = _first(page, ["page", "page_idx", "page_no"], None)
            for item in page.get("items", page.get("content", page.get("blocks", []))):
                if isinstance(item, dict):
                    normalized = _normalize_content_item(item)
                    normalized.setdefault("page", page_no)
                    flattened.append(normalized)
        return flattened
    raise ValueError(f"Cannot parse MinerU content list: {path}")


def _normalize_content_item(item: dict[str, Any]) -> dict[str, Any]:
    """Flatten MinerU v2 nested content without changing v1 records."""
    normalized = dict(item)
    content = item.get("content")
    if not isinstance(content, dict):
        return normalized

    source = content.get("image_source") or content.get("media_source") or content.get("source")
    if isinstance(source, dict) and source.get("path") and not _has_media_payload(normalized):
        normalized["img_path"] = source["path"]
    for key in ("image_caption", "chart_caption", "table_caption", "caption"):
        if key in content and not normalized.get(key):
            normalized[key] = _nested_text(content[key])
    for key in ("image_footnote", "chart_footnote", "table_footnote", "footnote"):
        if key in content and not normalized.get(key):
            normalized[key] = _nested_text(content[key])
    if content.get("html") and not normalized.get("table_html"):
        normalized["table_html"] = content["html"]
    if content.get("markdown") and not normalized.get("table_markdown"):
        normalized["table_markdown"] = content["markdown"]

    raw_type = re.sub(r"[^a-z0-9]+", "_", _original_type(normalized).lower()).strip("_")
    if raw_type in TEXT_BLOCK_TYPES or raw_type in NOISE_TYPE_TERMS or raw_type.startswith("page_"):
        normalized.setdefault("text", _nested_text(content))
    return normalized


def _nested_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(part for item in value for part in [_nested_text(item)] if part)
    if isinstance(value, dict):
        direct = value.get("content")
        if isinstance(direct, str):
            return direct.strip()
        return "\n".join(part for item in value.values() for part in [_nested_text(item)] if part)
    return ""


def _item_type(item: dict[str, Any]) -> str:
    raw = _original_type(item)
    normalized = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if normalized in NOISE_TYPE_TERMS or any(term in normalized for term in ("page_number", "page_header", "page_footer", "logo", "watermark", "decorat")):
        return "noise"
    if normalized in CHART_TYPE_TERMS or any(term in normalized for term in ("chart", "scatter_plot", "line_plot", "bar_plot", "histogram", "heatmap")):
        return "chart"
    if "table" in normalized:
        return "table"
    if normalized in IMAGE_TYPE_TERMS or any(term in normalized for term in ("image", "photo", "diagram", "illustration", "schematic")):
        return "image"
    if normalized in TEXT_BLOCK_TYPES or _has_text_payload(item):
        return "text"
    # Unknown non-text blocks are retained as generic media. This is deliberate:
    # a MinerU version adding a new media label must not silently drop evidence.
    return "generic"


def _original_type(item: dict[str, Any]) -> str:
    return str(_first(item, ["type", "block_type", "category", "modality"], "unknown")).strip() or "unknown"


def _has_text_payload(item: dict[str, Any]) -> bool:
    return any(_clean_text(item.get(key)) for key in ("text", "markdown", "md")) and not _has_media_payload(item)


def _has_media_payload(item: dict[str, Any]) -> bool:
    return any(item.get(key) not in (None, "") for key in (
        "img_path", "image_path", "chart_path", "table_path", "asset_path", "path",
        "table_html", "table_markdown", "img_caption", "image_caption", "chart_caption", "table_caption",
    ))


def _extract_page(item: dict[str, Any]) -> int | None:
    # MinerU page_idx is 0-based; page/page_no are usually already 1-based.
    has_page_idx = item.get("page_idx") not in (None, "")
    value = item.get("page_idx") if has_page_idx else _first(item, ["page", "page_no"], None)
    if value in (None, ""):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page + 1 if has_page_idx else page


def _extract_bbox(item: dict[str, Any]) -> list[float] | None:
    value = _first(item, ["bbox", "box", "position"], None)
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [float(x) for x in value[:4]]
    except (TypeError, ValueError):
        return None


def _resolve_media_path(item: dict[str, Any], asset_root: Path) -> Path | None:
    value = _first(item, ["img_path", "image_path", "chart_path", "table_path", "asset_path", "path"], None)
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = asset_root / path
    return path


def _split_text(text: str, max_token_size: int, overlap_token_size: int) -> list[str]:
    # 有 tiktoken 时按 token 切分；否则按字符长度近似切分，保证离线也能运行。
    text = _clean_text(text)
    if not text:
        return []
    if tiktoken is None:
        if len(text) <= max_token_size:
            return [text]
        step = max(1, max_token_size - overlap_token_size)
        return [text[start : start + max_token_size].strip() for start in range(0, len(text), step)]
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    if len(tokens) <= max_token_size:
        return [text]
    step = max(1, max_token_size - overlap_token_size)
    return [
        encoder.decode(tokens[start : start + max_token_size]).strip()
        for start in range(0, len(tokens), step)
        if tokens[start : start + max_token_size]
    ]


def _make_media_summary(modality: str, caption: str, ocr_text: str, table_text: str, footnote: str) -> str:
    # 将 caption/OCR/table/footnote 压成一段可用于检索的媒体文本。
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    if table_text:
        parts.append(table_text)
    if footnote:
        parts.append(f"Footnote: {footnote}")
    compact = " ".join(" ".join(parts).split())
    if not compact:
        return f"{modality} evidence"
    return compact[:997] + "..." if len(compact) > 1000 else compact


def _looks_like_heading(text: str) -> bool:
    text = text.strip()
    return bool(text and len(text) < 120 and (text.startswith("#") or re.match(r"^\d+(\.\d+)*\s+\S+", text)))


def _first(item: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return default


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text if item is not None)
    return str(text).strip()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def media_type_statistics(media_items: list[MMMedia]) -> dict[str, int]:
    counts = Counter(str(item.mapped_type) for item in media_items)
    return {name: counts.get(name, 0) for name in ("image", "chart", "table", "noise", "generic")}
