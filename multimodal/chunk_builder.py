from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None

from .io_utils import save_dataclasses, write_json
from .schema import MMChunk, MMMedia, dataclass_to_dict


def build_mm_chunks_from_mineru(
    mineru_output_dir: str,
    doc_id: str,
    source_pdf: str,
    max_token_size: int = 1024,
    overlap_token_size: int = 128,
) -> tuple[list[MMChunk], list[MMMedia]]:
    """Convert MinerU output into text chunks and image/table media records."""
    out = Path(mineru_output_dir)
    content_file = _find_content_list(out)
    if content_file:
        content_items = _load_content_items(content_file)
        asset_root = content_file.parent
    else:
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
        item_type = _item_type(item)
        page = _extract_page(item)
        bbox = _extract_bbox(item)
        if item_type == "text":
            text = _clean_text(_first(item, ["text", "content", "md", "markdown"], ""))
            if _looks_like_heading(text):
                section_title = text[:200]
            for part in _split_text(text, max_token_size, overlap_token_size):
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
        elif item_type in {"image", "table"}:
            path = _resolve_media_path(item, asset_root)
            media_id = f"{doc_id}_{item_type}_{media_order:06d}"
            caption = _clean_text(_first(item, [f"{item_type}_caption", "caption", "img_caption", "table_caption"], ""))
            footnote = _clean_text(_first(item, [f"{item_type}_footnote", "footnote", "img_footnote", "table_footnote"], ""))
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
                    caption=caption,
                    ocr_text=ocr_text,
                    summary=summary,
                    table_html=table_html if item_type == "table" else "",
                    table_markdown=table_markdown if item_type == "table" else "",
                    bbox=bbox,
                )
            )
            media_order += 1

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
    working = Path(working_dir)
    mm_chunk_file = working / "mm_chunk.json"
    mm_media_file = working / "mm_media.json"
    leanrag_chunk_file = working / "leanrag_chunk.json"
    save_dataclasses(chunks, mm_chunk_file)
    save_dataclasses(media_items, mm_media_file)
    export_leanrag_text_chunks(chunks, str(leanrag_chunk_file))
    return {
        "mm_chunk_file": str(mm_chunk_file),
        "mm_media_file": str(mm_media_file),
        "leanrag_chunk_file": str(leanrag_chunk_file),
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
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    for key in ("content_list", "content", "items", "blocks"):
        if isinstance(data.get(key), list):
            return [item for item in data[key] if isinstance(item, dict)]
    if isinstance(data.get("pages"), list):
        flattened = []
        for page in data["pages"]:
            if not isinstance(page, dict):
                continue
            page_no = _first(page, ["page", "page_idx", "page_no"], None)
            for item in page.get("items", page.get("content", page.get("blocks", []))):
                if isinstance(item, dict):
                    item.setdefault("page", page_no)
                    flattened.append(item)
        return flattened
    raise ValueError(f"Cannot parse MinerU content list: {path}")


def _item_type(item: dict[str, Any]) -> str:
    raw = str(_first(item, ["type", "category", "modality"], "text")).lower()
    if "table" in raw:
        return "table"
    if "image" in raw or "img" in raw or "figure" in raw:
        return "image"
    return "text"


def _extract_page(item: dict[str, Any]) -> int | None:
    value = _first(item, ["page_idx", "page", "page_no"], None)
    if value in (None, ""):
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page + 1 if page == 0 else page


def _extract_bbox(item: dict[str, Any]) -> list[float] | None:
    value = _first(item, ["bbox", "box", "position"], None)
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [float(x) for x in value[:4]]
    except (TypeError, ValueError):
        return None


def _resolve_media_path(item: dict[str, Any], asset_root: Path) -> Path | None:
    value = _first(item, ["img_path", "image_path", "table_path", "asset_path", "path"], None)
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = asset_root / path
    return path


def _split_text(text: str, max_token_size: int, overlap_token_size: int) -> list[str]:
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
