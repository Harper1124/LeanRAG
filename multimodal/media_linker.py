from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .io_utils import write_json
from .schema import MMChunk, MMMedia, is_indexable_media


# 将图片/表格与文本 chunk 建立弱关联，方便查询时从文本证据带出相关媒体证据。
def link_media_to_chunks(
    chunks: list[MMChunk],
    media_items: list[MMMedia],
    embedding_func: Callable | None,
    page_window: int = 1,
    topk_per_media: int = 3,
) -> tuple[list[MMChunk], list[MMMedia]]:
    """Attach images/tables to nearby and semantically related text chunks."""
    indexable_media = [item for item in media_items if is_indexable_media(item)]
    noise_ids = {item.media_id for item in media_items if not is_indexable_media(item)}
    for chunk in chunks:
        chunk.attached_media_ids = [media_id for media_id in chunk.attached_media_ids if media_id not in noise_ids]
    for item in media_items:
        if item.media_id in noise_ids:
            item.nearby_chunk_ids.clear()
            item.attached_entity_names.clear()
            item.attach_scores.clear()

    text_chunks = [chunk for chunk in chunks if chunk.modality == "text" and chunk.text]
    chunk_vectors = _embed_texts(embedding_func, [chunk.text for chunk in text_chunks])
    media_vectors = _embed_texts(embedding_func, [_media_text(item) for item in indexable_media])

    for media_index, media in enumerate(indexable_media):
        scored = []
        for chunk_index, chunk in enumerate(text_chunks):
            # 综合页码邻近、语义相似、文档顺序和显式提及四类信号打分。
            page_score = _page_window_score(media.page, chunk.page_start, page_window)
            if page_score <= 0 and media.page is not None and chunk.page_start is not None:
                continue
            sim_score = _cosine(media_vectors[media_index], chunk_vectors[chunk_index]) if media_vectors is not None else 0.0
            order_score = _nearby_order_score(media_index, chunk.order, len(text_chunks))
            mention_score = _explicit_mention_score(chunk.text, media)
            score = 0.40 * page_score + 0.30 * sim_score + 0.20 * order_score + 0.10 * mention_score
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, chunk in scored[:topk_per_media]:
            if score <= 0:
                continue
            if media.media_id not in chunk.attached_media_ids:
                chunk.attached_media_ids.append(media.media_id)
            if chunk.chunk_id not in media.nearby_chunk_ids:
                media.nearby_chunk_ids.append(chunk.chunk_id)
            media.attach_scores[chunk.chunk_id] = round(float(score), 6)
    return chunks, media_items


def link_media_to_entities(
    working_dir: str,
    chunks: list[MMChunk],
    media_items: list[MMMedia],
    embedding_func: Callable | None,
    topk_per_entity: int = 5,
) -> list[MMMedia]:
    """
    Attach media to entities through entity.source_id -> chunk.hash_code.

    The function writes entity_media.json for query-time lookup and updates each
    media item's attached_entity_names.
    """
    del embedding_func
    working = Path(working_dir)
    entity_path = working / "entity.jsonl"
    if not entity_path.exists():
        # 没有实体文件时仍写空索引，查询侧可以用文件存在性判断构建完成。
        write_json({}, working / "entity_media.json")
        return media_items

    # entity.source_id 通常指向 chunk.hash_code；据此把实体和 chunk 附带媒体连接起来。
    indexable_media = [item for item in media_items if is_indexable_media(item)]
    allowed_ids = {item.media_id for item in indexable_media}
    chunks_by_hash = {chunk.hash_code: chunk for chunk in chunks}
    media_by_chunk = {}
    for chunk in chunks:
        attached = [media_id for media_id in chunk.attached_media_ids if media_id in allowed_ids]
        media_by_chunk[chunk.hash_code] = attached
        media_by_chunk[chunk.chunk_id] = attached
    entity_media: dict[str, list[str]] = {}
    with entity_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            entity = json.loads(line)
            name = str(entity.get("entity_name", "")).strip()
            if not name:
                continue
            source_ids = _split_source_ids(entity.get("source_id", ""))
            attached = []
            for source_id in source_ids:
                chunk = chunks_by_hash.get(source_id)
                if chunk:
                    attached.extend(chunk.attached_media_ids)
                attached.extend(media_by_chunk.get(source_id, []))
            unique = list(dict.fromkeys(attached))[:topk_per_entity]
            entity_media[name] = unique
            for media in indexable_media:
                if media.media_id in unique and name not in media.attached_entity_names:
                    media.attached_entity_names.append(name)
    write_json(entity_media, working / "entity_media.json")
    return media_items


def _embed_texts(embedding_func: Callable | None, texts: list[str]) -> np.ndarray | None:
    # embedding 不是强依赖；失败时返回 None，主流程会退化为页码/顺序等启发式关联。
    if not embedding_func or not texts:
        return None
    try:
        import numpy as np

        vectors = embedding_func(texts)
    except Exception:
        return None
    arr = np.asarray(vectors, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _media_text(media: MMMedia) -> str:
    return "\n".join(
        part for part in [media.caption, media.ocr_text, media.summary, media.table_markdown, media.table_html] if part
    )


def _page_window_score(media_page: int | None, chunk_page: int | None, page_window: int) -> float:
    # 页码缺失时给一个中性分，避免直接丢弃可能相关的媒体。
    if media_page is None or chunk_page is None:
        return 0.5
    distance = abs(media_page - chunk_page)
    if distance > page_window:
        return 0.0
    return 1.0 - (distance / (page_window + 1))


def _nearby_order_score(media_index: int, chunk_order: int, chunk_count: int) -> float:
    if chunk_count <= 1:
        return 1.0
    expected_order = min(chunk_count - 1, media_index)
    return max(0.0, 1.0 - abs(chunk_order - expected_order) / max(chunk_count, 1))


def _explicit_mention_score(text: str, media: MMMedia) -> float:
    # 如果文本显式出现 figure/table 等词，或与媒体摘要有词重合，则提高关联分。
    haystack = text.lower()
    keywords = ["figure", "fig.", "image", "table"] if media.modality == "image" else ["table", "tab."]
    score = 1.0 if any(keyword in haystack for keyword in keywords) else 0.0
    words = set(re.findall(r"[A-Za-z0-9_%-]+", _media_text(media).lower()))
    text_words = set(re.findall(r"[A-Za-z0-9_%-]+", haystack))
    overlap = len(words & text_words)
    return min(1.0, score + overlap / 20.0)


def _cosine(a, b) -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not denom or math.isnan(denom):
        return 0.0
    return max(0.0, min(1.0, float(np.dot(a, b) / denom)))


def _split_source_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]
