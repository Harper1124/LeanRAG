from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .io_utils import load_dataclasses, read_json
from .schema import MMChunk, MMMedia, dataclass_to_dict


# 文本模型回答提示词：保持 LeanRAG prompt.py 的 Role/Goal/Rules/Workflows 结构，同时约束评测友好的短答案格式。
MM_TEXT_RESPONSE_PROMPT = """# Role: Multimodal Document Evidence Response Generator

## Profile
- language: English
- description: You are a precise document question-answering assistant. Your task is to answer the user's question using only the provided text, graph, visual, and table evidence.

## Goal
- Produce the shortest correct final answer supported by the evidence.
- Match the answer type implied by the question: string, integer, float, list, or not answerable.
- Avoid explanations that dilute exact-match, numeric, or list-based evaluation.

## Rules
- Use only the provided evidence. Do not fabricate, infer from outside knowledge, or guess.
- Return only the final answer. Do not include reasoning, citations, page references, quotes, markdown, or commentary.
- For numeric questions, return only the number and unit if the unit is required.
- For list questions, return a comma-separated list and nothing else.
- If the evidence does not contain enough information to answer, return exactly: Not answerable

## Workflows
1. Identify the answer type requested by the question.
2. Inspect text evidence first, then graph evidence, then visual and table evidence when present.
3. Select the smallest evidence-supported answer span or value.
4. Verify that the final answer is directly supported and follows the output rules.
"""


# 视觉模型回答提示词：用于带图片证据的问题，强调图表/布局/颜色等视觉信息。
MM_VISUAL_RESPONSE_PROMPT = """# Role: Multimodal Document Visual Evidence Response Generator

## Profile
- language: English
- description: You are a precise multimodal document question-answering assistant. Your task is to answer the user's question using only the provided text, graph, visual, table evidence, and attached images.

## Goal
- Produce the shortest correct final answer supported by the evidence and images.
- Match the answer type implied by the question: string, integer, float, list, or not answerable.
- Use visual evidence when the question asks about figures, charts, tables, diagrams, screenshots, colors, layouts, or other visual content.

## Rules
- Use only the provided evidence and attached images. Do not fabricate, infer from outside knowledge, or guess.
- Return only the final answer. Do not include reasoning, citations, page references, quotes, markdown, or commentary.
- For numeric questions, return only the number and unit if the unit is required.
- For list questions, return a comma-separated list and nothing else.
- If the evidence and images do not contain enough information to answer, return exactly: Not answerable

## Workflows
1. Identify whether the question requires text, table, figure, chart, or image understanding.
2. Inspect the attached images together with the structured evidence.
3. Extract the smallest evidence-supported answer span, value, or list.
4. Verify that the final answer is directly supported and follows the output rules.
"""


def query_mm_graph(
    global_config: dict,
    db,
    query: str,
    doc_id: str | None = None,
) -> tuple[str, dict]:
    """
    Multimodal LeanRAG query entry.

    Text retrieval is delegated to existing LeanRAG graph search when available.
    Returned text chunk ids are used to backfill attached images/tables.
    """
    del db
    working_dir = global_config["working_dir"]
    chunks, media_items = _load_mm_artifacts(working_dir)
    # 优先走 LeanRAG 的图/向量检索；检索失败时再退回关键词检索。
    text_evidence, graph_evidence, selected_entities = _retrieve_text_evidence(global_config, query, chunks)
    if not text_evidence:
        text_evidence = _keyword_retrieve(query, chunks, topk=global_config.get("text_topk", 5))

    direct_text, direct_visual, direct_tables = _direct_media_retrieve(global_config, query, chunks, media_items)
    text_evidence = _merge_evidence(text_evidence, direct_text, "chunk_id")[: global_config.get("text_topk", 5)]

    # 媒体证据同时使用两条通道：文本 chunk 回填，以及页码/全局/语义视觉直检。
    visual_evidence, table_evidence = _media_for_text_evidence(text_evidence, chunks, media_items)
    visual_evidence = _merge_evidence(direct_visual, visual_evidence, "media_id")[: _media_limit(global_config, query, "image")]
    table_evidence = _merge_evidence(direct_tables, table_evidence, "media_id")[: _media_limit(global_config, query, "table")]
    context = _format_context(text_evidence, graph_evidence, visual_evidence, table_evidence)

    if visual_evidence and global_config.get("answer_with_vlm_when_media", True):
        # 有图片证据时优先使用 VLM；不可用时自动降级到普通 LLM。
        answer = _call_vlm(global_config.get("use_vlm_func"), query, context, [item["path"] for item in visual_evidence])
        if answer is None:
            answer = _call_llm(global_config.get("use_llm_func"), query, context)
    else:
        answer = _call_llm(global_config.get("use_llm_func"), query, context)
    if answer is None:
        answer = context

    trace = {
        "text_evidence": text_evidence,
        "graph_evidence": graph_evidence,
        "visual_evidence": visual_evidence,
        "table_evidence": table_evidence,
        "selected_entities": selected_entities,
    }
    return str(answer), trace


def _retrieve_text_evidence(global_config: dict, query: str, chunks: list[MMChunk]):
    # 复用原 LeanRAG 的 entity vector search，再通过 source_id 找回文本 chunk。
    try:
        from database_utils import get_text_units, search_vector_search
    except Exception:
        return [], [], []
    embedding_func = global_config.get("embeddings_func")
    if not embedding_func:
        return [], [], []
    try:
        query_embedding = embedding_func(query)
        entity_results = search_vector_search(
            global_config["working_dir"],
            query_embedding,
            topk=global_config.get("topk", 10),
            level_mode=global_config.get("level_mode", 2),
        )
        source_ids = [item[-1] for item in entity_results]
        chunk_file = global_config.get("chunks_file") or str(Path(global_config["working_dir"]) / "leanrag_chunk.json")
        text_units = get_text_units(global_config["working_dir"], source_ids, chunk_file, k=global_config.get("text_topk", 5))
        chunk_by_hash = {chunk.hash_code: chunk for chunk in chunks}
        evidence = []
        for item in text_units:
            # get_text_units 只返回 hash/text 等 LeanRAG 字段时，用 mm_chunk 补回页码、bbox、媒体 id。
            chunk = chunk_by_hash.get(item.get("hash_code"))
            evidence.append(_chunk_evidence(chunk, item.get("score", 0)) if chunk else item)
        selected = [{"entity_name": item[0], "parent": item[1], "description": item[2], "source_id": item[3]} for item in entity_results]
        return evidence, [], selected
    except Exception:
        return [], [], []


def _keyword_retrieve(query: str, chunks: list[MMChunk], topk: int = 5) -> list[dict]:
    # 简单关键词兜底：不依赖 embedding 或图数据库，保证缺少索引时仍能返回上下文。
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    scored = []
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(text.count(term) for term in query_terms)
        if score:
            scored.append((score, chunk))
    if not scored:
        scored = [(1, chunk) for chunk in chunks[:topk]]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_chunk_evidence(chunk, score) for score, chunk in scored[:topk]]


def _media_for_text_evidence(text_evidence: list[dict], chunks: list[MMChunk], media_items: list[MMMedia]):
    # 根据文本证据 chunk 上的 attached_media_ids 聚合图片/表格，按出现次数排序。
    chunks_by_hash = {chunk.hash_code: chunk for chunk in chunks}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    media_by_id = {item.media_id: item for item in media_items}
    counter = Counter()
    for evidence in text_evidence:
        chunk = chunks_by_hash.get(evidence.get("hash_code")) or chunks_by_id.get(evidence.get("chunk_id"))
        if not chunk:
            continue
        counter.update(chunk.attached_media_ids)
    media = [media_by_id[media_id] for media_id, _ in counter.most_common() if media_id in media_by_id]
    visual = [_media_evidence(item, counter[item.media_id]) for item in media if item.modality == "image"]
    tables = [_media_evidence(item, counter[item.media_id]) for item in media if item.modality == "table"]
    return visual, tables


def _direct_media_retrieve(global_config: dict, query: str, chunks: list[MMChunk], media_items: list[MMMedia]):
    mode = _visual_query_mode(query)
    if not mode:
        return [], [], []

    pages = _query_pages(query, chunks, media_items)
    text_evidence = _page_text_evidence(chunks, pages, limit=global_config.get("page_text_topk", 6)) if pages else []
    if mode == "global":
        media = _all_relevant_media(query, media_items)
    elif pages:
        media = [item for item in media_items if item.page in pages]
    else:
        topk = max(_media_limit(global_config, query, "image"), _media_limit(global_config, query, "table"))
        media = _keyword_media_retrieve(query, media_items, topk=topk)

    visual = [_media_evidence(item, _media_query_score(query, item)) for item in media if item.modality == "image"]
    tables = [_media_evidence(item, _media_query_score(query, item)) for item in media if item.modality == "table"]
    if mode == "global":
        visual.sort(key=lambda item: (item.get("page") or 0, item.get("media_id") or ""))
        tables.sort(key=lambda item: (item.get("page") or 0, item.get("media_id") or ""))
    else:
        visual.sort(key=lambda item: (-(item.get("score", 0)), item.get("page") or 0, item.get("media_id") or ""))
        tables.sort(key=lambda item: (-(item.get("score", 0)), item.get("page") or 0, item.get("media_id") or ""))
    return text_evidence, visual, tables


def _visual_query_mode(query: str) -> str | None:
    text = query.lower()
    global_terms = ("all pictures", "all images", "all figures", "all charts", "among all", "which page", "how many pictures", "how many images")
    visual_terms = ("image", "picture", "figure", "fig.", "chart", "plot", "diagram", "slide", "screenshot", "visual", "color", "shape")
    table_terms = ("table", "row", "column")
    if any(term in text for term in global_terms):
        return "global"
    if _query_pages(query, [], []):
        return "page"
    if any(term in text for term in visual_terms + table_terms):
        return "semantic"
    return None


def _query_pages(query: str, chunks: list[MMChunk], media_items: list[MMMedia]) -> set[int]:
    text = query.lower()
    pages = {int(match) for match in re.findall(r"\bpage\s*(?:no\.?|number)?\s*(\d+)\b", text)}
    ordinal_pages = {
        "first page": 1,
        "second page": 2,
        "third page": 3,
        "fourth page": 4,
        "fifth page": 5,
        "sixth page": 6,
        "seventh page": 7,
        "eighth page": 8,
        "ninth page": 9,
        "tenth page": 10,
    }
    pages.update(page for phrase, page in ordinal_pages.items() if phrase in text)
    if "last page" in text:
        max_page = max(
            [item.page or 0 for item in media_items] + [chunk.page_end or chunk.page_start or 0 for chunk in chunks],
            default=0,
        )
        if max_page:
            pages.add(max_page)
    return {page for page in pages if page > 0}


def _page_text_evidence(chunks: list[MMChunk], pages: set[int], limit: int = 6) -> list[dict]:
    matches = []
    for chunk in chunks:
        start = chunk.page_start
        end = chunk.page_end or start
        if start is None:
            continue
        if any(start <= page <= (end or start) for page in pages):
            matches.append(_chunk_evidence(chunk, 10.0))
    matches.sort(key=lambda item: (item.get("page_start") or 0, item.get("order") or 0))
    return matches[:limit]


def _all_relevant_media(query: str, media_items: list[MMMedia]) -> list[MMMedia]:
    text = query.lower()
    wants_table = "table" in text
    wants_image = any(term in text for term in ("picture", "image", "figure", "chart", "plot", "diagram", "slide", "visual", "color", "shape"))
    if wants_table and not wants_image:
        return [item for item in media_items if item.modality == "table"]
    if wants_image:
        return [item for item in media_items if item.modality == "image"]
    return list(media_items)


def _keyword_media_retrieve(query: str, media_items: list[MMMedia], topk: int = 8) -> list[MMMedia]:
    scored = []
    for item in media_items:
        score = _media_query_score(query, item)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (pair[0], pair[1].page or 0), reverse=True)
    return [item for _, item in scored[:topk]]


def _media_query_score(query: str, item: MMMedia) -> float:
    query_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9%/-]+", query) if len(term) > 2]
    media_text = " ".join([item.caption, item.ocr_text, item.summary, item.table_markdown, item.table_html]).lower()
    score = sum(media_text.count(term) for term in query_terms)
    if item.page and re.search(rf"\bpage\s*(?:no\.?|number)?\s*{item.page}\b", query.lower()):
        score += 10
    return float(score)


def _media_limit(global_config: dict, query: str, modality: str) -> int:
    if _visual_query_mode(query) == "global":
        key = "global_max_images_per_query" if modality == "image" else "global_max_tables_per_query"
        fallback = 64 if modality == "image" else 24
        return int(global_config.get(key, fallback))
    key = "max_images_per_query" if modality == "image" else "max_tables_per_query"
    return int(global_config.get(key, 4))


def _merge_evidence(primary: list[dict], secondary: list[dict], key: str) -> list[dict]:
    merged = []
    seen = set()
    for item in list(primary or []) + list(secondary or []):
        item_key = item.get(key)
        if item_key is None:
            item_key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if item_key in seen:
            continue
        seen.add(item_key)
        merged.append(item)
    return merged


def _chunk_evidence(chunk: MMChunk, score: float = 0.0) -> dict:
    item = dataclass_to_dict(chunk)
    item["score"] = score
    return item


def _media_evidence(media: MMMedia, score: float = 0.0) -> dict:
    item = dataclass_to_dict(media)
    item["score"] = score
    return item


def _format_context(text_evidence, graph_evidence, visual_evidence, table_evidence) -> str:
    # 将所有证据序列化为 JSON，方便模型看到页码、bbox、路径和摘要等结构化字段。
    payload = {
        "text_evidence": text_evidence,
        "graph_evidence": graph_evidence,
        "visual_evidence": visual_evidence,
        "table_evidence": table_evidence,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _call_llm(func: Callable | None, query: str, context: str):
    if not func:
        return None
    try:
        # 兼容支持 system_prompt 的聊天函数。
        return func(query, system_prompt=f"{MM_TEXT_RESPONSE_PROMPT}\n\n## Evidence\n{context}")
    except TypeError:
        # 兼容只接受单字符串 prompt 的函数。
        return func(f"{MM_TEXT_RESPONSE_PROMPT}\n\n## Question\n{query}\n\n## Evidence\n{context}")


def _call_vlm(func: Callable | None, query: str, context: str, image_paths: list[str]):
    if not func:
        return None
    prompt = f"{MM_VISUAL_RESPONSE_PROMPT}\n\n## Question\n{query}\n\n## Evidence\n{context}"
    try:
        return func(prompt=prompt, image_paths=image_paths)
    except TypeError:
        pass
    try:
        return func(query=prompt, image_paths=image_paths)
    except TypeError:
        pass
    try:
        return func(prompt, image_paths)
    except TypeError:
        return func(prompt)


def _load_mm_artifacts(working_dir: str) -> tuple[list[MMChunk], list[MMMedia]]:
    working = Path(working_dir)
    chunks = load_dataclasses(working / "mm_chunk.json", MMChunk)
    media = load_dataclasses(working / "mm_media.json", MMMedia) if (working / "mm_media.json").exists() else []
    return chunks, media


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return {}


def _install_default_model_funcs(config: dict[str, Any], full_config: dict[str, Any]) -> None:
    # 从 config.yaml 自动安装 LLM、embedding、VLM 函数，调用方也可以预先传入自定义函数覆盖。
    try:
        from .openai_clients import make_chat_func, make_vlm_func
    except Exception:
        return
    if "use_llm_func" not in config and full_config.get("deepseek", {}).get("base_url"):
        llm_conf = dict(full_config["deepseek"])
        llm_conf.setdefault("api_key_env", "DASHSCOPE_API_KEY")
        config["use_llm_func"] = make_chat_func(llm_conf)
    if "embeddings_func" not in config:
        try:
            from query_graph import embedding

            config["embeddings_func"] = embedding
        except Exception:
            pass
    if "use_vlm_func" not in config:
        vlm_conf = {
            "model": config.get("vlm_model"),
            "base_url": config.get("vlm_base_url"),
            "api_key": config.get("vlm_api_key", ""),
            "api_key_env": config.get("vlm_api_key_env", "DASHSCOPE_API_KEY"),
        }
        if vlm_conf["model"] and vlm_conf["base_url"]:
            config["use_vlm_func"] = make_vlm_func(vlm_conf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a multimodal LeanRAG working directory.")
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    full_config = _load_config(args.config)
    config = full_config.get("multimodal", {})
    _install_default_model_funcs(config, full_config)
    config.update(
        {
            "working_dir": args.working_dir,
            "chunks_file": str(Path(args.working_dir) / "leanrag_chunk.json"),
        }
    )
    answer, trace = query_mm_graph(config, None, args.query, doc_id=args.doc_id)
    print(answer)
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
