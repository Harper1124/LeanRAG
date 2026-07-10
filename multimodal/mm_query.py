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
    working_dir = global_config["working_dir"]
    chunks, media_items = _load_mm_artifacts(working_dir)
    mm_retrieval_trace = _run_phase2_direct_recall(global_config, query, doc_id) if _use_mm_hybrid(global_config) else _empty_direct_trace()
    mm_retrieval_trace = _run_phase4_graph_expansion(global_config, mm_retrieval_trace, doc_id)
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
    context = _format_context(text_evidence, graph_evidence, visual_evidence, table_evidence, global_config)

    phase3_answer, phase3_trace = _run_phase3_answer_pipeline(global_config, db, query, mm_retrieval_trace)
    if phase3_answer is not None:
        answer = phase3_answer
    elif visual_evidence and global_config.get("answer_with_vlm_when_media", True):
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
    trace.update(mm_retrieval_trace)
    trace.update(phase3_trace)
    return str(answer), trace


def _use_mm_hybrid(global_config: dict) -> bool:
    retrieval = global_config.get("retrieval")
    if isinstance(retrieval, dict):
        return retrieval.get("mode", "mm_hybrid") == "mm_hybrid"
    return global_config.get("retrieval_mode", "mm_hybrid") == "mm_hybrid"


def _empty_direct_trace() -> dict:
    return {
        "query_info": {},
        "retrieved_nodes_by_type": {"text": [], "entity": [], "media": [], "page": [], "aggregate": []},
        "direct_recall": {"text": [], "entity": [], "media": [], "page": []},
        "merged_candidates": [],
        "failure_stage": None,
    }


def _run_phase2_direct_recall(global_config: dict, query: str, doc_id: str | None) -> dict:
    try:
        from .retrieval.mm_retriever import mm_hybrid_retrieve

        _, _, trace = mm_hybrid_retrieve(query, global_config, doc_id=doc_id)
        return trace
    except Exception as exc:
        return {
            "query_info": {},
            "retrieved_nodes_by_type": {"text": [], "entity": [], "media": [], "page": [], "aggregate": []},
            "direct_recall": {"text": [], "entity": [], "media": [], "page": []},
            "merged_candidates": [],
            "failure_stage": "direct_recall_failed",
            "direct_recall_error": str(exc),
        }


def _run_phase4_graph_expansion(global_config: dict, retrieval_trace: dict, doc_id: str | None) -> dict:
    graph_trace = {
        "enabled": _graph_expansion_enabled(global_config),
        "expanded_nodes": [],
        "expanded_edges": [],
        "num_expanded_nodes": 0,
        "num_expanded_edges": 0,
        "errors": [],
    }
    retrieval_trace["graph_expansion"] = graph_trace
    if not graph_trace["enabled"]:
        return retrieval_trace
    anchors = retrieval_trace.get("merged_candidates") or []
    if not anchors:
        return retrieval_trace
    try:
        from .graph.graph_expander import expand_graph
        from .graph.mm_graph_loader import load_mm_graph
        from .retrieval.candidate_merge import merge_candidates

        working_dir = global_config["working_dir"]
        graph_config = global_config.get("graph_expansion") or {}
        graph = load_mm_graph(
            working_dir,
            node_file=global_config.get("nodes_file") or global_config.get("node_file") or "mm_nodes.jsonl",
            edge_file=graph_config.get("edge_file", "mm_edges.jsonl"),
            edge_seed_file=graph_config.get("edge_seed_file", "mm_edges_seed.jsonl"),
        )
        expanded_candidates, expanded_edges = expand_graph(
            anchors=anchors,
            graph=graph,
            query_info=retrieval_trace.get("query_info") or {},
            config=global_config,
            doc_id=doc_id,
        )
        fusion_config = global_config.get("fusion") or {}
        merged = merge_candidates(
            list(anchors) + list(expanded_candidates),
            multi_hit_bonus=float(fusion_config.get("multi_hit_bonus", 0.10)),
        )
        retrieval_trace["merged_candidates"] = merged
        retrieval_trace["retrieved_nodes_by_type"] = _retrieved_nodes_by_type(merged, global_config)
        graph_trace["expanded_nodes"] = [_slim_expanded_node(item) for item in expanded_candidates]
        graph_trace["expanded_edges"] = expanded_edges
        graph_trace["num_expanded_nodes"] = len(expanded_candidates)
        graph_trace["num_expanded_edges"] = len(expanded_edges)
        if getattr(graph, "warnings", None):
            graph_trace["warnings"] = graph.warnings
    except Exception as exc:
        graph_trace["errors"].append(str(exc))
        retrieval_trace["failure_stage"] = "graph_expansion_failed"
    return retrieval_trace


def _graph_expansion_enabled(global_config: dict) -> bool:
    section = global_config.get("graph_expansion")
    if isinstance(section, dict):
        return bool(section.get("enabled", True))
    return False


def _slim_expanded_node(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": item.get("node_id"),
        "node_type": item.get("node_type"),
        "page_id": item.get("page_id"),
        "score": float(item.get("score") or 0.0),
        "source": item.get("source", "graph_expansion"),
        "debug": item.get("debug") or {},
    }


def _retrieved_nodes_by_type(merged_candidates: list[dict[str, Any]], global_config: dict) -> dict[str, list[dict[str, Any]]]:
    budget = global_config.get("evidence_budget") or {}
    retrieved = {"text": [], "entity": [], "media": [], "page": [], "aggregate": []}
    for item in merged_candidates:
        node_type = item.get("node_type")
        if node_type in retrieved:
            retrieved[node_type].append(item)
    limits = {
        "text": int(budget.get("max_text_nodes", 6)),
        "entity": int(budget.get("max_entity_nodes", 12)),
        "media": int(budget.get("max_media_nodes", 6)),
        "page": int(budget.get("max_page_nodes", 4)),
        "aggregate": 12,
    }
    return {key: value[: limits[key]] for key, value in retrieved.items()}


def _run_phase3_answer_pipeline(global_config: dict, db, query: str, retrieval_trace: dict) -> tuple[str | None, dict]:
    if not _use_mm_hybrid(global_config):
        return None, {}
    merged_candidates = retrieval_trace.get("merged_candidates") or []
    if not merged_candidates:
        return None, {}
    try:
        from .generation.answer_planner import plan_answer
        from .generation.deterministic_answer import try_deterministic_answer
        from .generation.evidence_package import build_evidence_package
        from .generation.final_generator import generate_final_answer
        from .generation.table_reasoner import run_table_reasoner
        from .generation.vlm_reasoner import run_vlm_reasoner
        from .graph.context_aggregation import merge_aggregation_context, multimodal_context_aggregation
        from .graph.mm_graph_loader import load_mm_graph

        query_info = retrieval_trace.get("query_info") or {}
        evidence_package = build_evidence_package(merged_candidates, query_info, global_config)
        deterministic_answer, deterministic_trace = try_deterministic_answer(query, global_config)
        if deterministic_answer is not None:
            answer_plan = plan_answer(query, evidence_package, query_info, global_config)
            return deterministic_answer, {
                "evidence_package": evidence_package,
                "answer_plan": answer_plan,
                "vlm_calls": [],
                "table_reasoner_calls": [],
                "selected_evidence_nodes": evidence_package.get("all_selected_nodes", []),
                "context_aggregation": {"enabled": _context_aggregation_enabled(global_config), "skipped": True},
                "deterministic_answer": deterministic_trace,
                "final_generation": {
                    "prompt_preview": "",
                    "used_llm": False,
                    "error": None,
                    "raw_answer": deterministic_answer,
                    "postprocess": {"target": query_info.get("expected_answer_type"), "changed": False, "rule": "deterministic"},
                },
                "failure_stage": None,
            }
        context_agg = {"enabled": False}
        if _context_aggregation_enabled(global_config):
            try:
                graph_config = global_config.get("graph_expansion") or {}
                graph = load_mm_graph(
                    global_config["working_dir"],
                    node_file=global_config.get("nodes_file") or global_config.get("node_file") or "mm_nodes.jsonl",
                    edge_file=graph_config.get("edge_file", "mm_edges.jsonl"),
                    edge_seed_file=graph_config.get("edge_seed_file", "mm_edges_seed.jsonl"),
                )
                context_agg = multimodal_context_aggregation(
                    question=query,
                    evidence_package=evidence_package,
                    merged_candidates=merged_candidates,
                    graph=graph,
                    db=db,
                    global_config=global_config,
                    query_info=query_info,
                )
                evidence_package = merge_aggregation_context(evidence_package, context_agg, global_config)
            except Exception as exc:
                context_agg = {
                    "enabled": True,
                    "mode": "media_projection_lca",
                    "anchor_nodes": [],
                    "projected_nodes": [],
                    "filtered_projection_candidates": [],
                    "lca_input_entities": [],
                    "lca_result": {},
                    "page_context_nodes": [],
                    "kept_media_nodes": [],
                    "aggregation_context_added": False,
                    "errors": [str(exc)],
                }
        answer_plan = plan_answer(query, evidence_package, query_info, global_config)
        max_vlm_images = int((global_config.get("evidence_budget") or {}).get("max_vlm_images", 4))
        vlm_calls = (
            run_vlm_reasoner(query, answer_plan.get("selected_visual_nodes", []), global_config.get("use_vlm_func"), max_vlm_images)
            if answer_plan.get("use_vlm")
            else []
        )
        table_calls = (
            run_table_reasoner(
                query,
                answer_plan.get("selected_table_nodes", []),
                global_config.get("use_llm_func"),
                max_context_chars=int((global_config.get("generation") or {}).get("max_prompt_chars", 24000)) // 2,
            )
            if answer_plan.get("use_table_reasoner")
            else []
        )
        answer, final_generation = generate_final_answer(
            query,
            evidence_package,
            answer_plan,
            vlm_calls,
            table_calls,
            global_config,
        )
        trace = {
            "evidence_package": evidence_package,
            "answer_plan": answer_plan,
            "vlm_calls": vlm_calls,
            "table_reasoner_calls": table_calls,
            "selected_evidence_nodes": evidence_package.get("all_selected_nodes", []),
            "context_aggregation": context_agg,
            "deterministic_answer": deterministic_trace,
            "final_generation": final_generation,
            "failure_stage": _phase3_failure_stage(evidence_package, vlm_calls, table_calls),
        }
        return answer, trace
    except Exception as exc:
        return None, {
            "evidence_package": {
                "text_evidence": [],
                "entity_evidence": [],
                "page_evidence": [],
                "visual_evidence": [],
                "table_evidence": [],
                "all_selected_nodes": [],
            },
            "answer_plan": {},
            "vlm_calls": [],
            "table_reasoner_calls": [],
            "selected_evidence_nodes": [],
            "context_aggregation": {"enabled": _context_aggregation_enabled(global_config), "errors": [str(exc)]},
            "failure_stage": "generation_failed",
            "generation_error": str(exc),
        }


def _phase3_failure_stage(evidence_package: dict, vlm_calls: list[dict], table_calls: list[dict]) -> str | None:
    if not evidence_package.get("all_selected_nodes"):
        return "not_enough_evidence"
    if any(call.get("error") and not call.get("vlm_answer") for call in vlm_calls):
        return "vlm_failed"
    if any(call.get("error") and not call.get("table_answer") for call in table_calls):
        return "table_reasoner_failed"
    return None


def _context_aggregation_enabled(global_config: dict) -> bool:
    section = global_config.get("context_aggregation")
    if isinstance(section, dict):
        return bool(section.get("enabled", False))
    return False


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


def _format_context(text_evidence, graph_evidence, visual_evidence, table_evidence, global_config: dict | None = None) -> str:
    # Keep traces complete, but send a compact answer-facing evidence view to the model.
    global_config = global_config or {}
    max_chars = int(global_config.get("max_evidence_field_chars", 2000))
    payload = {
        "text_evidence": [_slim_text_evidence(item, max_chars) for item in text_evidence],
        "graph_evidence": graph_evidence,
        "visual_evidence": [_slim_media_evidence(item, max_chars, include_table=False) for item in visual_evidence],
        "table_evidence": [_slim_media_evidence(item, max_chars, include_table=True) for item in table_evidence],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _slim_text_evidence(item: dict, max_chars: int) -> dict:
    keys = ("chunk_id", "hash_code", "doc_id", "text", "page_start", "page_end", "section_title", "bbox", "order", "score")
    return {key: _truncate_value(item.get(key), max_chars) for key in keys if key in item}


def _slim_media_evidence(item: dict, max_chars: int, include_table: bool) -> dict:
    keys = ["media_id", "doc_id", "modality", "page", "path", "caption", "ocr_text", "summary", "bbox", "score"]
    if include_table:
        keys.append("table_markdown")
    return {key: _truncate_value(item.get(key), max_chars) for key in keys if key in item}


def _truncate_value(value, max_chars: int):
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars].rstrip() + "\n...[truncated]"
    return value


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
    parser.add_argument("--output_file", default=None, help="Optional JSON file path for saving the answer and trace.")
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
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"answer": answer, "trace": trace}
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved full result to {output_path}")
    else:
        print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
