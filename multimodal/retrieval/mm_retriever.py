from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..io_utils import read_jsonl
from .candidate_merge import merge_candidates
from .media_retriever import MediaRetriever
from .page_retriever import PageRetriever
from .query_analyzer import analyze_query


DEFAULT_RETRIEVAL = {
    "mode": "mm_hybrid",
    "topk_text": 8,
    "topk_entity": 12,
    "topk_media": 8,
    "topk_page": 4,
    "always_run_all_retrievers": True,
}
DEFAULT_FUSION = {"mode": "rule_based", "multi_hit_bonus": 0.10, "page_hint_bonus": 0.20, "modality_bonus": 0.15}
DEFAULT_BUDGET = {"max_text_nodes": 6, "max_entity_nodes": 12, "max_media_nodes": 6, "max_table_nodes": 4, "max_page_nodes": 4}


def mm_hybrid_retrieve(query: str, global_config: dict[str, Any], doc_id: str | None = None):
    working_dir = Path(global_config.get("working_dir", "."))
    nodes_path = Path(global_config.get("nodes_file") or global_config.get("node_file") or working_dir / "mm_nodes.jsonl")
    if not nodes_path.exists():
        return [], analyze_query(query), _empty_trace("node_file_missing", str(nodes_path))

    retrieval_config = _section(global_config, "retrieval", DEFAULT_RETRIEVAL)
    fusion_config = _section(global_config, "fusion", DEFAULT_FUSION)
    budget_config = _section(global_config, "evidence_budget", DEFAULT_BUDGET)
    query_info = analyze_query(query)
    nodes = [node for node in read_jsonl(nodes_path) if doc_id is None or node.get("doc_id") == doc_id]

    text_candidates = _node_lexical_search(
        nodes,
        query,
        "text",
        topk=_dynamic_topk(retrieval_config["topk_text"], query_info, "text"),
        retriever="text_retriever",
    )
    entity_candidates = _node_lexical_search(
        nodes,
        query,
        "entity",
        topk=_dynamic_topk(retrieval_config["topk_entity"], query_info, "entity"),
        retriever="entity_retriever",
    )
    media_candidates = MediaRetriever(nodes_path, doc_id=doc_id).search(
        query,
        query_info=query_info,
        topk=_dynamic_topk(retrieval_config["topk_media"], query_info, "media"),
    )
    page_candidates = PageRetriever(nodes_path, doc_id=doc_id).search(
        query,
        query_info=query_info,
        seed_candidates=text_candidates + media_candidates,
        topk=_dynamic_topk(retrieval_config["topk_page"], query_info, "page"),
    )

    candidates = text_candidates + entity_candidates + media_candidates + page_candidates
    merged = merge_candidates(candidates, multi_hit_bonus=float(fusion_config["multi_hit_bonus"]))
    merged = _apply_rule_bonuses(merged, query_info, fusion_config)
    trace = build_retrieval_trace(
        query_info,
        {
            "text": text_candidates,
            "entity": entity_candidates,
            "media": media_candidates,
            "page": page_candidates,
        },
        merged,
        budget_config,
    )
    return merged, query_info, trace


def build_retrieval_trace(
    query_info: dict[str, Any],
    direct: dict[str, list[dict[str, Any]]],
    merged: list[dict[str, Any]],
    budget_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    budget_config = budget_config or DEFAULT_BUDGET
    retrieved = {"text": [], "entity": [], "media": [], "page": [], "aggregate": []}
    for item in merged:
        node_type = item.get("node_type")
        if node_type in retrieved:
            retrieved[node_type].append(item)
    retrieved["text"] = retrieved["text"][: int(budget_config.get("max_text_nodes", 6))]
    retrieved["entity"] = retrieved["entity"][: int(budget_config.get("max_entity_nodes", 12))]
    media_limit = int(budget_config.get("max_media_nodes", 6))
    table_limit = int(budget_config.get("max_table_nodes", 4))
    media_items = retrieved["media"]
    tables = [item for item in media_items if (item.get("metadata") or {}).get("media_type") == "table"][:table_limit]
    non_tables = [item for item in media_items if (item.get("metadata") or {}).get("media_type") != "table"]
    retrieved["media"] = (tables + non_tables)[:media_limit]
    retrieved["page"] = retrieved["page"][: int(budget_config.get("max_page_nodes", 4))]

    direct_recall = {
        "text": direct.get("text", []),
        "entity": direct.get("entity", []),
        "media": [_slim_media_candidate(item) for item in direct.get("media", [])],
        "page": direct.get("page", []),
    }
    has_direct = any(direct_recall.get(key) for key in ("text", "entity", "media", "page"))
    return {
        "query_info": query_info,
        "retrieved_nodes_by_type": retrieved,
        "direct_recall": direct_recall,
        "merged_candidates": merged,
        "failure_stage": None if has_direct else "direct_recall_empty",
    }


def _node_lexical_search(
    nodes: list[dict[str, Any]],
    query: str,
    node_type: str,
    topk: int,
    retriever: str,
) -> list[dict[str, Any]]:
    scoped = [node for node in nodes if node.get("node_type") == node_type]
    doc_freq = _doc_freq(scoped)
    terms = _terms(query)
    scored = []
    for node in scoped:
        score = _bm25_like_score(terms, node.get("text_for_embedding", ""), doc_freq, len(scoped))
        if score > 0:
            scored.append((score, node))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_candidate(node, score, retriever, rank) for rank, (score, node) in enumerate(scored[:topk], start=1)]


def _candidate(node: dict[str, Any], score: float, retriever: str, rank: int) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "doc_id": node.get("doc_id"),
        "page_id": node.get("page_id"),
        "score": float(score),
        "retriever": retriever,
        "rank": rank,
        "source": "direct_recall",
        "raw_ref": node.get("raw_ref") or {},
        "metadata": node.get("metadata") or {},
    }


def _apply_rule_bonuses(
    candidates: list[dict[str, Any]],
    query_info: dict[str, Any],
    fusion_config: dict[str, Any],
) -> list[dict[str, Any]]:
    page_hints = {hint.get("page") for hint in query_info.get("page_hints", []) if hint.get("page")}
    media_hints = set(query_info.get("media_hints", []))
    page_bonus = float(fusion_config.get("page_hint_bonus", 0.20))
    modality_bonus = float(fusion_config.get("modality_bonus", 0.15))
    for item in candidates:
        debug = item.setdefault("debug", {})
        bonuses = {}
        if item.get("page_id") in page_hints:
            item["score"] += page_bonus
            bonuses["page_hint_bonus"] = page_bonus
        media_type = str((item.get("metadata") or {}).get("media_type") or "").lower()
        if item.get("node_type") == "media":
            if media_type == "table" and media_hints.intersection({"table", "row", "column", "cell", "header", "metric", "f1"}):
                item["score"] += modality_bonus
                bonuses["modality_bonus"] = modality_bonus
            elif media_type in {"chart", "image", "figure"} and media_hints:
                item["score"] += modality_bonus
                bonuses["modality_bonus"] = modality_bonus
        if bonuses:
            debug["rule_bonuses"] = bonuses
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _dynamic_topk(base_topk: int, query_info: dict[str, Any], key: str) -> int:
    prior = query_info.get("modality_prior", {}).get(key, 0.2)
    if key == "media":
        prior = max(prior, query_info.get("modality_prior", {}).get("table", 0.0))
    return max(1, int(round(base_topk * (0.75 + prior))))


def _section(config: dict[str, Any], name: str, defaults: dict[str, Any]) -> dict[str, Any]:
    section = dict(defaults)
    value = config.get(name)
    if isinstance(value, dict):
        section.update(value)
    return section


def _empty_trace(failure_stage: str, reason: str) -> dict[str, Any]:
    return {
        "query_info": {},
        "retrieved_nodes_by_type": {"text": [], "entity": [], "media": [], "page": [], "aggregate": []},
        "direct_recall": {"text": [], "entity": [], "media": [], "page": []},
        "merged_candidates": [],
        "failure_stage": failure_stage,
        "debug": {"reason": reason},
    }


def _slim_media_candidate(item: dict[str, Any]) -> dict[str, Any]:
    raw_ref = item.get("raw_ref") or {}
    return {
        "node_id": item.get("node_id"),
        "node_type": item.get("node_type"),
        "doc_id": item.get("doc_id"),
        "page_id": item.get("page_id"),
        "score": item.get("score"),
        "retriever": item.get("retriever"),
        "rank": item.get("rank"),
        "source": item.get("source"),
        "raw_ref": {
            "media_id": raw_ref.get("media_id"),
            "path": raw_ref.get("path"),
            "table_markdown": raw_ref.get("table_markdown"),
            "table_html": raw_ref.get("table_html"),
        },
        "metadata": {"media_type": (item.get("metadata") or {}).get("media_type")},
    }


def _doc_freq(nodes: list[dict[str, Any]]) -> Counter[str]:
    freq = Counter()
    for node in nodes:
        freq.update(set(_terms(node.get("text_for_embedding", ""))))
    return freq


def _bm25_like_score(query_terms: list[str], text: str, doc_freq: Counter[str], total_docs: int) -> float:
    if not query_terms:
        return 0.0
    counts = Counter(_terms(text))
    score = 0.0
    for term in query_terms:
        tf = counts.get(term, 0)
        if tf <= 0:
            continue
        idf = math.log((total_docs + 1) / (1 + doc_freq.get(term, 0))) + 1.0
        score += (1.0 + math.log(tf)) * idf
    return score / max(1.0, len(query_terms))


def _terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9%/-]+", str(text).lower()) if len(term) > 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 multimodal direct recall over mm_nodes.jsonl.")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("--topk_media", type=int, default=None)
    parser.add_argument("--topk_page", type=int, default=None)
    args = parser.parse_args()
    nodes_path = Path(args.nodes)
    retrieval = dict(DEFAULT_RETRIEVAL)
    if args.topk_media is not None:
        retrieval["topk_media"] = args.topk_media
    if args.topk_page is not None:
        retrieval["topk_page"] = args.topk_page
    _, _, trace = mm_hybrid_retrieve(
        args.query,
        {"working_dir": str(nodes_path.parent), "nodes_file": str(nodes_path), "retrieval": retrieval},
        doc_id=args.doc_id,
    )
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
