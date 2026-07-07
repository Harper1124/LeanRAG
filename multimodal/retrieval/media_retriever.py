from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from ..io_utils import read_jsonl
from .media_ref import matching_media_refs


class MediaRetriever:
    def __init__(self, nodes_path: str | Path, doc_id: str | None = None):
        self.nodes_path = Path(nodes_path)
        self.nodes = [
            node
            for node in read_jsonl(self.nodes_path)
            if node.get("node_type") == "media" and (doc_id is None or node.get("doc_id") == doc_id)
        ]
        self._doc_freq = _doc_freq(self.nodes)

    def search(self, query: str, query_info: dict[str, Any] | None = None, topk: int = 8) -> list[dict[str, Any]]:
        query_info = query_info or {}
        query_terms = _terms(query)
        page_hints = {hint.get("page") for hint in query_info.get("page_hints", []) if hint.get("page")}
        media_hints = set(query_info.get("media_hints", []))
        media_refs = query_info.get("media_refs", [])
        scored = []
        for node in self.nodes:
            matched_refs = matching_media_refs(node, media_refs)
            if media_refs and not matched_refs:
                continue
            score = _bm25_like_score(query_terms, node.get("text_for_embedding", ""), self._doc_freq, len(self.nodes))
            media_type = str((node.get("metadata") or {}).get("media_type") or "unknown").lower()
            if matched_refs:
                score += 5.0 + (0.25 * len(matched_refs))
            if node.get("page_id") in page_hints:
                score += 0.35
            if media_type == "table" and media_hints.intersection({"table", "row", "column", "cell", "header", "metric", "f1"}):
                score += 0.30
            if media_type == "chart" and media_hints.intersection({"chart", "graph", "axis", "line", "bar", "pie", "trend"}):
                score += 0.30
            if media_type in {"image", "figure", "chart"} and media_hints.intersection({"figure", "image", "picture", "photo", "icon", "color", "shape", "map"}):
                score += 0.20
            if query_terms and score <= 0:
                continue
            scored.append((score, node, matched_refs))

        if not scored and self.nodes and not media_refs:
            scored = [(0.01, node, []) for node in self.nodes]
        scored.sort(key=lambda item: (item[0], -(item[1].get("page_id") or 0)), reverse=True)
        return [_candidate(node, score, rank, matched_refs) for rank, (score, node, matched_refs) in enumerate(scored[:topk], start=1)]


def _candidate(node: dict[str, Any], score: float, rank: int, matched_refs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": "media",
        "doc_id": node.get("doc_id"),
        "page_id": node.get("page_id"),
        "score": float(score),
        "retriever": "media_retriever",
        "rank": rank,
        "source": "direct_recall",
        "raw_ref": node.get("raw_ref") or {},
        "metadata": node.get("metadata") or {},
        "text_for_embedding": node.get("text_for_embedding", ""),
        "caption": node.get("caption", ""),
        "ocr_text": node.get("ocr_text", ""),
        "summary": node.get("summary", ""),
        "debug": {"matched_media_refs": matched_refs or []},
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
    if not counts:
        return 0.0
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
