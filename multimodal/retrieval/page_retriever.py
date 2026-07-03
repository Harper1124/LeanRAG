from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..io_utils import read_jsonl


class PageRetriever:
    def __init__(self, nodes_path: str | Path, doc_id: str | None = None):
        self.nodes_path = Path(nodes_path)
        self.nodes = [
            node
            for node in read_jsonl(self.nodes_path)
            if node.get("node_type") == "page" and (doc_id is None or node.get("doc_id") == doc_id)
        ]
        self.by_page = {node.get("page_id"): node for node in self.nodes}

    def search(
        self,
        query: str,
        query_info: dict[str, Any] | None = None,
        seed_candidates: list[dict[str, Any]] | None = None,
        topk: int = 4,
    ) -> list[dict[str, Any]]:
        del query
        query_info = query_info or {}
        candidates = []
        matched_pages = set()
        for hint in query_info.get("page_hints", []):
            raw_page = hint.get("page")
            matched = self._match_page_hint(raw_page)
            if not matched:
                continue
            page_id, node, note = matched
            matched_pages.add(page_id)
            candidates.append(_candidate(node, 1.0, 0, hint, note))

        if seed_candidates:
            page_scores = defaultdict(float)
            for item in seed_candidates:
                page_id = item.get("page_id")
                if page_id in self.by_page:
                    page_scores[page_id] += float(item.get("score") or 0.0)
            for page_id, score in page_scores.items():
                if page_id in matched_pages:
                    continue
                candidates.append(_candidate(self.by_page[page_id], min(0.95, score), 0, None, "aggregated_from_candidates"))

        candidates.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(candidates[:topk], start=1):
            item["rank"] = rank
        return candidates[:topk]

    def _match_page_hint(self, raw_page: int | None):
        if raw_page is None:
            return None
        # User-facing page numbers are normally 1-based. As a defensive fallback,
        # also try raw+1 for data that still carries 0-based page hints.
        if raw_page in self.by_page:
            return raw_page, self.by_page[raw_page], "matched_as_1_based"
        if raw_page + 1 in self.by_page:
            return raw_page + 1, self.by_page[raw_page + 1], "matched_as_0_based_plus_1"
        if raw_page - 1 in self.by_page:
            return raw_page - 1, self.by_page[raw_page - 1], "matched_minus_1_fallback"
        return None


def _candidate(
    node: dict[str, Any],
    score: float,
    rank: int,
    page_hint: dict[str, Any] | None,
    match_note: str,
) -> dict[str, Any]:
    metadata = dict(node.get("metadata") or {})
    metadata["page_hint_match"] = {"hint": page_hint, "matched_page_id": node.get("page_id"), "note": match_note}
    return {
        "node_id": node.get("node_id"),
        "node_type": "page",
        "doc_id": node.get("doc_id"),
        "page_id": node.get("page_id"),
        "score": float(score),
        "retriever": "page_retriever",
        "rank": rank,
        "source": "direct_recall",
        "raw_ref": node.get("raw_ref") or {},
        "metadata": metadata,
    }
