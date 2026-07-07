from __future__ import annotations

from typing import Any


OPTIONAL_FIELDS = ("text_for_embedding", "caption", "ocr_text", "summary")


def merge_candidates(candidates: list[dict[str, Any]], multi_hit_bonus: float = 0.10) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in candidates:
        node_id = item.get("node_id")
        if not node_id:
            continue
        retriever = item.get("retriever", "unknown")
        score = float(item.get("score") or 0.0)
        if node_id not in merged:
            debug = {"original_scores": {retriever: score}, "ranks": {retriever: item.get("rank")}}
            if isinstance(item.get("debug"), dict):
                debug.update(item["debug"])
            merged[node_id] = {
                "node_id": node_id,
                "node_type": item.get("node_type"),
                "doc_id": item.get("doc_id"),
                "page_id": item.get("page_id"),
                "score": score,
                "retrievers": [retriever],
                "source": item.get("source", "direct_recall"),
                "raw_ref": item.get("raw_ref") or {},
                "metadata": item.get("metadata") or {},
                "debug": debug,
            }
            for field in OPTIONAL_FIELDS:
                if item.get(field):
                    merged[node_id][field] = item.get(field)
            continue
        current = merged[node_id]
        if retriever not in current["retrievers"]:
            current["retrievers"].append(retriever)
        current["debug"]["original_scores"][retriever] = max(
            score,
            float(current["debug"]["original_scores"].get(retriever, 0.0)),
        )
        current["debug"]["ranks"][retriever] = item.get("rank")
        if score > float(current["score"]):
            current["score"] = score
            current["metadata"] = item.get("metadata") or current["metadata"]
            current["raw_ref"] = item.get("raw_ref") or current["raw_ref"]
            for field in OPTIONAL_FIELDS:
                if item.get(field):
                    current[field] = item.get(field)
        if isinstance(item.get("debug"), dict):
            for key, value in item["debug"].items():
                if key not in {"original_scores", "ranks"}:
                    current["debug"][key] = value

    for item in merged.values():
        original_scores = item["debug"]["original_scores"]
        bonus = multi_hit_bonus if len(original_scores) > 1 else 0.0
        item["score"] = max(original_scores.values()) + bonus
        item["debug"]["multi_hit_bonus"] = bonus
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)
