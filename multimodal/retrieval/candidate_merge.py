from __future__ import annotations

from typing import Any


OPTIONAL_FIELDS = ("text_for_embedding", "caption", "ocr_text", "summary")


def merge_candidates(candidates: list[dict[str, Any]], multi_hit_bonus: float = 0.10) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in candidates:
        node_id = item.get("node_id")
        if not node_id:
            continue
        retrievers = item.get("retrievers") or ([item.get("retriever")] if item.get("retriever") else ["unknown"])
        retrievers = [retriever for retriever in retrievers if retriever]
        retriever = retrievers[0] if retrievers else "unknown"
        score = float(item.get("score") or 0.0)
        if node_id not in merged:
            debug = {
                "original_scores": {name: score for name in retrievers},
                "ranks": {name: item.get("rank") for name in retrievers},
            }
            if isinstance(item.get("debug"), dict):
                debug.update(item["debug"])
            if item.get("source") == "graph_expansion":
                debug["expansion_paths"] = [_expansion_path(item)]
            merged[node_id] = {
                "node_id": node_id,
                "node_type": item.get("node_type"),
                "doc_id": item.get("doc_id"),
                "page_id": item.get("page_id"),
                "score": score,
                "retrievers": list(retrievers),
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
        for retriever in retrievers:
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
        if item.get("source") == "graph_expansion":
            current["debug"].setdefault("expansion_paths", []).append(_expansion_path(item))
        if isinstance(item.get("debug"), dict):
            for key, value in item["debug"].items():
                if key not in {"original_scores", "ranks"}:
                    current["debug"][key] = value

    for item in merged.values():
        original_scores = item["debug"]["original_scores"]
        bonus = multi_hit_bonus if len(original_scores) > 1 else 0.0
        item["score"] = max(original_scores.values()) + bonus
        item["debug"]["multi_hit_bonus"] = bonus
        sources = {
            source
            for source in str(item.get("source", "")).split("+")
            if source
        }
        if "entity_source_resolver" in item["retrievers"]:
            sources.add("entity_source_id")
        if "graph_expansion" in item["retrievers"]:
            sources.add("graph_expansion")
        if any(name not in {"graph_expansion", "entity_source_resolver"} for name in item["retrievers"]):
            sources.add("direct_recall")
        item["source"] = "+".join(
            source for source in ("direct_recall", "entity_source_id", "graph_expansion") if source in sources
        ) or item.get("source", "")
    return sorted(merged.values(), key=lambda item: item["score"], reverse=True)


def _expansion_path(item: dict[str, Any]) -> dict[str, Any]:
    debug = item.get("debug") or {}
    return {
        "from_anchor": debug.get("from_anchor"),
        "edge_type": debug.get("edge_type"),
        "hop": debug.get("hop"),
        "edge_id": debug.get("edge_id"),
        "edge_weight": debug.get("edge_weight"),
    }
