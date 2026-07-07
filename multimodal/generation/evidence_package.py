from __future__ import annotations

from typing import Any


DEFAULT_BUDGET = {
    "max_text_nodes": 6,
    "max_entity_nodes": 12,
    "max_media_nodes": 6,
    "max_table_nodes": 4,
    "max_page_nodes": 4,
    "max_vlm_images": 4,
}
VISUAL_TYPES = {"image", "chart", "figure", "screenshot", "photo", "unknown"}


def build_evidence_package(
    merged_candidates: list[dict[str, Any]],
    query_info: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    del query_info
    budget = _budget(config)
    package = {
        "text_evidence": [],
        "entity_evidence": [],
        "page_evidence": [],
        "visual_evidence": [],
        "table_evidence": [],
        "all_selected_nodes": [],
    }
    seen_selected = set()
    for candidate in sorted(merged_candidates or [], key=lambda item: float(item.get("score") or 0.0), reverse=True):
        node_type = candidate.get("node_type")
        slim = _slim_candidate(candidate)
        added = False
        if node_type == "text" and len(package["text_evidence"]) < budget["max_text_nodes"]:
            package["text_evidence"].append(slim)
            added = True
        elif node_type == "entity" and len(package["entity_evidence"]) < budget["max_entity_nodes"]:
            package["entity_evidence"].append(slim)
            added = True
        elif node_type == "page" and len(package["page_evidence"]) < budget["max_page_nodes"]:
            package["page_evidence"].append(slim)
            added = True
        elif node_type == "media":
            is_table = _is_table(candidate)
            is_visual = _is_visual(candidate)
            if is_table and len(package["table_evidence"]) < budget["max_table_nodes"]:
                package["table_evidence"].append(slim)
                added = True
            if is_visual and len(package["visual_evidence"]) < budget["max_media_nodes"]:
                package["visual_evidence"].append(slim)
                added = True
        if added and slim.get("node_id") not in seen_selected:
            seen_selected.add(slim.get("node_id"))
            package["all_selected_nodes"].append(slim)
    return package


def _is_table(candidate: dict[str, Any]) -> bool:
    raw_ref = candidate.get("raw_ref") or {}
    media_type = str((candidate.get("metadata") or {}).get("media_type") or "").lower()
    return media_type == "table" or bool(raw_ref.get("table_markdown") or raw_ref.get("table_html"))


def _is_visual(candidate: dict[str, Any]) -> bool:
    raw_ref = candidate.get("raw_ref") or {}
    media_type = str((candidate.get("metadata") or {}).get("media_type") or "unknown").lower()
    return media_type in VISUAL_TYPES and bool(raw_ref.get("path") or media_type != "unknown")


def _slim_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": candidate.get("node_id"),
        "node_type": candidate.get("node_type"),
        "doc_id": candidate.get("doc_id"),
        "page_id": candidate.get("page_id"),
        "score": float(candidate.get("score") or 0.0),
        "retrievers": candidate.get("retrievers") or ([candidate.get("retriever")] if candidate.get("retriever") else []),
        "source": candidate.get("source", "direct_recall"),
        "raw_ref": candidate.get("raw_ref") or {},
        "metadata": candidate.get("metadata") or {},
        "debug": candidate.get("debug") or {},
    }


def _budget(config: dict[str, Any] | None) -> dict[str, int]:
    value = dict(DEFAULT_BUDGET)
    if isinstance(config, dict):
        section = config.get("evidence_budget", {})
        if isinstance(section, dict):
            value.update(section)
    return {key: int(val) for key, val in value.items()}
