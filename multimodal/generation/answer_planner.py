from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ..retrieval.mm_retriever import mm_hybrid_retrieve
from .evidence_package import DEFAULT_BUDGET, build_evidence_package


DEFAULT_GENERATION = {
    "use_vlm": True,
    "answer_with_vlm_when_media": True,
    "use_table_reasoner": True,
    "answer_not_enough_evidence": "Not answerable",
    "max_prompt_chars": 24000,
}


def plan_answer(
    question: str,
    evidence_package: dict[str, list[dict[str, Any]]],
    query_info: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_info = query_info or {}
    generation = _generation_config(config)
    budget = _budget(config)
    visual_nodes = evidence_package.get("visual_evidence", [])[: budget["max_vlm_images"]]
    table_nodes = evidence_package.get("table_evidence", [])[: budget["max_table_nodes"]]
    text_nodes = evidence_package.get("text_evidence", [])[: budget["max_text_nodes"]]
    entity_nodes = evidence_package.get("entity_evidence", [])[: budget["max_entity_nodes"]]
    page_nodes = evidence_package.get("page_evidence", [])[: budget["max_page_nodes"]]

    has_text_like = bool(text_nodes or entity_nodes or page_nodes)
    has_visual = bool(visual_nodes)
    has_table = bool(table_nodes)
    if not (has_text_like or has_visual or has_table):
        answer_mode = "not_enough_evidence"
    elif has_visual and has_table:
        answer_mode = "visual_table"
    elif has_visual:
        answer_mode = "visual"
    elif has_table:
        answer_mode = "table"
    elif has_text_like:
        answer_mode = "text_only"
    else:
        answer_mode = "mixed"

    use_vlm = bool(generation["use_vlm"] and generation["answer_with_vlm_when_media"] and has_visual)
    use_table_reasoner = bool(generation["use_table_reasoner"] and has_table and _has_table_intent(question, query_info))
    return {
        "use_vlm": use_vlm,
        "use_table_reasoner": use_table_reasoner,
        "use_text_llm": answer_mode != "not_enough_evidence",
        "answer_mode": answer_mode,
        "expected_answer_type": query_info.get("expected_answer_type", "unknown"),
        "selected_visual_nodes": visual_nodes,
        "selected_table_nodes": table_nodes,
        "selected_text_nodes": text_nodes,
        "selected_entity_nodes": entity_nodes,
        "selected_page_nodes": page_nodes,
    }


def _has_table_intent(question: str, query_info: dict[str, Any]) -> bool:
    text = question.lower()
    for ref in query_info.get("media_refs") or []:
        if str(ref.get("kind") or "").lower() == "table":
            return True
    if str(query_info.get("query_type") or "").lower() == "table":
        return True
    if re.search(r"\b(table|tab\.|row|column|cell)\b", text):
        return True
    return bool(re.search(r"\bclaims?\b", text) and re.search(r"\b(dataset|datasets|scientific articles|newspaper|wiki)\b", text))


def _generation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    value = dict(DEFAULT_GENERATION)
    if isinstance(config, dict):
        for key in DEFAULT_GENERATION:
            if key in config:
                value[key] = config[key]
        if isinstance(config.get("generation"), dict):
            value.update(config["generation"])
    return value


def _budget(config: dict[str, Any] | None) -> dict[str, int]:
    value = dict(DEFAULT_BUDGET)
    if isinstance(config, dict) and isinstance(config.get("evidence_budget"), dict):
        value.update(config["evidence_budget"])
    return {key: int(val) for key, val in value.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan Phase 3 multimodal answer pipeline for a query.")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--doc_id", default=None)
    args = parser.parse_args()
    nodes = Path(args.nodes)
    merged, query_info, _ = mm_hybrid_retrieve(
        args.query,
        {"working_dir": str(nodes.parent), "nodes_file": str(nodes)},
        doc_id=args.doc_id,
    )
    evidence_package = build_evidence_package(merged, query_info, {})
    plan = plan_answer(args.query, evidence_package, query_info, {})
    print(json.dumps({"query_info": query_info, "evidence_package": evidence_package, "answer_plan": plan}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
