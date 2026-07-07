from __future__ import annotations

import json
from typing import Any, Callable


FINAL_PROMPT = """Question:
{question}

Text Evidence:
{text_evidence}

Entity Evidence:
{entity_evidence}

Page Evidence:
{page_evidence}

Visual Evidence:
{visual_evidence}

Table Evidence:
{table_evidence}

Instruction:
Answer using only the evidence above.
If the evidence is insufficient, answer "Not answerable".
For numeric/list questions, answer concisely.
"""


def generate_final_answer(
    question: str,
    evidence_package: dict[str, list[dict[str, Any]]],
    answer_plan: dict[str, Any],
    vlm_calls: list[dict[str, Any]],
    table_reasoner_calls: list[dict[str, Any]],
    global_config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    generation = _generation_config(global_config)
    if answer_plan.get("answer_mode") == "not_enough_evidence":
        return generation["answer_not_enough_evidence"], {"prompt_preview": "", "used_llm": False, "error": None}
    prompt = _build_prompt(question, evidence_package, vlm_calls, table_reasoner_calls, int(generation["max_prompt_chars"]))
    llm_func = global_config.get("use_llm_func")
    if llm_func:
        try:
            return str(llm_func(prompt)), {"prompt_preview": prompt[:1000], "used_llm": True, "error": None}
        except TypeError:
            try:
                return str(llm_func(question, system_prompt=prompt)), {"prompt_preview": prompt[:1000], "used_llm": True, "error": None}
            except Exception as exc:
                return _fallback_answer(vlm_calls, table_reasoner_calls, generation), {
                    "prompt_preview": prompt[:1000],
                    "used_llm": False,
                    "error": str(exc),
                }
        except Exception as exc:
            return _fallback_answer(vlm_calls, table_reasoner_calls, generation), {
                "prompt_preview": prompt[:1000],
                "used_llm": False,
                "error": str(exc),
            }
    return _fallback_answer(vlm_calls, table_reasoner_calls, generation), {
        "prompt_preview": prompt[:1000],
        "used_llm": False,
        "error": "llm_func_missing",
    }


def _build_prompt(
    question: str,
    evidence_package: dict[str, list[dict[str, Any]]],
    vlm_calls: list[dict[str, Any]],
    table_reasoner_calls: list[dict[str, Any]],
    max_chars: int,
) -> str:
    text = FINAL_PROMPT.format(
        question=question,
        text_evidence=_dump_nodes(evidence_package.get("text_evidence", []), include_raw=False),
        entity_evidence=_dump_nodes(evidence_package.get("entity_evidence", []), include_raw=False),
        page_evidence=_dump_nodes(evidence_package.get("page_evidence", []), include_raw=False),
        visual_evidence=_visual_text(evidence_package.get("visual_evidence", []), vlm_calls),
        table_evidence=_table_text(evidence_package.get("table_evidence", []), table_reasoner_calls),
    )
    return text[:max_chars]


def _dump_nodes(nodes: list[dict[str, Any]], include_raw: bool) -> str:
    rows = []
    for node in nodes:
        item = {
            "node_id": node.get("node_id"),
            "page_id": node.get("page_id"),
            "score": node.get("score"),
            "text": node.get("text_for_embedding") or node.get("metadata", {}).get("text") or node.get("raw_ref", {}),
        }
        if include_raw:
            item["raw_ref"] = node.get("raw_ref", {})
        rows.append(item)
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _visual_text(nodes: list[dict[str, Any]], vlm_calls: list[dict[str, Any]]) -> str:
    vlm_by_node = {}
    for call in vlm_calls:
        for node_id in call.get("node_ids", []):
            vlm_by_node[node_id] = {"vlm_answer": call.get("vlm_answer"), "error": call.get("error")}
    rows = []
    for node in nodes:
        raw_ref = node.get("raw_ref") or {}
        rows.append(
            {
                "node_id": node.get("node_id"),
                "page": node.get("page_id"),
                "media_id": raw_ref.get("media_id"),
                "path": raw_ref.get("path"),
                "metadata": node.get("metadata"),
                "vlm_result": vlm_by_node.get(node.get("node_id")),
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _table_text(nodes: list[dict[str, Any]], table_calls: list[dict[str, Any]]) -> str:
    table_by_node = {}
    for call in table_calls:
        for node_id in call.get("used_table_nodes", []):
            table_by_node[node_id] = {"table_answer": call.get("table_answer"), "error": call.get("error")}
    rows = []
    for node in nodes:
        raw_ref = node.get("raw_ref") or {}
        rows.append(
            {
                "node_id": node.get("node_id"),
                "page": node.get("page_id"),
                "media_id": raw_ref.get("media_id"),
                "table_markdown": raw_ref.get("table_markdown"),
                "table_html": raw_ref.get("table_html"),
                "table_reasoner_result": table_by_node.get(node.get("node_id")),
            }
        )
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _fallback_answer(
    vlm_calls: list[dict[str, Any]],
    table_reasoner_calls: list[dict[str, Any]],
    generation: dict[str, Any],
) -> str:
    for call in table_reasoner_calls:
        if call.get("table_answer"):
            return str(call["table_answer"])
    for call in vlm_calls:
        if call.get("vlm_answer"):
            return str(call["vlm_answer"])
    return str(generation["answer_not_enough_evidence"])


def _generation_config(config: dict[str, Any]) -> dict[str, Any]:
    value = {
        "answer_not_enough_evidence": "Not answerable",
        "max_prompt_chars": 24000,
    }
    for key in list(value):
        if key in config:
            value[key] = config[key]
    if isinstance(config.get("generation"), dict):
        value.update(config["generation"])
    return value
