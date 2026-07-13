from __future__ import annotations

import json
import re
from typing import Any, Callable

from multimodal.retrieval.media_ref import extract_media_refs
from .table_utils import has_structured_table


FINAL_PROMPT = """Question:
{question}

Expected Answer Type:
{answer_type}

Text Evidence:
{text_evidence}

Entity Evidence:
{entity_evidence}

Page Evidence:
{page_evidence}

Aggregation Context:
{aggregation_context}

Visual Evidence:
{visual_evidence}

Table Evidence:
{table_evidence}

Instruction:
Answer using only the evidence above.
Before answering, silently check whether the evidence directly supports the requested value.
If the evidence is insufficient or only related to a nearby topic, answer exactly: "Not answerable".
Return only the final answer, with no explanation, citations, markdown, or reasoning.
For integer questions, return only an integer such as 3.
For float questions, return only the number or percent.
For list questions, return only a comma-separated list of answer items.
For string questions, return only the shortest answer span.
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
    if (
        _target_answer_type(question, answer_plan, global_config) == "unknown"
        and _has_explicit_unknown_format(global_config)
        and _use_answer_format_oracle_abstention(global_config)
    ):
        return generation["answer_not_enough_evidence"], {
            "prompt_preview": "",
            "used_llm": False,
            "error": None,
            "raw_answer": "",
            "postprocess": {"target": "unknown", "changed": True, "rule": "answer_format_unknown"},
        }
    if answer_plan.get("answer_mode") == "not_enough_evidence":
        return generation["answer_not_enough_evidence"], {"prompt_preview": "", "used_llm": False, "error": None}
    answer_type = _target_answer_type(question, answer_plan, global_config)
    sufficiency = _evidence_sufficiency(question, evidence_package, answer_type)
    if not sufficiency["sufficient"]:
        return generation["answer_not_enough_evidence"], {
            "prompt_preview": "",
            "used_llm": False,
            "error": None,
            "raw_answer": "",
            "evidence_sufficiency": sufficiency,
            "postprocess": {"target": answer_type, "changed": True, "rule": "evidence_insufficient"},
        }
    prompt = _build_prompt(
        question,
        evidence_package,
        vlm_calls,
        table_reasoner_calls,
        int(generation["max_prompt_chars"]),
        answer_type,
    )
    llm_func = global_config.get("use_llm_func")
    if llm_func:
        try:
            answer = str(llm_func(prompt))
            return _with_postprocess(answer, question, answer_plan, global_config, prompt, True, None, sufficiency)
        except TypeError:
            try:
                answer = str(llm_func(question, system_prompt=prompt))
                return _with_postprocess(answer, question, answer_plan, global_config, prompt, True, None, sufficiency)
            except Exception as exc:
                answer = _fallback_answer(vlm_calls, table_reasoner_calls, generation)
                return _with_postprocess(answer, question, answer_plan, global_config, prompt, False, str(exc), sufficiency)
        except Exception as exc:
            answer = _fallback_answer(vlm_calls, table_reasoner_calls, generation)
            return _with_postprocess(answer, question, answer_plan, global_config, prompt, False, str(exc), sufficiency)
    answer = _fallback_answer(vlm_calls, table_reasoner_calls, generation)
    return _with_postprocess(answer, question, answer_plan, global_config, prompt, False, "llm_func_missing", sufficiency)


def _with_postprocess(
    raw_answer: str,
    question: str,
    answer_plan: dict[str, Any],
    global_config: dict[str, Any],
    prompt: str,
    used_llm: bool,
    error: str | None,
    sufficiency: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    processed, postprocess = postprocess_final_answer(raw_answer, question, answer_plan, global_config)
    return processed, {
        "prompt_preview": prompt[:1000],
        "used_llm": used_llm,
        "error": error,
        "raw_answer": raw_answer,
        "evidence_sufficiency": sufficiency or {},
        "postprocess": postprocess,
    }


def postprocess_final_answer(
    answer: str,
    question: str = "",
    answer_plan: dict[str, Any] | None = None,
    global_config: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    raw = str(answer or "").strip()
    answer_plan = answer_plan or {}
    global_config = global_config or {}
    target = _target_answer_type(question, answer_plan, global_config)
    not_answerable = str(_generation_config(global_config)["answer_not_enough_evidence"])
    lowered = raw.lower()

    if target == "unknown" and _has_explicit_unknown_format(global_config) and _use_answer_format_oracle_abstention(global_config):
        return not_answerable, {"target": target, "changed": raw != not_answerable, "rule": "answer_format_unknown"}
    if _looks_not_answerable(lowered):
        return not_answerable, {"target": target, "changed": raw != not_answerable, "rule": "not_answerable"}
    if target == "int":
        value = _extract_int(raw)
        if value is not None:
            return value, {"target": target, "changed": value != raw, "rule": "extract_int"}
    if target == "float":
        value = _extract_float(raw)
        if value is not None:
            return value, {"target": target, "changed": value != raw, "rule": "extract_float"}
    if target == "list":
        value = _extract_list(raw)
        if value is not None:
            return value, {"target": target, "changed": value != raw, "rule": "extract_list"}
    return raw, {"target": target, "changed": False, "rule": "none"}


def _target_answer_type(question: str, answer_plan: dict[str, Any], global_config: dict[str, Any]) -> str:
    answer_format = str(global_config.get("answer_format") or "").strip().lower()
    if answer_format in {"int", "integer"}:
        return "int"
    if answer_format in {"float", "number"}:
        return "float"
    if answer_format in {"list", "array"}:
        return "list"
    if answer_format in {"str", "string"}:
        return "str"
    if answer_format in {"unknown", "none", "null", "not answerable", "unanswerable"}:
        return "unknown"

    expected = str(answer_plan.get("expected_answer_type") or "").strip().lower()
    if expected in {"int", "float", "list"}:
        return expected
    text = question.lower()
    if re.search(r"\b(how many|number of|count)\b", text):
        return "int"
    if re.search(r"\b(percentage|ratio|float|average|gap|score|rate|difference)\b", text):
        return "float"
    if re.search(r"\b(list|what are|which two|list all|name all)\b", text):
        return "list"
    return "unknown"


def _has_explicit_unknown_format(global_config: dict[str, Any]) -> bool:
    answer_format = str(global_config.get("answer_format") or "").strip().lower()
    return answer_format in {"unknown", "none", "null", "not answerable", "unanswerable"}


def _use_answer_format_oracle_abstention(global_config: dict[str, Any]) -> bool:
    if "use_answer_format_oracle_abstention" in global_config:
        return bool(global_config["use_answer_format_oracle_abstention"])
    generation = global_config.get("generation")
    if isinstance(generation, dict) and "use_answer_format_oracle_abstention" in generation:
        return bool(generation["use_answer_format_oracle_abstention"])
    return False


def _evidence_sufficiency(question: str, evidence_package: dict[str, Any], answer_type: str) -> dict[str, Any]:
    refs = extract_media_refs(question)
    pages = _question_pages(question)
    table_nodes = evidence_package.get("table_evidence", []) or []
    visual_nodes = evidence_package.get("visual_evidence", []) or []
    text_nodes = evidence_package.get("text_evidence", []) or []
    page_nodes = evidence_package.get("page_evidence", []) or []
    all_nodes = list(text_nodes) + list(evidence_package.get("entity_evidence", []) or []) + list(page_nodes) + list(visual_nodes) + list(table_nodes)
    reasons = []

    if refs and not _has_matched_ref(refs, visual_nodes + table_nodes):
        reasons.append("explicit_media_ref_not_grounded")
    if pages and not _has_evidence_on_pages(pages, all_nodes):
        reasons.append("page_hint_not_grounded")
    if _question_needs_table(question) and not any(has_structured_table(node) for node in table_nodes):
        reasons.append("structured_table_missing")
    if answer_type == "unknown" and not _has_keyword_support(question, all_nodes):
        reasons.append("question_object_not_supported")

    return {
        "sufficient": not reasons,
        "reasons": reasons,
        "media_refs": refs,
        "page_hints": sorted(pages),
        "table_nodes": len(table_nodes),
        "structured_table_nodes": sum(1 for node in table_nodes if has_structured_table(node)),
    }


def _has_matched_ref(refs: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> bool:
    wanted = {(str(ref.get("kind")), str(ref.get("number"))) for ref in refs}
    matched = set()
    for node in nodes:
        for ref in ((node.get("debug") or {}).get("matched_media_refs") or []):
            matched.add((str(ref.get("kind")), str(ref.get("number"))))
    return bool(wanted and wanted.issubset(matched))


def _question_pages(question: str) -> set[int]:
    text = question.lower()
    pages = {int(match) for match in re.findall(r"\bpages?\s*(\d+)\b", text)}
    for start, end in re.findall(r"\bpages?\s*(\d+)\s*[-–]\s*(\d+)\b", text):
        left, right = int(start), int(end)
        if right < left:
            left, right = right, left
        if right - left <= 30:
            pages.update(range(left, right + 1))
    return pages


def _has_evidence_on_pages(pages: set[int], nodes: list[dict[str, Any]]) -> bool:
    evidence_pages = set()
    for node in nodes:
        for key in ("page_id", "page", "page_start", "page_end"):
            if node.get(key) is not None:
                try:
                    evidence_pages.add(int(node[key]))
                except (TypeError, ValueError):
                    pass
    return bool(pages & evidence_pages)


def _question_needs_table(question: str) -> bool:
    text = question.lower()
    if re.search(r"\b(table|row|column|cell)\b", text):
        return True
    return bool(re.search(r"\bclaims?\b", text) and re.search(r"\b(dataset|datasets|scientific articles|newspaper|wiki)\b", text))


def _has_keyword_support(question: str, nodes: list[dict[str, Any]]) -> bool:
    terms = [
        term
        for term in re.findall(r"[a-z0-9][a-z0-9/-]+", question.lower())
        if len(term) > 2 and term not in _SUFFICIENCY_STOPWORDS
    ]
    if not terms:
        return bool(nodes)
    evidence_text = "\n".join(_node_text(node) for node in nodes).lower()
    hits = sum(1 for term in set(terms) if term in evidence_text)
    required = 2 if len(set(terms)) >= 4 else 1
    return hits >= required


_SUFFICIENCY_STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "does",
    "have",
    "with",
    "from",
    "this",
    "that",
    "paper",
    "document",
    "according",
    "answer",
    "format",
    "write",
    "please",
}


def _node_text(node: dict[str, Any]) -> str:
    raw_ref = node.get("raw_ref") or {}
    metadata = node.get("metadata") or {}
    parts = [
        node.get("text_for_embedding"),
        node.get("caption"),
        node.get("ocr_text"),
        node.get("summary"),
        raw_ref.get("table_markdown"),
        raw_ref.get("table_html"),
        raw_ref.get("media_id"),
        metadata.get("media_type"),
        metadata.get("text"),
    ]
    return "\n".join(str(part) for part in parts if part)


def _looks_not_answerable(lowered_answer: str) -> bool:
    patterns = (
        "not answerable",
        "not specified",
        "not provided",
        "not mentioned",
        "no explicit mention",
        "no direct statement",
        "no specific answer",
        "not enough information",
        "insufficient evidence",
        "not available",
        "not shown",
        "not present",
        "cannot be determined",
        "can't be determined",
        "cannot determine",
        "cannot be confirmed",
        "do not contain enough",
        "does not contain enough",
    )
    return any(pattern in lowered_answer for pattern in patterns)


def _has_confident_answer_tail(text: str) -> bool:
    tail = _answer_tail(text).strip()
    if not tail:
        return False
    lowered_tail = tail.lower()
    if _looks_not_answerable(lowered_tail):
        return False
    return bool(re.search(r"[A-Za-z0-9]", tail))


def _extract_int(text: str) -> str | None:
    answer_tail = _answer_tail(text)
    if answer_tail:
        value = _first_int_like(answer_tail)
        if value is not None:
            return value
    direct = _extract_direct_int_phrase(text)
    if direct is not None:
        return direct
    if len(text.split()) > 12:
        return None
    numbers = _numeric_matches(text)
    integer_like = [raw for raw in numbers if not re.search(r"[.%/]", raw)]
    if len(integer_like) == 1:
        return str(int(integer_like[-1].replace(",", "")))
    words = _number_word_matches(text)
    if len(words) == 1:
        return str(words[-1])
    return None


def _answer_tail(text: str) -> str:
    patterns = (
        r"(?:therefore,?\s*)?(?:the\s+)?(?:final\s+)?answer(?:\s+to[\s\S]{0,240}?)?\s*(?:is|:)\s*[:\-]?\s*",
        r"(?:therefore,?\s*)?we\s+can\s+conclude\s+that\s+",
    )
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return ""
    match = max(matches, key=lambda item: item.start())
    return text[match.end() :]


def _first_int_like(text: str) -> str | None:
    token_pattern = re.compile(
        r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)\b|"
        r"(?<![A-Za-z0-9])-?\d+(?:,\d{3})*(?![A-Za-z0-9-])",
        re.IGNORECASE,
    )
    for match in token_pattern.finditer(text):
        if _is_labeled_number(text, match.start()):
            continue
        raw = match.group(0)
        word_value = _number_word_to_int(raw)
        if word_value is not None:
            return str(word_value)
        return str(int(raw.replace(",", "")))
    return None


def _is_labeled_number(text: str, start: int) -> bool:
    prefix = text[max(0, start - 16) : start].lower()
    return bool(re.search(r"\b(fig(?:ure)?|table|page|p\.|eq(?:uation)?|section)\s*$", prefix))


def _extract_direct_int_phrase(text: str) -> str | None:
    number_word = (
        r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
        r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
    )
    number = rf"-?\d+(?:,\d{{3}})*|{number_word}"
    patterns = (
        rf"\bthere\s+(?:are|is)\s+(?:about\s+|approximately\s+|at\s+least\s+|exactly\s+)?(?P<num>{number})\b",
        rf"\banswer\s+(?:is|:)\s*(?:about\s+|approximately\s+|at\s+least\s+|exactly\s+)?[*_`]*(?P<num>{number})\b",
        rf"\btherefore[^.\n]*?\bis\s+[*_`]*(?P<num>{number})\b",
    )
    matches = []
    for pattern in patterns:
        matches.extend(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return None
    raw = matches[-1].group("num")
    word_value = _number_word_to_int(raw)
    if word_value is not None:
        return str(word_value)
    return str(int(raw.replace(",", "")))


def _extract_float(text: str) -> str | None:
    if len(text.split()) > 18 and not _answer_tail(text):
        return None
    numbers = _numeric_matches(text)
    if not numbers:
        words = _number_word_matches(text)
        return str(float(words[-1])).rstrip("0").rstrip(".") if words else None
    raw = numbers[-1].replace(",", "")
    if raw.endswith("%"):
        return raw
    if "/" in raw:
        left, right = raw.split("/", 1)
        try:
            denom = float(right)
            if denom:
                return _format_float(float(left) / denom)
        except ValueError:
            return raw
    return raw


def _extract_list(text: str) -> str | None:
    bracket = re.search(r"\[[^\]]+\]", text, re.DOTALL)
    if bracket:
        value = " ".join(bracket.group(0).split())
        if re.fullmatch(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", value):
            return None
        return value
    lines = [line.strip(" -\t") for line in text.splitlines() if line.strip()]
    bullet_items = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.)]\s*", "", line).strip()
        if cleaned and cleaned != line:
            bullet_items.append(cleaned)
    if len(bullet_items) >= 2:
        return ", ".join(_strip_formatting(item) for item in bullet_items)
    answer_match = re.search(r"(?:answer is|answer:|are:)\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    if answer_match:
        value = _strip_formatting(answer_match.group(1).strip())
        if "," in value or ";" in value or " and " in value.lower():
            return value
    return None


def _numeric_matches(text: str) -> list[str]:
    return re.findall(
        r"(?<![A-Za-z0-9])-?\d+(?:,\d{3})*/\d+(?:,\d{3})*(?![A-Za-z0-9-])|"
        r"(?<![A-Za-z0-9])-?\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9-])",
        text,
    )


def _number_word_matches(text: str) -> list[int]:
    values = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }
    return [values[word] for word in re.findall(r"\b[a-z]+\b", text.lower()) if word in values]


def _number_word_to_int(text: str) -> int | None:
    matches = _number_word_matches(text)
    return matches[-1] if matches else None


def _format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _strip_formatting(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def _build_prompt(
    question: str,
    evidence_package: dict[str, list[dict[str, Any]]],
    vlm_calls: list[dict[str, Any]],
    table_reasoner_calls: list[dict[str, Any]],
    max_chars: int,
    answer_type: str,
) -> str:
    text = FINAL_PROMPT.format(
        question=question,
        answer_type=answer_type,
        text_evidence=_dump_nodes(evidence_package.get("text_evidence", []), include_raw=False),
        entity_evidence=_dump_nodes(evidence_package.get("entity_evidence", []), include_raw=False),
        page_evidence=_dump_nodes(evidence_package.get("page_evidence", []), include_raw=False),
        aggregation_context=str(evidence_package.get("aggregation_context") or ""),
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
            supported_answer = _supported_subanswer(call.get("table_answer"))
            if supported_answer or call.get("error"):
                table_by_node[node_id] = {"table_answer": supported_answer or None, "error": call.get("error")}
    rows = []
    for node in nodes:
        raw_ref = node.get("raw_ref") or {}
        table_info = (node.get("metadata") or {}).get("table_info") or {}
        rows.append(
            {
                "node_id": node.get("node_id"),
                "page": node.get("page_id"),
                "media_id": raw_ref.get("media_id"),
                "table_markdown": raw_ref.get("table_markdown"),
                "table_html": raw_ref.get("table_html"),
                "table_info": {
                    "format": table_info.get("format"),
                    "n_rows": table_info.get("n_rows"),
                    "n_cols": table_info.get("n_cols"),
                    "cells": (table_info.get("cells") or [])[:80],
                },
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
        answer = _supported_subanswer(call.get("table_answer"))
        if answer:
            return answer
    for call in vlm_calls:
        answer = _supported_subanswer(call.get("vlm_answer"))
        if answer:
            return answer
    return str(generation["answer_not_enough_evidence"])


def _supported_subanswer(answer: Any) -> str:
    text = str(answer or "").strip()
    if not text:
        return ""
    if _looks_not_answerable(text.lower()):
        return ""
    return text


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
