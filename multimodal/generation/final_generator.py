from __future__ import annotations

import json
import re
from typing import Any, Callable


FINAL_PROMPT = """Question:
{question}

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
            answer = str(llm_func(prompt))
            return _with_postprocess(answer, question, answer_plan, global_config, prompt, True, None)
        except TypeError:
            try:
                answer = str(llm_func(question, system_prompt=prompt))
                return _with_postprocess(answer, question, answer_plan, global_config, prompt, True, None)
            except Exception as exc:
                answer = _fallback_answer(vlm_calls, table_reasoner_calls, generation)
                return _with_postprocess(answer, question, answer_plan, global_config, prompt, False, str(exc))
        except Exception as exc:
            answer = _fallback_answer(vlm_calls, table_reasoner_calls, generation)
            return _with_postprocess(answer, question, answer_plan, global_config, prompt, False, str(exc))
    answer = _fallback_answer(vlm_calls, table_reasoner_calls, generation)
    return _with_postprocess(answer, question, answer_plan, global_config, prompt, False, "llm_func_missing")


def _with_postprocess(
    raw_answer: str,
    question: str,
    answer_plan: dict[str, Any],
    global_config: dict[str, Any],
    prompt: str,
    used_llm: bool,
    error: str | None,
) -> tuple[str, dict[str, Any]]:
    processed, postprocess = postprocess_final_answer(raw_answer, question, answer_plan, global_config)
    return processed, {
        "prompt_preview": prompt[:1000],
        "used_llm": used_llm,
        "error": error,
        "raw_answer": raw_answer,
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

    if target == "unknown" and _looks_not_answerable(lowered):
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
    if answer_format in {"unknown", "not answerable", "unanswerable"}:
        return "unknown"

    expected = str(answer_plan.get("expected_answer_type") or "").strip().lower()
    if expected in {"int", "float", "list"}:
        return expected
    text = question.lower()
    if re.search(r"\b(how many|number of|count)\b", text):
        return "int"
    if re.search(r"\b(percentage|ratio|float|average|gap|score|rate|difference)\b", text):
        return "float"
    if re.search(r"\b(list|what are|which two|all the)\b", text):
        return "list"
    return "unknown"


def _looks_not_answerable(lowered_answer: str) -> bool:
    patterns = (
        "not answerable",
        "not specified",
        "not provided",
        "not mentioned",
        "no explicit mention",
        "no direct statement",
        "insufficient evidence",
        "cannot be determined",
        "can't be determined",
        "cannot determine",
        "do not contain enough",
        "does not contain enough",
    )
    return any(pattern in lowered_answer for pattern in patterns)


def _extract_int(text: str) -> str | None:
    answer_tail = _answer_tail(text)
    if answer_tail:
        value = _first_int_like(answer_tail)
        if value is not None:
            return value
    direct = _extract_direct_int_phrase(text)
    if direct is not None:
        return direct
    numbers = _numeric_matches(text)
    integer_like = [raw for raw in numbers if not re.search(r"[.%/]", raw)]
    if integer_like:
        return str(int(integer_like[-1].replace(",", "")))
    words = _number_word_matches(text)
    if words:
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
        return " ".join(bracket.group(0).split())
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
) -> str:
    text = FINAL_PROMPT.format(
        question=question,
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
