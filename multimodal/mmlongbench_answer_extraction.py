from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

from .openai_clients import resolve_api_key


ANSWER_EXTRACTION_PROMPT = """Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List.
If you find the analysis the question can not be answered from the given documents, type "Not answerable".
Exception: If the analysis only tells you that it can not read/understand the images or documents, type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
Extracted answer: answer
Answer format: Integer|Float|String|List
"""


DEFAULT_EXTRACTION_CONFIG = {
    "short_string_max_chars": 80,
    "short_identifier_max_chars": 80,
    "enable_guard": True,
}


@dataclass
class ExtractedAnswer:
    answer: str
    answer_format: str
    raw_response: str
    error: str | None = None


@dataclass
class GuardedExtraction:
    raw_prediction: str
    extracted_answer: str | None
    extracted_answer_format: str | None
    guarded_answer: str
    guarded_answer_format: str
    guard_action: str = "none"
    guard_reason: str = ""
    precheck_bypassed_llm: bool = False
    raw_normalized_prediction: str = ""
    pre_guard_extracted_answer: str = ""
    post_guard_scored_prediction: str = ""
    extraction_error: str | None = None
    extraction_raw_response: str = ""


def make_answer_extractor(config: dict[str, Any]):
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config["model"]
    prompt = str(config.get("prompt") or ANSWER_EXTRACTION_PROMPT)
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 256))

    def extract(question: str, output: str) -> ExtractedAnswer:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"\n\nQuestion:{question}\nAnalysis:{output}\n",
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            raw = str(response.choices[0].message.content or "").strip()
            answer, answer_format = parse_extraction_response(raw)
            return ExtractedAnswer(answer=answer, answer_format=answer_format, raw_response=raw)
        except Exception as exc:
            return ExtractedAnswer(answer="Fail to answer", answer_format="Str", raw_response="", error=str(exc))

    return extract


def parse_extraction_response(text: str) -> tuple[str, str]:
    text = str(text or "").strip()
    answer_match = re.search(
        r"Extracted\s+answer\s*:\s*(?P<answer>.*?)(?:\n\s*Answer\s+format\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    format_match = re.search(r"Answer\s+format\s*:\s*\[?\s*(?P<format>[A-Za-z]+)", text, flags=re.IGNORECASE)
    answer = answer_match.group("answer").strip() if answer_match else _strip_answer_label(text)
    answer_format = _normalize_answer_format(format_match.group("format") if format_match else "")
    return answer, answer_format


def guard_extraction(
    prediction: Any,
    answer_format: str | None,
    extractor=None,
    question: str = "",
    existing_extracted: ExtractedAnswer | None = None,
    config: dict[str, Any] | None = None,
) -> GuardedExtraction:
    options = _extraction_options(config)
    raw_prediction = "" if prediction is None else str(prediction)
    raw_normalized = normalize_extracted_answer(raw_prediction, answer_format=answer_format)
    expected_format = _gold_answer_format(answer_format)

    if options["enable_guard"]:
        precheck = deterministic_precheck(raw_prediction, expected_format, options)
        if precheck is not None:
            return GuardedExtraction(
                raw_prediction=raw_prediction,
                extracted_answer=precheck.answer,
                extracted_answer_format=precheck.answer_format,
                guarded_answer=precheck.answer,
                guarded_answer_format=precheck.answer_format,
                guard_action=precheck.action,
                guard_reason=precheck.reason,
                precheck_bypassed_llm=True,
                raw_normalized_prediction=raw_normalized,
                pre_guard_extracted_answer=precheck.answer,
                post_guard_scored_prediction=precheck.answer,
                extraction_raw_response=precheck.raw_response,
                extraction_error=precheck.error,
            )

    if existing_extracted is not None:
        extracted = existing_extracted
    elif extractor is not None:
        extracted = extractor(question, raw_prediction)
    else:
        extracted = ExtractedAnswer(answer=raw_prediction, answer_format=expected_format, raw_response="", error=None)

    if not options["enable_guard"]:
        normalized = normalize_extracted_answer(extracted.answer, answer_format=expected_format)
        return GuardedExtraction(
            raw_prediction=raw_prediction,
            extracted_answer=extracted.answer,
            extracted_answer_format=_normalize_answer_format(extracted.answer_format),
            guarded_answer=normalized,
            guarded_answer_format=_normalize_answer_format(extracted.answer_format),
            guard_action="none",
            guard_reason="guard_disabled",
            precheck_bypassed_llm=False,
            raw_normalized_prediction=raw_normalized,
            pre_guard_extracted_answer=str(extracted.answer or ""),
            post_guard_scored_prediction=normalized,
            extraction_raw_response=extracted.raw_response,
            extraction_error=extracted.error,
        )

    guarded = apply_extraction_guard(raw_prediction, expected_format, extracted, options)
    guarded.raw_normalized_prediction = raw_normalized
    return guarded


@dataclass
class _PrecheckResult:
    answer: str
    answer_format: str
    action: str
    reason: str
    raw_response: str = ""
    error: str | None = None


def deterministic_precheck(raw_prediction: str, expected_format: str, config: dict[str, Any]) -> _PrecheckResult | None:
    normalized = normalize_extracted_answer(raw_prediction, answer_format=expected_format)
    if is_not_answerable(normalized):
        return _PrecheckResult("Not answerable", "Str", "bypass_not_answerable", "raw_prediction_is_not_answerable")

    if expected_format in {"Int", "Float"}:
        scalar = _extract_standalone_number(normalized, integer=expected_format == "Int")
        if scalar is not None:
            return _PrecheckResult(scalar, expected_format, "bypass_short_numeric", "raw_prediction_is_standalone_numeric")

    if _is_single_url(normalized):
        return _PrecheckResult(normalized, "Str", "bypass_url", "raw_prediction_is_single_url")

    if _is_short_identifier(normalized, int(config["short_identifier_max_chars"])):
        return _PrecheckResult(normalized, "Str", "bypass_identifier", "raw_prediction_is_short_identifier")

    if expected_format == "Str" and _is_safe_short_string(normalized, int(config["short_string_max_chars"])):
        return _PrecheckResult(normalized, "Str", "bypass_short_string", "raw_prediction_is_safe_short_string")

    if expected_format == "List":
        items = parse_list_answer(normalized)
        if len(items) >= 2:
            return _PrecheckResult(_format_list_answer(items), "List", "bypass_raw_list", "raw_prediction_is_parseable_list")

    return None


def apply_extraction_guard(
    raw_prediction: str,
    expected_format: str,
    extracted: ExtractedAnswer,
    config: dict[str, Any],
) -> GuardedExtraction:
    raw_normalized = normalize_extracted_answer(raw_prediction, answer_format=expected_format)
    pre_guard = "" if extracted.answer is None else str(extracted.answer)
    extracted_format = _normalize_answer_format(extracted.answer_format)
    normalized = normalize_extracted_answer(pre_guard, answer_format=expected_format)
    action = "none"
    reason = ""

    if is_empty_extraction(normalized):
        normalized = raw_normalized
        extracted_format = expected_format
        action = "parse_error_fallback"
        reason = "empty_or_template_only_extraction"
    elif is_not_answerable(normalized):
        normalized = "Not answerable"
        extracted_format = "Str"
        action = "normalize_not_answerable"
        reason = "normalized_not_answerable_variant"

    raw_list_items = parse_list_answer(raw_normalized) if expected_format == "List" else []
    extracted_list_items = parse_list_answer(normalized) if expected_format == "List" else []
    if expected_format == "List" and len(raw_list_items) >= 2:
        if is_not_answerable(normalized):
            normalized = _format_list_answer(raw_list_items)
            extracted_format = "List"
            action = "fallback_raw_list"
            reason = "extracted_not_answerable_but_raw_contains_multiple_items"
        elif extracted_format != "List" or len(extracted_list_items) <= 1 or len(extracted_list_items) < len(raw_list_items):
            normalized = _format_list_answer(raw_list_items)
            extracted_format = "List"
            action = "fallback_raw_list"
            reason = "extracted_list_is_single_or_shorter_than_raw_list"

    if expected_format in {"Int", "Float"}:
        scalar = _extract_standalone_number(normalized, integer=expected_format == "Int")
        if scalar is not None:
            normalized = scalar
            extracted_format = expected_format
            if action == "none":
                action = "coerce_numeric_format"
                reason = "coerced_extracted_numeric_to_expected_format"
        elif extracted_format == "List":
            raw_scalar = _extract_standalone_number(raw_normalized, integer=expected_format == "Int")
            if raw_scalar is not None:
                normalized = raw_scalar
                extracted_format = expected_format
                action = "coerce_numeric_format"
                reason = "fallback_raw_numeric_for_inconsistent_extracted_list"

    if expected_format == "Float" and extracted_format == "Str":
        scalar = _extract_standalone_number(normalized, integer=False)
        if scalar is not None:
            normalized = scalar
            extracted_format = "Float"
            action = "coerce_numeric_format"
            reason = "coerced_string_numeric_to_float"

    if expected_format == "Str" and extracted_format in {"Int", "Float"}:
        raw_scalar = _extract_standalone_number(raw_normalized, integer=extracted_format == "Int")
        if raw_scalar is None:
            normalized = raw_normalized
            extracted_format = "Str"
            action = "fallback_raw_prediction"
            reason = "string_question_extracted_as_numeric_without_numeric_raw"

    if expected_format == "Unknown" and not is_not_answerable(normalized):
        reason = reason or "unknown_question_extracted_non_not_answerable"

    if _is_safe_short_string(raw_normalized, int(config["short_string_max_chars"])) and _short_answer_was_unnecessarily_rewritten(raw_normalized, normalized):
        normalized = raw_normalized
        extracted_format = expected_format if expected_format != "Unknown" else "Str"
        action = "preserve_short_answer"
        reason = "raw_short_answer_was_unnecessarily_rewritten"

    if is_empty_extraction(normalized):
        normalized = raw_normalized
        extracted_format = expected_format if expected_format != "Unknown" else "Str"
        action = "fallback_raw_prediction"
        reason = "guarded_answer_empty_after_normalization"

    return GuardedExtraction(
        raw_prediction=raw_prediction,
        extracted_answer=extracted.answer,
        extracted_answer_format=extracted_format,
        guarded_answer=normalized,
        guarded_answer_format=extracted_format,
        guard_action=action,
        guard_reason=reason,
        precheck_bypassed_llm=False,
        raw_normalized_prediction=raw_normalized,
        pre_guard_extracted_answer=pre_guard,
        post_guard_scored_prediction=normalized,
        extraction_error=extracted.error,
        extraction_raw_response=extracted.raw_response,
    )


def normalize_extracted_answer(value: Any, answer_format: str | None = None) -> str:
    text = "" if value is None else str(value)
    text = text.strip().strip("\ufeff")
    text = _strip_answer_label(text)
    text = _strip_format_label_suffix(text)
    text = _strip_wrapping_quotes(text)
    if _bracket_inner_is_not_answerable(text):
        return "Not answerable"
    if is_not_answerable(text):
        return "Not answerable"
    if _is_template_only(text):
        return ""
    text = _strip_template_lines(text)
    text = _strip_wrapping_quotes(text.strip())
    if _bracket_inner_is_scalar(text) and not _looks_like_multi_value_list(text):
        text = text[1:-1].strip()
    if is_not_answerable(text):
        return "Not answerable"
    if _is_null_like(text) or _only_punctuation(text):
        return ""
    return text.strip()


def parse_list_answer(text: Any) -> list[str]:
    raw = normalize_extracted_answer(text)
    if not raw or is_not_answerable(raw) or _is_single_url(raw):
        return []

    parsed_items = _parse_bracket_list(raw)
    if parsed_items:
        return _dedupe_nonempty(parsed_items)

    bullet_items = []
    for line in raw.splitlines():
        match = re.match(r"^\s*(?:[-*]\s+|\d+[\.)]\s+)(.+?)\s*$", line)
        if match:
            bullet_items.append(match.group(1).strip())
    if len(bullet_items) >= 2:
        return _dedupe_nonempty(bullet_items)

    if "\n" in raw:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) >= 2 and all(_is_reasonable_list_item(item) for item in lines):
            return _dedupe_nonempty(lines)

    delimiter = ";" if ";" in raw else ","
    if delimiter in raw and not _is_single_url(raw):
        parts = [part.strip() for part in raw.split(delimiter)]
        if len(parts) >= 2 and all(_is_reasonable_list_item(item) for item in parts):
            return _dedupe_nonempty(parts)
    return []


def _normalize_answer_format(value: str) -> str:
    text = str(value or "").strip().strip("[]").strip().lower()
    if text in {"int", "integer"}:
        return "Int"
    if text in {"float", "number"}:
        return "Float"
    if text in {"str", "string", "none", "unknown"}:
        return "Str"
    if text in {"list", "array"}:
        return "List"
    return "Str"


def _extraction_options(config: dict[str, Any] | None) -> dict[str, Any]:
    options = dict(DEFAULT_EXTRACTION_CONFIG)
    if isinstance(config, dict):
        options.update({key: value for key, value in config.items() if key in options})
    return options


def _gold_answer_format(value: str | None) -> str:
    text = str(value or "").strip().lower()
    if text in {"int", "integer"}:
        return "Int"
    if text in {"float", "number"}:
        return "Float"
    if text in {"list", "array"}:
        return "List"
    if text in {"unknown", "none", "null", "not answerable", "unanswerable"}:
        return "Unknown"
    return "Str"


def is_not_answerable(value: Any) -> bool:
    text = str(value or "").strip()
    text = _strip_answer_label(text)
    text = _strip_format_label_suffix(text)
    text = _strip_wrapping_quotes(text)
    if _bracket_inner_is_not_answerable(text):
        return True
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return normalized in {"not answerable", "unanswerable", "unknown"}


def is_empty_extraction(value: Any) -> bool:
    text = normalize_extracted_answer(value)
    return not text or _is_null_like(text) or _is_template_only(text) or _only_punctuation(text)


def _strip_answer_label(text: str) -> str:
    cleaned = str(text or "").strip()
    patterns = (
        r"^\s*Extracted\s+answer\s*:\s*",
        r"^\s*Answer\s*:\s*",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _strip_format_label_suffix(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"\n\s*Answer\s+format\s*:\s*\[?\s*[A-Za-z]+\s*\]?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*[;|]\s*Answer\s+format\s*:\s*\[?\s*[A-Za-z]+\s*\]?\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\|\s*\[?\s*(?:String|Str|Integer|Int|Float|List)\s*\]?\s*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _strip_template_lines(text: str) -> str:
    lines = []
    for line in str(text or "").splitlines():
        if re.fullmatch(r"\s*(?:Answer\s+format|Extracted\s+answer|Answer)\s*:?\s*", line, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\s*\[\s*(?:answer|answer\s+format|String|Integer|Float|List)\s*\]\s*", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _strip_wrapping_quotes(text: str) -> str:
    cleaned = str(text or "").strip()
    while len(cleaned) >= 2 and ((cleaned[0] == cleaned[-1] == '"') or (cleaned[0] == cleaned[-1] == "'")):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _bracket_inner_is_not_answerable(text: str) -> bool:
    match = re.fullmatch(r"\[\s*['\"]?\s*Not\s+answerable\s*['\"]?\s*\]", str(text or "").strip(), flags=re.IGNORECASE)
    return bool(match)


def _bracket_inner_is_scalar(text: str) -> bool:
    stripped = str(text or "").strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    inner = stripped[1:-1].strip()
    return bool(inner) and "," not in inner and ";" not in inner


def _looks_like_multi_value_list(text: str) -> bool:
    stripped = str(text or "").strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False
    inner = stripped[1:-1]
    return "," in inner or ";" in inner


def _is_template_only(text: str) -> bool:
    normalized = re.sub(r"[\s:\[\]|-]+", " ", str(text or "").lower()).strip()
    return normalized in {"answer", "answer format", "extracted answer", "string", "integer", "float", "list"}


def _is_null_like(text: str) -> bool:
    return str(text or "").strip().lower() in {"", "none", "null", "n/a", "na"}


def _only_punctuation(text: str) -> bool:
    return bool(str(text or "").strip()) and not re.search(r"[A-Za-z0-9]", str(text or ""))


def _extract_standalone_number(text: str, integer: bool) -> str | None:
    cleaned = normalize_extracted_answer(text)
    cleaned = cleaned.replace(",", "").strip()
    pattern = r"-?\d+" if integer else r"-?\d+(?:\.\d+)?%?"
    if not re.fullmatch(pattern, cleaned):
        return None
    if integer:
        try:
            return str(int(cleaned))
        except ValueError:
            return None
    return cleaned


def _is_single_url(text: str) -> bool:
    return bool(re.fullmatch(r"https?://\S+", str(text or "").strip(), flags=re.IGNORECASE))


def _is_short_identifier(text: str, max_chars: int) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned or len(cleaned) > max_chars or "\n" in cleaned:
        return False
    if re.search(r"\s", cleaned):
        return False
    if re.fullmatch(r"[\w.+-]+\.(?:py|ipynb|json|yaml|yml|csv|tsv|txt|pdf|md|html|xml)", cleaned, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:/+-]{1,}", cleaned) and any(char.isupper() or char.isdigit() or char in "._:+-" for char in cleaned):
        return True
    return False


def _is_safe_short_string(text: str, max_chars: int) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned or len(cleaned) > max_chars or "\n" in cleaned:
        return False
    lowered = cleaned.lower()
    long_markers = (
        "based on",
        "according to",
        "according to the context",
        "therefore",
        "the answer is",
        "this means",
        "it appears",
        "provided information",
        "context",
    )
    if any(marker in lowered for marker in long_markers):
        return False
    if re.search(r"\bor\b|\band/or\b|;", lowered):
        return False
    if len(parse_list_answer(cleaned)) >= 2:
        return False
    return bool(re.search(r"[A-Za-z0-9]", cleaned))


def _short_answer_was_unnecessarily_rewritten(raw: str, extracted: str) -> bool:
    raw_clean = str(raw or "").strip()
    extracted_clean = str(extracted or "").strip()
    if not raw_clean or not extracted_clean or raw_clean == extracted_clean:
        return False
    raw_norm = raw_clean.lower()
    extracted_norm = extracted_clean.lower()
    if raw_norm in extracted_norm or extracted_norm in raw_norm:
        return True
    raw_tokens = raw_norm.split()
    extracted_tokens = extracted_norm.split()
    if len(raw_tokens) <= 4 and len(extracted_tokens) <= 8:
        overlap = set(raw_tokens) & set(extracted_tokens)
        return bool(overlap) and overlap != set(raw_tokens)
    return False


def _parse_bracket_list(text: str) -> list[str]:
    stripped = str(text or "").strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return []
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        parsed = None
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed]
    inner = stripped[1:-1].strip()
    if "," in inner or ";" in inner:
        delimiter = ";" if ";" in inner else ","
        return [part.strip().strip("'\"") for part in inner.split(delimiter)]
    return []


def _is_reasonable_list_item(item: str) -> bool:
    cleaned = str(item or "").strip()
    if not cleaned or _only_punctuation(cleaned) or is_not_answerable(cleaned):
        return False
    if len(cleaned) > 160:
        return False
    return True


def _dedupe_nonempty(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        cleaned = normalize_extracted_answer(item).strip().strip("-*").strip()
        cleaned = re.sub(r"^\d+[\.)]\s*", "", cleaned).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _format_list_answer(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"
