from __future__ import annotations

import re
from collections import Counter
from typing import Any


IMAGE_TERMS = {
    "figure",
    "fig",
    "image",
    "picture",
    "photo",
    "icon",
    "color",
    "shape",
    "screenshot",
    "map",
    "diagram",
}
CHART_TERMS = {"chart", "graph", "axis", "line", "bar", "pie", "trend", "percentage", "plot"}
TABLE_TERMS = {"table", "row", "column", "cell", "header", "metric", "f1"}
COUNTING_PATTERNS = (r"\bhow many\b", r"\bcount\b", r"\bnumber of\b", r"\btotal number\b")


def analyze_query(query: str) -> dict[str, Any]:
    text = query or ""
    lowered = text.lower()
    tokens = _keywords(lowered)
    token_counts = Counter(tokens)

    image_hits = _hits(token_counts, IMAGE_TERMS)
    chart_hits = _hits(token_counts, CHART_TERMS)
    table_hits = _hits(token_counts, TABLE_TERMS)
    is_counting = any(re.search(pattern, lowered) for pattern in COUNTING_PATTERNS)
    page_hints = _extract_page_hints(lowered)

    query_type = "text"
    if table_hits:
        query_type = "table"
    if image_hits:
        query_type = "image"
    if chart_hits:
        query_type = "chart"
    if is_counting:
        query_type = "counting" if not (image_hits or table_hits or chart_hits) else "layout"
    if (image_hits or chart_hits or table_hits) and any(term in lowered for term in ("according to", "shown", "depicted")):
        query_type = "cross_modal"
    if not tokens:
        query_type = "unknown"

    prior = {"text": 0.30, "entity": 0.30, "media": 0.20, "table": 0.10, "page": 0.10}
    if image_hits:
        prior["media"] += 0.25
        prior["text"] -= 0.05
    if chart_hits:
        prior["media"] += 0.20
        prior["table"] += 0.05
        prior["entity"] -= 0.05
    if table_hits:
        prior["table"] += 0.30
        prior["media"] += 0.10
        prior["text"] -= 0.05
    if page_hints:
        prior["page"] += 0.30
    if is_counting:
        prior["page"] += 0.10
        prior["media"] += 0.10

    prior = _normalize_prior(prior)
    return {
        "query_type": query_type,
        "expected_answer_type": _expected_answer_type(lowered),
        "modality_prior": prior,
        "page_hints": page_hints,
        "media_hints": sorted((IMAGE_TERMS | CHART_TERMS | TABLE_TERMS).intersection(tokens)),
        "keywords": tokens,
    }


def _extract_page_hints(text: str) -> list[dict[str, Any]]:
    hints = []
    patterns = (
        r"\b(?:on|at|in|from)?\s*page\s*(?:no\.?|number)?\s*(\d+)\b",
        r"\bp\.\s*(\d+)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            value = int(match.group(1))
            hints.append({"raw": match.group(0).strip(), "page": value, "base": "as_written"})
    ordinals = {
        "first page": 1,
        "second page": 2,
        "third page": 3,
        "fourth page": 4,
        "fifth page": 5,
        "sixth page": 6,
        "seventh page": 7,
        "eighth page": 8,
        "ninth page": 9,
        "tenth page": 10,
    }
    for phrase, page in ordinals.items():
        if phrase in text:
            hints.append({"raw": phrase, "page": page, "base": "ordinal"})
    seen = set()
    unique = []
    for hint in hints:
        key = (hint["raw"], hint["page"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(hint)
    return unique


def _expected_answer_type(text: str) -> str:
    if any(re.search(pattern, text) for pattern in COUNTING_PATTERNS):
        return "int"
    if any(term in text for term in ("percentage", "ratio", "score", "rate", "value")):
        return "float"
    if any(term in text for term in ("which", "list", "what are")):
        return "list" if "list" in text or "what are" in text else "str"
    return "unknown"


def _keywords(text: str) -> list[str]:
    stop = {"the", "and", "for", "with", "that", "this", "does", "what", "which", "from", "according"}
    return [token for token in re.findall(r"[a-z0-9%/-]+", text) if len(token) > 1 and token not in stop]


def _hits(counter: Counter[str], terms: set[str]) -> int:
    return sum(counter.get(term, 0) for term in terms)


def _normalize_prior(prior: dict[str, float]) -> dict[str, float]:
    cleaned = {key: max(0.01, value) for key, value in prior.items()}
    total = sum(cleaned.values()) or 1.0
    return {key: round(value / total, 4) for key, value in cleaned.items()}
