from __future__ import annotations

import re
from typing import Any


REF_PATTERN = re.compile(
    r"\b(?P<kind>fig(?:ure)?|table|chart)\s*\.?\s*(?P<number>\d+[a-z]?)\b",
    re.IGNORECASE,
)


def extract_media_refs(text: str) -> list[dict[str, str]]:
    refs = []
    seen = set()
    for match in REF_PATTERN.finditer(text or ""):
        kind = _normalize_kind(match.group("kind"))
        number = match.group("number").lower()
        key = (kind, number)
        if key in seen:
            continue
        seen.add(key)
        refs.append({"kind": kind, "number": number, "raw": match.group(0).strip()})
    return refs


def candidate_matches_media_ref(candidate: dict[str, Any], ref: dict[str, Any]) -> bool:
    kind = _normalize_kind(str(ref.get("kind") or ""))
    number = str(ref.get("number") or "").lower()
    if not kind or not number:
        return False

    text = _candidate_text(candidate)
    patterns = _ref_patterns(kind, number)
    if not any(pattern.search(text) for pattern in patterns):
        return False

    media_type = str((candidate.get("metadata") or {}).get("media_type") or "").lower()
    if kind == "table":
        return media_type == "table" or bool((candidate.get("raw_ref") or {}).get("table_markdown") or (candidate.get("raw_ref") or {}).get("table_html"))
    if kind == "chart":
        return media_type in {"chart", "figure", "image", "unknown"}
    return media_type in {"figure", "image", "chart", "unknown"}


def matching_media_refs(candidate: dict[str, Any], refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [ref for ref in refs or [] if candidate_matches_media_ref(candidate, ref)]


def media_ref_kinds(refs: list[dict[str, Any]]) -> set[str]:
    return {_normalize_kind(str(ref.get("kind") or "")) for ref in refs or [] if ref.get("kind")}


def _normalize_kind(kind: str) -> str:
    lowered = kind.lower()
    if lowered in {"fig", "figure"}:
        return "figure"
    return lowered


def _ref_patterns(kind: str, number: str) -> list[re.Pattern[str]]:
    if kind == "figure":
        labels = ("figure", "fig")
    else:
        labels = (kind,)
    return [
        re.compile(rf"\b{label}\s*\.?\s*{re.escape(number)}\b", re.IGNORECASE)
        for label in labels
    ]


def _candidate_text(candidate: dict[str, Any]) -> str:
    raw_ref = candidate.get("raw_ref") or {}
    metadata = candidate.get("metadata") or {}
    parts = [
        candidate.get("text_for_embedding"),
        candidate.get("caption"),
        candidate.get("summary"),
        candidate.get("ocr_text"),
        raw_ref.get("media_id"),
        raw_ref.get("table_markdown"),
        raw_ref.get("table_html"),
        metadata.get("caption"),
        metadata.get("summary"),
    ]
    return "\n".join(str(part) for part in parts if part).lower()
