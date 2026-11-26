from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def split_source_ids(value: Any) -> list[str]:
    """Split legacy pipe-delimited sources without truncating or reordering evidence."""
    if isinstance(value, (list, tuple, set)):
        parts = [part for item in value for part in str(item or "").split("|")]
    else:
        parts = str(value or "").split("|")
    return list(dict.fromkeys(part.strip() for part in parts if part.strip()))


def complete_source_id(value: Any) -> str:
    return "|".join(split_source_ids(value))


def resolve_source_evidence(
    working_dir: str | Path,
    source_values: Iterable[Any],
    chunks_file: str | Path | None = None,
    max_text: int = 5,
    max_media: int = 5,
    include_media: bool = True,
) -> dict[str, Any]:
    """Resolve entity source IDs against both text chunks and mm_media records."""
    working = Path(working_dir)
    requests = _source_requests(source_values)
    text_by_id = _load_text_index(working, chunks_file)
    media_by_id = _load_media_index(working)
    counts = Counter(request["source_id"] for request in requests)
    origins: dict[str, set[str]] = defaultdict(set)
    for request in requests:
        if request["entity_name"]:
            origins[request["source_id"]].add(request["entity_name"])

    ordered_ids = sorted(counts, key=lambda source_id: (-counts[source_id], source_id))
    text_evidence: list[dict[str, Any]] = []
    media_evidence: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for source_id in ordered_ids:
        origin_entities = sorted(origins[source_id])
        if source_id in text_by_id:
            evidence = _text_evidence(text_by_id[source_id], source_id, counts[source_id], origin_entities)
            selected = len(text_evidence) < max(0, int(max_text))
            if selected:
                text_evidence.append(evidence)
            trace.append(_trace(source_id, "text_chunk", selected, origin_entities))
        elif source_id in media_by_id:
            evidence = _media_evidence(media_by_id[source_id], source_id, counts[source_id], origin_entities)
            selected = bool(include_media and len(media_evidence) < max(0, int(max_media)))
            if selected:
                media_evidence.append(evidence)
            trace.append(_trace(source_id, "media", selected, origin_entities))
        else:
            unresolved.append(source_id)
            trace.append(_trace(source_id, "unresolved", False, origin_entities))
    return {
        "text_evidence": text_evidence,
        "media_evidence": media_evidence,
        "unresolved_source_ids": unresolved,
        "trace": trace,
        "counts": {
            "requested": len(ordered_ids),
            "resolved_text": sum(item["resolved_kind"] == "text_chunk" for item in trace),
            "resolved_media": sum(item["resolved_kind"] == "media" for item in trace),
            "unresolved": len(unresolved),
            "selected_text": len(text_evidence),
            "selected_media": len(media_evidence),
        },
    }


def _source_requests(values: Iterable[Any]) -> list[dict[str, str]]:
    requests: list[dict[str, str]] = []
    for value in values or []:
        if isinstance(value, dict):
            source_value = value.get("source_id")
            entity_name = str(value.get("entity_name") or "").strip()
        else:
            source_value = value
            entity_name = ""
        for source_id in split_source_ids(source_value):
            requests.append({"source_id": source_id, "entity_name": entity_name})
    return requests


def _load_text_index(working: Path, chunks_file: str | Path | None) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = [working / "mm_chunk.json"]
    if chunks_file:
        candidate = Path(chunks_file)
        if candidate not in candidates:
            candidates.append(candidate)
    for path in candidates:
        if not path.is_file():
            continue
        value = _read_json(path)
        if isinstance(value, list):
            rows.extend(item for item in value if isinstance(item, dict))
    result: dict[str, dict[str, Any]] = {}
    for item in rows:
        for key in (item.get("hash_code"), item.get("chunk_id")):
            if key and str(key) not in result:
                result[str(key)] = item
    return result


def _load_media_index(working: Path) -> dict[str, dict[str, Any]]:
    path = working / "mm_media.json"
    if not path.is_file():
        return {}
    value = _read_json(path)
    return {
        str(item["media_id"]): item
        for item in value if isinstance(item, dict) and item.get("media_id")
    } if isinstance(value, list) else {}


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _text_evidence(item: dict[str, Any], source_id: str, score: int, origins: list[str]) -> dict[str, Any]:
    text = str(item.get("text") or "")
    page = item.get("page_start") if item.get("page_start") is not None else item.get("page")
    return {
        **item,
        "hash_code": item.get("hash_code") or source_id,
        "chunk_id": item.get("chunk_id") or source_id,
        "modality": "text",
        "page": page,
        "summary": item.get("summary") or text[:240],
        "score": score,
        "source_resolution": "entity_source_id",
        "origin_entities": origins,
    }


def _media_evidence(item: dict[str, Any], source_id: str, score: int, origins: list[str]) -> dict[str, Any]:
    media_type = str(item.get("mapped_type") or item.get("type") or item.get("modality") or "generic").lower()
    path = str(item.get("path") or "")
    return {
        **item,
        "media_id": item.get("media_id") or source_id,
        "modality": media_type,
        "mapped_type": media_type,
        "type": media_type,
        "path": path,
        "asset_path": path,
        "source_path": path,
        "page": item.get("page"),
        "bbox": item.get("bbox"),
        "caption": item.get("caption") or "",
        "summary": item.get("summary") or "",
        "ocr_text": item.get("ocr_text") or "",
        "table_html": item.get("table_html") or "",
        "table_markdown": item.get("table_markdown") or "",
        "score": score,
        "source_resolution": "entity_source_id",
        "origin_entities": origins,
    }


def _trace(source_id: str, kind: str, selected: bool, origins: list[str]) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "resolved_kind": kind,
        "selected": selected,
        "expansion_origin": "entity_source_id",
        "origin_entities": origins,
    }
