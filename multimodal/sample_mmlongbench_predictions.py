from __future__ import annotations

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any


ALIASES = {
    "chart": {"chart", "graph", "plot"},
    "table": {"table"},
    "figure": {"figure", "fig", "image", "picture"},
}


def sample_predictions(
    input_file: str | Path,
    output_file: str | Path,
    per_type: int,
    target_types: list[str] | None = None,
    answer_formats: list[str] | None = None,
    random_sample: bool = False,
    seed: int = 42,
    compact: bool = False,
    annotate: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_file)
    output_path = Path(output_file)
    target_types = target_types or []
    answer_formats = answer_formats or []
    wanted = [f"evidence:{item}" for item in _normalize_evidence_types(target_types)]
    wanted.extend(f"answer_format:{_normalize_answer_format(item)}" for item in answer_formats)
    if not wanted:
        raise ValueError("No target evidence types or answer formats were requested.")

    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = {item: [] for item in wanted}
    seen_question_ids: dict[str, set[str]] = {item: set() for item in wanted}
    scanned = 0
    malformed = 0

    with input_path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            scanned += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue

            matched = [item for item in wanted if item in detect_sample_keys(row)]
            if not matched:
                continue

            if not random_sample:
                candidates = [item for item in matched if len(buckets[item]) < per_type]
                if not candidates:
                    continue
                item = min(candidates, key=lambda key: len(buckets[key]))
                buckets[item].append(_prepare_row(row, item, compact, annotate))
                if all(len(buckets[item]) >= per_type for item in wanted):
                    break
                continue

            for item in matched:
                question_key = str(row.get("question_id") or f"line:{line_no}")
                if question_key in seen_question_ids[item]:
                    continue
                seen_question_ids[item].add(question_key)

                stored = _prepare_row(row, item, compact, annotate)
                _reservoir_add(buckets[item], stored, per_type, len(seen_question_ids[item]), rng)

    rows: list[dict[str, Any]] = []
    emitted_ids: set[tuple[str, str]] = set()
    for item in wanted:
        for row in buckets[item]:
            key = (str(row.get("question_id")), item)
            if key not in emitted_ids:
                emitted_ids.add(key)
                rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "scanned_lines": scanned,
        "malformed_lines": malformed,
        "written_lines": len(rows),
        "matched_by_type": {item: len(buckets[item]) for item in wanted},
    }


def detect_sample_keys(row: dict[str, Any]) -> set[str]:
    keys = {f"evidence:{item}" for item in detect_types(row)}
    answer_format = _row_answer_format(row)
    if answer_format:
        keys.add(f"answer_format:{answer_format}")
    return keys


def detect_types(row: dict[str, Any]) -> set[str]:
    source_values: list[str] = []
    has_source_field = False

    for key in ("evidence_source", "evidence_sources", "source", "source_type", "evidence_type"):
        if key in row and row.get(key) is not None:
            has_source_field = True
        source_values.extend(_flatten_source_value(row.get(key)))

    trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
    for key in ("evidence_source", "evidence_sources", "source", "source_type", "evidence_type"):
        if key in trace and trace.get(key) is not None:
            has_source_field = True
        source_values.extend(_flatten_source_value(trace.get(key)))

    detected_from_source = _detect_from_values(source_values)
    if has_source_field:
        return detected_from_source

    values: list[str] = []
    if row.get("table_evidence") or trace.get("table_evidence"):
        values.append("table")

    for group_key in ("visual_evidence", "table_evidence", "selected_evidence_nodes", "all_selected_nodes"):
        for item in _as_list(row.get(group_key)) + _as_list(trace.get(group_key)):
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            raw_ref = item.get("raw_ref") if isinstance(item.get("raw_ref"), dict) else {}
            values.extend(
                _flatten_source_value(
                    [
                        item.get("modality"),
                        item.get("media_type"),
                        metadata.get("media_type"),
                        metadata.get("modality"),
                        raw_ref.get("media_type"),
                    ]
                )
            )
            caption = " ".join(
                str(item.get(key) or raw_ref.get(key) or "")
                for key in ("caption", "summary", "text_for_embedding")
            )
            if caption:
                values.append(caption)

    return _detect_from_values(values)


def _detect_from_values(values: list[str]) -> set[str]:
    detected: set[str] = set()
    for value in values:
        lowered = str(value).lower()
        for canonical, aliases in ALIASES.items():
            if any(alias in lowered for alias in aliases):
                detected.add(canonical)
    return detected


def _prepare_row(row: dict[str, Any], matched_type: str, compact: bool, annotate: bool) -> dict[str, Any]:
    output = _compact_row(row) if compact else dict(row)
    if annotate:
        output["_sampled_type"] = matched_type
    return output


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": row.get("doc_id"),
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "answer_format": row.get("answer_format"),
        "evidence_source": row.get("evidence_source") or row.get("evidence_sources"),
        "prediction": row.get("prediction"),
        "text_evidence": [_slim_text(item) for item in _as_list(row.get("text_evidence"))[:3]],
        "visual_evidence": [_slim_media(item) for item in _as_list(row.get("visual_evidence"))[:3]],
        "table_evidence": [_slim_media(item) for item in _as_list(row.get("table_evidence"))[:3]],
    }


def _slim_text(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    return {
        "chunk_id": item.get("chunk_id"),
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "section_title": item.get("section_title"),
        "score": item.get("score"),
        "text": _truncate(item.get("text"), 500),
    }


def _slim_media(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    return {
        "media_id": item.get("media_id"),
        "modality": item.get("modality"),
        "page": item.get("page"),
        "path": item.get("path"),
        "score": item.get("score"),
        "caption": _truncate(item.get("caption"), 500),
        "summary": _truncate(item.get("summary"), 500),
        "table_markdown": _truncate(item.get("table_markdown"), 500),
    }


def _flatten_source_value(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_source_value(item))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_source_value(item))
        return flattened

    text = str(value).strip()
    if text.startswith("[") or text.startswith("("):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = None
        if parsed is not None and parsed is not value:
            return _flatten_source_value(parsed)
    return [text]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _reservoir_add(
    bucket: list[dict[str, Any]],
    row: dict[str, Any],
    per_type: int,
    seen_count: int,
    rng: random.Random,
) -> None:
    if len(bucket) < per_type:
        bucket.append(row)
        return
    index = rng.randrange(seen_count)
    if index < per_type:
        bucket[index] = row


def _normalize_evidence_types(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        item = _normalize_evidence_type(value)
        if item in ALIASES and item not in normalized:
            normalized.append(item)
    return normalized


def _normalize_evidence_type(value: str) -> str:
    lowered = value.strip().lower()
    for canonical, aliases in ALIASES.items():
        if lowered == canonical or lowered in aliases:
            return canonical
    return lowered


def _normalize_answer_format(value: str) -> str:
    lowered = str(value).strip().lower()
    aliases = {
        "integer": "int",
        "number": "int",
        "numeric": "int",
        "string": "str",
        "text": "str",
        "none": "none",
        "null": "none",
        "not_answerable": "none",
        "not answerable": "none",
    }
    return aliases.get(lowered, lowered)


def _row_answer_format(row: dict[str, Any]) -> str:
    value = row.get("answer_format") or row.get("answer_type")
    if value in (None, ""):
        trace = row.get("trace") if isinstance(row.get("trace"), dict) else {}
        value = trace.get("answer_format") or trace.get("answer_type")
    return _normalize_answer_format(value) if value not in (None, "") else ""


def _truncate(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream-sample MMLongBench-Doc prediction rows by evidence type and answer format."
    )
    parser.add_argument("--input_file", required=True, help="Path to mmlongbench_doc_predictions.jsonl.")
    parser.add_argument("--output_file", required=True, help="Small JSONL file to write sampled rows.")
    parser.add_argument(
        "--evidence_types",
        "--types",
        nargs="+",
        default=["Chart", "Table", "Figure"],
        help="Evidence types to sample, for example Chart Table Figure.",
    )
    parser.add_argument(
        "--answer_formats",
        nargs="+",
        default=["Int", "Str", "Float", "List", "None"],
        help="Answer formats to sample, for example Int Str Float List None.",
    )
    parser.add_argument("--per_type", type=int, default=3, help="Number of rows to copy for each type.")
    parser.add_argument("--random", action="store_true", help="Use reservoir sampling instead of first matches.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compact", action="store_true", help="Write a compact version of each row.")
    parser.add_argument("--no_annotation", action="store_true", help="Do not add _sampled_type to rows.")
    args = parser.parse_args()

    summary = sample_predictions(
        input_file=args.input_file,
        output_file=args.output_file,
        target_types=args.evidence_types,
        answer_formats=args.answer_formats,
        per_type=args.per_type,
        random_sample=args.random,
        seed=args.seed,
        compact=args.compact,
        annotate=not args.no_annotation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
