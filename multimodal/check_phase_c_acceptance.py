from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .io_utils import read_json, read_jsonl, write_json


REQUIRED_ARTIFACTS = (
    "mm_media.json",
    "processed_media.json",
    "mm_chunk.json",
    "entity.jsonl",
    "relation.jsonl",
    "phase3/media_semantic_units.jsonl",
    "phase3/media_graph_chunks.json",
    "phase3/legacy_media_graph/entity.jsonl",
    "phase3/legacy_media_graph/relation.jsonl",
    "phase3/merged_graph/entity.jsonl",
    "phase3/merged_graph/relation.jsonl",
    "phase3/media_graph_manifest.json",
    "all_entities.json",
    "community.json",
    "generate_relations.json",
    "milvus_demo.db",
    "mm_nodes.jsonl",
    "mm_edges_seed.jsonl",
    "mm_edges.jsonl",
)

LAYOUT_RELATION_TYPES = {
    "document_contains_page", "page_contains_node", "text_caption_of_media", "text_refers_to_media",
    "entity_link_media", "media_near_text", "media_same_page_media", "page_next_page", "page_prev_page",
}


def check_phase_c_acceptance(working_dir: str | Path, eval_dir: str | Path | None = None) -> dict[str, Any]:
    working = Path(working_dir).resolve()
    missing = [name for name in REQUIRED_ARTIFACTS if not (working / name).exists()]
    checks: dict[str, Any] = {
        "required_artifacts": _check(not missing, missing=missing),
    }
    if missing:
        return _result(working, checks, {})

    manifest = read_json(working / "phase3/media_graph_manifest.json")
    stages = manifest.get("stages") or {}
    checks["manifest_completed"] = _check(
        manifest.get("status") == "completed"
        and (stages.get("hierarchy") or {}).get("status") == "completed"
        and (stages.get("hierarchy") or {}).get("input") == "phase3/merged_graph",
        status=manifest.get("status"),
        hierarchy=stages.get("hierarchy"),
    )
    changed_inputs = _changed_manifest_inputs(working, manifest)
    checks["root_text_graph_unchanged"] = _check(not changed_inputs, changed=changed_inputs)

    media = read_json(working / "mm_media.json")
    chunks = read_json(working / "mm_chunk.json")
    graph_chunks = read_json(working / "phase3/media_graph_chunks.json")
    entities = read_jsonl(working / "phase3/merged_graph/entity.jsonl")
    relations = read_jsonl(working / "phase3/merged_graph/relation.jsonl")
    media_ids = {str(item.get("media_id")) for item in media if item.get("media_id")}
    text_ids = {str(item.get("hash_code")) for item in chunks if item.get("hash_code")}
    invalid_graph_chunks = [
        item.get("hash_code") for item in graph_chunks
        if not item.get("hash_code") or str(item.get("hash_code")) not in media_ids or not str(item.get("text") or "").strip()
    ]
    checks["graph_chunks_resolve_to_media"] = _check(not invalid_graph_chunks, invalid=invalid_graph_chunks)

    duplicate_source_rows = []
    mixed_source_entities = []
    for entity in entities:
        sources = [item for item in str(entity.get("source_id") or "").split("|") if item]
        if len(sources) != len(set(sources)):
            duplicate_source_rows.append(entity.get("entity_name"))
        if set(sources) & text_ids and set(sources) & media_ids:
            mixed_source_entities.append(entity.get("entity_name"))
    checks["source_ids_stable"] = _check(not duplicate_source_rows, duplicate_entities=duplicate_source_rows)
    checks["mixed_text_media_entities"] = _check(
        bool(mixed_source_entities), count=len(mixed_source_entities), examples=mixed_source_entities[:10]
    )

    layout_leaks = [
        [row.get("src_tgt"), row.get("tgt_src"), row.get("relation_type") or row.get("source")]
        for row in relations
        if str(row.get("relation_type") or row.get("source") or "").strip().lower() in LAYOUT_RELATION_TYPES
    ]
    checks["no_layout_relations_in_merged_graph"] = _check(not layout_leaks, leaks=layout_leaks[:20])

    bad_media_paths = []
    table_missing_fields = []
    for item in media:
        media_type = str(item.get("mapped_type") or item.get("modality") or item.get("type") or "").lower()
        if media_type in {"image", "chart"}:
            path = str(item.get("path") or "")
            if not path or not _media_path_exists(working, path):
                bad_media_paths.append({"media_id": item.get("media_id"), "path": path})
        if media_type == "table":
            missing_fields = [field for field in ("table_html", "table_markdown") if field not in item]
            if missing_fields:
                table_missing_fields.append({"media_id": item.get("media_id"), "missing": missing_fields})
    checks["image_chart_paths_accessible"] = _check(not bad_media_paths, invalid=bad_media_paths[:20])
    checks["table_structure_fields_preserved"] = _check(not table_missing_fields, invalid=table_missing_fields[:20])

    evaluation = _check_evaluation(Path(eval_dir).resolve()) if eval_dir else {}
    if evaluation:
        checks["query_evaluation"] = _check(evaluation.pop("ok"), **evaluation)
    details = {
        "counts": {
            "media": len(media), "graph_chunks": len(graph_chunks), "merged_entities": len(entities),
            "merged_relations": len(relations), "mixed_source_entities": len(mixed_source_entities),
        },
    }
    return _result(working, checks, details)


def _check_evaluation(eval_dir: Path) -> dict[str, Any]:
    names = ("mmlongbench_doc_predictions.jsonl", "mmlongbench_doc_traces.jsonl", "mmlongbench_doc_scores.json")
    missing = [name for name in names if not (eval_dir / name).is_file()]
    if missing:
        return {"ok": False, "missing": missing}
    predictions = read_jsonl(eval_dir / names[0])
    traces = read_jsonl(eval_dir / names[1])
    scores = read_json(eval_dir / names[2])
    trace_errors = []
    vlm_calls = []
    unresolved = 0
    for row in traces:
        trace = row.get("trace") or {}
        if trace.get("failure_stage"):
            trace_errors.append({"question_id": row.get("question_id"), "failure_stage": trace.get("failure_stage")})
        vlm_calls.extend(trace.get("vlm_calls") or [])
        unresolved += int((((trace.get("source_resolution") or {}).get("counts") or {}).get("unresolved") or 0))
    failed_vlm = [call.get("error") or "empty vlm answer" for call in vlm_calls if call.get("error") or not call.get("vlm_answer")]
    overall = ((scores.get("summary") or {}).get("overall") or {})
    ok = bool(predictions) and len(predictions) == len(traces) and not trace_errors and not failed_vlm and unresolved == 0
    return {
        "ok": ok,
        "predictions": len(predictions),
        "traces": len(traces),
        "trace_errors": trace_errors,
        "vlm_calls": len(vlm_calls),
        "failed_vlm_calls": failed_vlm,
        "unresolved_source_ids": unresolved,
        "quality_baseline": {
            key: overall.get(key) for key in (
                "answer_score", "official_raw_answer_score", "official_extracted_answer_score",
                "exact_match", "page_hit", "page_recall", "page_recall_near",
            )
        },
    }


def _changed_manifest_inputs(working: Path, manifest: dict[str, Any]) -> list[str]:
    changed = []
    inputs = manifest.get("inputs") or {}
    for name in ("entity.jsonl", "relation.jsonl"):
        expected = (inputs.get(name) or {}).get("sha256")
        path = working / name
        if not expected or not path.is_file() or _sha256(path) != expected:
            changed.append(name)
    return changed


def _media_path_exists(working: Path, value: str) -> bool:
    path = Path(value)
    return path.is_file() or (not path.is_absolute() and (working / path).is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(ok: bool, **details: Any) -> dict[str, Any]:
    return {"ok": bool(ok), **details}


def _result(working: Path, checks: dict[str, Any], details: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "passed" if checks and all(item.get("ok") for item in checks.values()) else "failed",
        "working_dir": str(working),
        "checks": checks,
        **details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase C acceptance audit for MM-LeanRAG artifacts.")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()
    result = check_phase_c_acceptance(args.working_dir, args.eval_dir)
    if args.output_file:
        write_json(result, args.output_file)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
