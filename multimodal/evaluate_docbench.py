from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .docbench_loader import load_docbench
from .mm_query import _install_default_model_funcs, _load_config, query_mm_graph


def run_docbench_eval(
    dataset_dir: str,
    working_root: str,
    output_file: str,
    trace_file: str | None = None,
    limit: int | None = None,
    config_file: str = "config.yaml",
) -> None:
    samples = [sample for sample in load_docbench(dataset_dir) if sample.get("question")]
    if not samples:
        raise ValueError(
            f"No QA samples were loaded from {dataset_dir}. "
            "Check that qa.jsonl is non-empty, valid JSONL, and contains a question field."
        )
    if limit is not None:
        samples = samples[:limit]

    full_config = _load_config(config_file)
    mm_defaults = full_config.get("multimodal", {})
    output_path = Path(output_file)
    trace_path = Path(trace_file) if trace_file else _default_trace_file(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as pred_f, trace_path.open("w", encoding="utf-8") as trace_f:
        for sample in samples:
            working_dir, attempted_dirs = _resolve_working_dir(Path(working_root), sample)
            if working_dir is not None:
                full_row = _query_sample(sample, working_dir, mm_defaults, full_config)
            else:
                full_row = _missing_workspace_row(sample, attempted_dirs)

            trace = full_row.get("trace") or {}
            pred_row = compact_row(full_row)
            if isinstance(trace, dict) and trace.get("error"):
                pred_row["trace_error"] = trace.get("error")
            trace_row = {
                "doc_id": full_row.get("doc_id"),
                "question_id": full_row.get("question_id"),
                "trace": trace,
            }
            pred_f.write(json.dumps(pred_row, ensure_ascii=False) + "\n")
            trace_f.write(json.dumps(trace_row, ensure_ascii=False) + "\n")


def _query_sample(sample: dict[str, Any], working_dir: Path, mm_defaults: dict, full_config: dict) -> dict[str, Any]:
    config = dict(mm_defaults)
    _install_default_model_funcs(config, full_config)
    config.update({
        "working_dir": str(working_dir),
        "chunks_file": str(working_dir / "leanrag_chunk.json"),
        "answer_format": (sample.get("metadata") or {}).get("answer_format"),
        "topk": 10,
        "level_mode": 2,
        "text_topk": 5,
        "max_images_per_query": 4,
        "max_tables_per_query": 4,
        "global_max_images_per_query": 64,
        "global_max_tables_per_query": 24,
        "answer_with_vlm_when_media": True,
    })
    prediction, trace = query_mm_graph(
        config,
        None,
        sample["question"],
        doc_id=_workspace_doc_id(working_dir, sample["doc_id"]),
    )
    return _prediction_row(sample, prediction, trace)


def _prediction_row(sample: dict[str, Any], prediction: str, trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_id": sample["doc_id"],
        "question_id": sample["question_id"],
        "question": sample["question"],
        "gold_answer": sample.get("answer", ""),
        "answer_format": _metadata_first(sample, "answer_format", "answer_type"),
        "evidence_source": _metadata_first(
            sample,
            "evidence_source",
            "evidence_sources",
            "source",
            "source_type",
            "evidence_type",
        ),
        "prediction": prediction,
        "text_evidence": trace.get("text_evidence", []),
        "visual_evidence": trace.get("visual_evidence", []),
        "table_evidence": trace.get("table_evidence", []),
        "trace": trace,
    }


def _missing_workspace_row(sample: dict[str, Any], attempted_dirs: list[Path]) -> dict[str, Any]:
    return {
        "doc_id": sample["doc_id"],
        "question_id": sample.get("question_id", ""),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("answer", ""),
        "answer_format": _metadata_first(sample, "answer_format", "answer_type"),
        "evidence_source": _metadata_first(
            sample,
            "evidence_source",
            "evidence_sources",
            "source",
            "source_type",
            "evidence_type",
        ),
        "prediction": "",
        "text_evidence": [],
        "visual_evidence": [],
        "table_evidence": [],
        "trace": {"error": "working_dir not found; tried: " + ", ".join(str(path) for path in attempted_dirs)},
    }


def _resolve_working_dir(working_root: Path, sample: dict[str, Any]) -> tuple[Path | None, list[Path]]:
    """Support both benchmark-filename and filename-stem workspace layouts."""
    raw_doc_id = str(sample.get("doc_id") or "").strip()
    pdf_name = Path(str(sample.get("pdf_path") or "")).name
    names = [raw_doc_id, Path(raw_doc_id).stem, pdf_name, Path(pdf_name).stem]
    attempted: list[Path] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        candidate = working_root / name
        attempted.append(candidate)
        if candidate.is_dir():
            return candidate, attempted
    return None, attempted


def _workspace_doc_id(working_dir: Path, fallback: str) -> str:
    manifest_path = working_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        value = manifest.get("doc_id") if isinstance(manifest, dict) else None
        if value not in (None, ""):
            return str(value)
    except (OSError, ValueError, TypeError):
        pass
    # The directory name is the best representation of ids used in its graph
    # artifacts when an old manifest has no doc_id.
    return working_dir.name or str(fallback)


def _metadata_first(sample: dict[str, Any], *keys: str) -> object:
    metadata = sample.get("metadata") or {}
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return value
    return None


def _default_trace_file(output_path: Path) -> Path:
    if output_path.name == "mmlongbench_doc_predictions.jsonl":
        return output_path.with_name("mmlongbench_doc_traces.jsonl")
    if output_path.suffix == ".jsonl":
        return output_path.with_name(f"{output_path.stem}_traces.jsonl")
    return output_path.with_suffix(output_path.suffix + ".traces.jsonl")


def compact_row(
    row: dict[str, Any],
    text_limit: int = 500,
    media_text_limit: int = 500,
    max_text_evidence: int = 3,
    max_visual_evidence: int = 3,
    max_table_evidence: int = 3,
) -> dict[str, Any]:
    return _drop_empty({
        "doc_id": row.get("doc_id"),
        "question_id": row.get("question_id"),
        "question": row.get("question"),
        "gold_answer": row.get("gold_answer"),
        "answer_format": row.get("answer_format"),
        "evidence_source": row.get("evidence_source") or row.get("evidence_sources"),
        "prediction": row.get("prediction"),
        "text_evidence": [
            _slim_text_evidence(item, text_limit)
            for item in _as_list(row.get("text_evidence"))[:max_text_evidence]
        ],
        "visual_evidence": [
            _slim_media_evidence(item, media_text_limit)
            for item in _as_list(row.get("visual_evidence"))[:max_visual_evidence]
        ],
        "table_evidence": [
            _slim_media_evidence(item, media_text_limit)
            for item in _as_list(row.get("table_evidence"))[:max_table_evidence]
        ],
    })


def _slim_text_evidence(item: Any, text_limit: int) -> Any:
    if not isinstance(item, dict):
        return item
    return _drop_empty({
        "chunk_id": item.get("chunk_id"),
        "doc_id": item.get("doc_id"),
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "section_title": item.get("section_title"),
        "score": item.get("score"),
        "text": _truncate(item.get("text") or item.get("text_for_embedding"), text_limit),
    })


def _slim_media_evidence(item: Any, text_limit: int) -> Any:
    if not isinstance(item, dict):
        return item
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    raw_ref = item.get("raw_ref") if isinstance(item.get("raw_ref"), dict) else {}
    return _drop_empty({
        "media_id": item.get("media_id") or raw_ref.get("media_id"),
        "doc_id": item.get("doc_id"),
        "modality": item.get("modality") or metadata.get("modality"),
        "media_type": item.get("media_type") or metadata.get("media_type"),
        "page": _first_present(item.get("page"), item.get("page_id")),
        "path": item.get("path") or raw_ref.get("path"),
        "score": item.get("score"),
        "caption": _truncate(item.get("caption") or item.get("text_for_embedding"), text_limit),
        "summary": _truncate(item.get("summary"), text_limit),
        "ocr_text": _truncate(item.get("ocr_text"), text_limit),
        "table_markdown": _truncate(item.get("table_markdown") or raw_ref.get("table_markdown"), text_limit),
        "table_html": _truncate(item.get("table_html") or raw_ref.get("table_html"), text_limit),
    })


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _drop_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: cleaned
            for key, item in value.items()
            if (cleaned := _drop_empty(item)) not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [cleaned for item in value if (cleaned := _drop_empty(item)) not in (None, "", [], {})]
    return value


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _truncate(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    if max_chars < 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MM-LeanRAG on a DocBench subset.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_root", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    trace_file = _default_trace_file(Path(args.output_file))
    run_docbench_eval(
        args.dataset_dir,
        args.working_root,
        args.output_file,
        trace_file=str(trace_file),
        limit=args.limit,
        config_file=args.config,
    )
    print(f"Wrote compact predictions to {args.output_file}")
    print(f"Wrote full traces to {trace_file}")


if __name__ == "__main__":
    main()
