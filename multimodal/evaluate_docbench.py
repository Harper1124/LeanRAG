from __future__ import annotations

import argparse
from pathlib import Path

from .docbench_loader import load_docbench
from .io_utils import write_jsonl
from .mm_query import _install_default_model_funcs, _load_config, query_mm_graph


def run_docbench_eval(
    dataset_dir: str,
    working_root: str,
    output_file: str,
    limit: int | None = None,
    config_file: str = "config.yaml",
) -> None:
    samples = [sample for sample in load_docbench(dataset_dir) if sample.get("question")]
    if limit is not None:
        samples = samples[:limit]
    full_config = _load_config(config_file)
    mm_defaults = full_config.get("multimodal", {})
    rows = []
    for sample in samples:
        working_dir = Path(working_root) / sample["doc_id"]
        if not working_dir.exists():
            rows.append(_missing_workspace_row(sample, working_dir))
            continue
        config = dict(mm_defaults)
        _install_default_model_funcs(config, full_config)
        config.update({
            "working_dir": str(working_dir),
            "chunks_file": str(working_dir / "leanrag_chunk.json"),
            "topk": 10,
            "level_mode": 2,
            "text_topk": 5,
            "max_images_per_query": 4,
            "max_tables_per_query": 4,
            "answer_with_vlm_when_media": True,
        })
        prediction, trace = query_mm_graph(config, None, sample["question"], doc_id=sample["doc_id"])
        rows.append(
            {
                "doc_id": sample["doc_id"],
                "question_id": sample["question_id"],
                "question": sample["question"],
                "gold_answer": sample.get("answer", ""),
                "prediction": prediction,
                "text_evidence": trace.get("text_evidence", []),
                "visual_evidence": trace.get("visual_evidence", []),
                "table_evidence": trace.get("table_evidence", []),
                "trace": trace,
            }
        )
    write_jsonl(rows, output_file)


def _missing_workspace_row(sample: dict, working_dir: Path) -> dict:
    return {
        "doc_id": sample["doc_id"],
        "question_id": sample.get("question_id", ""),
        "question": sample.get("question", ""),
        "gold_answer": sample.get("answer", ""),
        "prediction": "",
        "text_evidence": [],
        "visual_evidence": [],
        "table_evidence": [],
        "trace": {"error": f"working_dir not found: {working_dir}"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MM-LeanRAG on a DocBench subset.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_root", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_docbench_eval(args.dataset_dir, args.working_root, args.output_file, args.limit, args.config)
    print(f"Wrote predictions and evidence trace to {args.output_file}")


if __name__ == "__main__":
    main()
