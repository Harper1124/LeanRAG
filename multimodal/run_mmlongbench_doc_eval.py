from __future__ import annotations

import argparse
import json
from pathlib import Path

from .build_docbench import build_docbench, _load_config
from .evaluate_docbench import run_docbench_eval
from .prepare_mmlongbench_doc import prepare_mmlongbench_doc
from .score_mmlongbench_doc import score_mmlongbench_doc


def run_mmlongbench_doc_eval(
    dataset_dir: str,
    working_root: str,
    output_dir: str,
    source: str = "github",
    github_repo: str = "mayubo2333/MMLongBench-Doc",
    repo_id: str = "yubo2333/MMLongBench-Doc",
    local_data_file: str | None = None,
    local_documents_dir: str | None = None,
    prepare: bool = False,
    build: bool = False,
    predict: bool = True,
    score: bool = True,
    max_docs: int | None = None,
    limit: int | None = None,
    config_file: str = "config.yaml",
    skip_graph: bool = False,
    force: bool = False,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full_config = _load_config(config_file)
    if prepare:
        prepare_mmlongbench_doc(
            dataset_dir,
            source=source,
            github_repo=github_repo,
            repo_id=repo_id,
            max_docs=max_docs,
            local_data_file=local_data_file,
            local_documents_dir=local_documents_dir,
        )
    if build:
        mm_config = full_config.get("multimodal", {})
        build_docbench(
            docbench_dir=dataset_dir,
            working_root=working_root,
            build_graph=not skip_graph,
            force=force,
            use_media_caption=bool(mm_config.get("use_media_caption", False)),
            use_table_summary=bool(mm_config.get("use_table_summary", False)),
            model_config=full_config,
        )

    predictions_file = output / "mmlongbench_doc_predictions.jsonl"
    scores_file = output / "mmlongbench_doc_scores.json"
    result = {"predictions_file": str(predictions_file), "scores_file": str(scores_file)}
    if predict:
        run_docbench_eval(dataset_dir, working_root, str(predictions_file), limit=limit, config_file=config_file)
    if score:
        result["scores"] = score_mmlongbench_doc(dataset_dir, str(predictions_file), str(scores_file))["summary"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end MMLongBench-Doc evaluation for MM-LeanRAG.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source", choices=["github", "hf"], default="github")
    parser.add_argument("--github_repo", default="mayubo2333/MMLongBench-Doc")
    parser.add_argument("--repo_id", default="yubo2333/MMLongBench-Doc")
    parser.add_argument("--local_data_file", default=None)
    parser.add_argument("--local_documents_dir", default=None)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--no_predict", action="store_true")
    parser.add_argument("--no_score", action="store_true")
    parser.add_argument("--skip_graph", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    result = run_mmlongbench_doc_eval(
        dataset_dir=args.dataset_dir,
        working_root=args.working_root,
        output_dir=args.output_dir,
        source=args.source,
        github_repo=args.github_repo,
        repo_id=args.repo_id,
        local_data_file=args.local_data_file,
        local_documents_dir=args.local_documents_dir,
        prepare=args.prepare,
        build=args.build,
        predict=not args.no_predict,
        score=not args.no_score,
        max_docs=args.max_docs,
        limit=args.limit,
        config_file=args.config,
        skip_graph=args.skip_graph,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
