from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    extract_answers: bool = False,
    evaluation_model: str | None = None,
    evaluation_base_url: str | None = None,
    evaluation_api_key: str = "",
    evaluation_api_key_env: str | None = None,
) -> dict:
    from .build_docbench import _load_config

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    full_config = _load_config(config_file)

    if prepare:
        from .prepare_mmlongbench_doc import prepare_mmlongbench_doc

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
        from .build_docbench import build_docbench

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
    traces_file = output / "mmlongbench_doc_traces.jsonl"
    scores_file = output / "mmlongbench_doc_scores.json"
    result = {
        "predictions_file": str(predictions_file),
        "traces_file": str(traces_file),
        "scores_file": str(scores_file),
    }

    if predict:
        from .evaluate_docbench import run_docbench_eval

        run_docbench_eval(
            dataset_dir,
            working_root,
            str(predictions_file),
            trace_file=str(traces_file),
            limit=limit,
            config_file=config_file,
        )

    if score:
        evaluation_config = _evaluation_config(
            full_config,
            extract_answers,
            evaluation_model,
            evaluation_base_url,
            evaluation_api_key,
            evaluation_api_key_env,
        )
        result["scores"] = score_mmlongbench_doc(
            dataset_dir,
            str(predictions_file),
            str(scores_file),
            extract_answers=extract_answers,
            evaluation_model_config=evaluation_config,
        )["summary"]

    return result


def _evaluation_config(
    full_config: dict[str, Any],
    extract_answers: bool,
    evaluation_model: str | None,
    evaluation_base_url: str | None,
    evaluation_api_key: str,
    evaluation_api_key_env: str | None,
) -> dict[str, Any] | None:
    if not extract_answers:
        return None
    configured = full_config.get("evaluation_model") or full_config.get("evaluation") or {}
    extraction_options = full_config.get("evaluation_extraction") or {}
    deepseek = full_config.get("deepseek", {})
    config = {
        "model": evaluation_model or configured.get("model") or deepseek.get("model"),
        "base_url": evaluation_base_url or configured.get("base_url") or deepseek.get("base_url"),
        "api_key": evaluation_api_key or configured.get("api_key") or deepseek.get("api_key", ""),
        "api_key_env": evaluation_api_key_env or configured.get("api_key_env") or deepseek.get("api_key_env"),
        "temperature": configured.get("temperature", 0.0),
        "max_tokens": configured.get("max_tokens", 256),
    }
    if isinstance(extraction_options, dict):
        config.update(extraction_options)
    if not config["model"] or not config["base_url"]:
        raise ValueError("evaluation model and base_url are required when extract_answers=True")
    return config


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
    parser.add_argument("--extract_answers", action="store_true")
    parser.add_argument("--evaluation_model", default=None)
    parser.add_argument("--evaluation_base_url", default=None)
    parser.add_argument("--evaluation_api_key", default="")
    parser.add_argument("--evaluation_api_key_env", default=None)
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
        extract_answers=args.extract_answers,
        evaluation_model=args.evaluation_model,
        evaluation_base_url=args.evaluation_base_url,
        evaluation_api_key=args.evaluation_api_key,
        evaluation_api_key_env=args.evaluation_api_key_env,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
