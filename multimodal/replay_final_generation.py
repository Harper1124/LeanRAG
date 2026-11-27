from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .generation.final_generator import generate_final_answer
from .mm_query import _load_config
from .openai_clients import make_chat_func


def _load_question(path: Path, question_id: str) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("question_id")) == question_id:
                return row
    raise ValueError(f"Question {question_id!r} not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare final prompts using frozen retrieval and VLM results.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--audit-only", action="store_true", help="Check prompt contents without calling a model.")
    args = parser.parse_args()

    prediction = _load_question(args.predictions, args.question_id)
    trace_row = _load_question(args.traces, args.question_id)
    if prediction.get("doc_id") != trace_row.get("doc_id"):
        parser.error("The prediction and trace refer to different documents.")
    trace = trace_row["trace"]
    full_config = _load_config(args.config)
    config = dict(full_config.get("multimodal") or {})
    config["answer_format"] = prediction.get("answer_format")
    if args.audit_only:
        config["use_llm_func"] = lambda *_: ""
    else:
        chat_config = dict(full_config.get("deepseek") or {})
        if not chat_config.get("base_url") or not chat_config.get("model"):
            parser.error("The config needs deepseek.base_url and deepseek.model.")
        chat_config.setdefault("api_key_env", "DASHSCOPE_API_KEY")
        config["use_llm_func"] = make_chat_func({**chat_config, "temperature": 0.0})

    result = {
        "question_id": args.question_id,
        "question": prediction["question"],
        "gold_answer": prediction.get("gold_answer"),
        "recorded_answer": prediction.get("prediction"),
        "runs": {},
    }
    for label, reserve in (("baseline", False), ("reserve_vlm_evidence", True)):
        config["generation"] = {**(config.get("generation") or {}), "reserve_vlm_evidence": reserve}
        answer, final_trace = generate_final_answer(
            prediction["question"],
            trace["evidence_package"],
            trace["answer_plan"],
            trace.get("vlm_calls", []),
            trace.get("table_reasoner_calls", []),
            config,
        )
        result["runs"][label] = {
            "prompt_budget": final_trace.get("prompt_budget"),
            "postprocess": final_trace.get("postprocess"),
            "error": final_trace.get("error"),
        }
        if not args.audit_only:
            result["runs"][label]["answer"] = answer
            result["runs"][label]["used_llm"] = final_trace.get("used_llm", False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
