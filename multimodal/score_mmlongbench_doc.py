from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .docbench_loader import load_docbench
from .io_utils import read_jsonl, write_json


def score_mmlongbench_doc(dataset_dir: str, predictions_file: str, output_file: str) -> dict:
    gold = {(row["doc_id"], row["question_id"]): row for row in load_docbench(dataset_dir) if row.get("question")}
    predictions = read_jsonl(predictions_file)
    rows = []
    for pred in predictions:
        key = (str(pred.get("doc_id", "")), str(pred.get("question_id", "")))
        sample = gold.get(key)
        if sample is None:
            sample = _match_gold_by_question(gold, pred)
        rows.append(_score_row(sample, pred))

    summary = {
        "overall": _aggregate(rows),
        "by_answer_format": _aggregate_groups(rows, "answer_format"),
        "by_doc_type": _aggregate_groups(rows, "doc_type"),
        "by_evidence_source": _aggregate_multivalue_groups(rows, "evidence_sources"),
        "num_predictions": len(predictions),
        "num_gold": len(gold),
    }
    output = {"summary": summary, "items": rows}
    write_json(output, output_file)
    return output


def _score_row(sample: dict | None, pred: dict) -> dict:
    metadata = (sample or {}).get("metadata", {})
    gold_answer = (sample or {}).get("answer", pred.get("gold_answer", ""))
    prediction = pred.get("prediction", "")
    answer_format = _clean_scalar(metadata.get("answer_format"))
    evidence_pages = _parse_list(metadata.get("evidence_pages"))
    evidence_sources = [_clean_scalar(item) for item in _parse_list(metadata.get("evidence_sources"))]
    retrieved_pages = sorted(_extract_pages(pred))
    answer_metrics = _answer_metrics(gold_answer, prediction, answer_format)
    evidence_metrics = _evidence_metrics(evidence_pages, retrieved_pages)
    return {
        "doc_id": pred.get("doc_id", (sample or {}).get("doc_id", "")),
        "question_id": pred.get("question_id", (sample or {}).get("question_id", "")),
        "question": pred.get("question", (sample or {}).get("question", "")),
        "gold_answer": gold_answer,
        "prediction": prediction,
        "answer_format": answer_format or "Unknown",
        "doc_type": _clean_scalar(metadata.get("doc_type")) or "Unknown",
        "evidence_pages": evidence_pages,
        "retrieved_pages": retrieved_pages,
        "evidence_sources": evidence_sources,
        **answer_metrics,
        **evidence_metrics,
        "trace_error": (pred.get("trace") or {}).get("error"),
    }


def _answer_metrics(gold: Any, pred: Any, answer_format: str | None) -> dict:
    gold_text = _stringify_answer(gold)
    pred_text = _stringify_answer(pred)
    is_unanswerable = _normalize(gold_text) in {"not answerable", "none", "nan", ""}
    if is_unanswerable:
        score = 1.0 if _normalize(pred_text) in {"not answerable", "none", "unknown", "unanswerable", ""} else 0.0
        return {"answer_score": score, "exact_match": score, "token_f1": score, "numeric_match": None, "list_f1": None}

    exact = 1.0 if _normalize(gold_text) == _normalize(pred_text) else 0.0
    token_f1 = _token_f1(gold_text, pred_text)
    numeric_match = _numeric_match(gold_text, pred_text) if answer_format in {"Int", "Float"} else None
    list_f1 = _list_f1(gold, pred_text) if answer_format == "List" else None
    if list_f1 is not None:
        score = list_f1
    elif numeric_match is not None:
        score = numeric_match
    else:
        score = max(exact, token_f1)
    return {"answer_score": score, "exact_match": exact, "token_f1": token_f1, "numeric_match": numeric_match, "list_f1": list_f1}


def _evidence_metrics(gold_pages: list[int], pred_pages: list[int]) -> dict:
    if not gold_pages:
        return {"page_hit": None, "page_precision": None, "page_recall": None, "page_f1": None}
    gold_set = set(gold_pages)
    pred_set = set(pred_pages)
    hit = 1.0 if gold_set & pred_set else 0.0
    precision = len(gold_set & pred_set) / len(pred_set) if pred_set else 0.0
    recall = len(gold_set & pred_set) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"page_hit": hit, "page_precision": precision, "page_recall": recall, "page_f1": f1}


def _extract_pages(pred: dict) -> set[int]:
    pages = set()
    evidence_groups = [pred.get("text_evidence", []), pred.get("visual_evidence", []), pred.get("table_evidence", [])]
    trace = pred.get("trace") or {}
    evidence_groups.extend([trace.get("text_evidence", []), trace.get("visual_evidence", []), trace.get("table_evidence", [])])
    for group in evidence_groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            for key in ("page", "page_start", "page_end"):
                if item.get(key) is not None:
                    pages.add(_safe_int(item[key]))
            if item.get("page_start") is not None and item.get("page_end") is not None:
                start, end = _safe_int(item["page_start"]), _safe_int(item["page_end"])
                if start and end and end >= start and end - start <= 20:
                    pages.update(range(start, end + 1))
    return {page for page in pages if page is not None}


def _aggregate(rows: list[dict]) -> dict:
    return {
        "count": len(rows),
        "answer_score": _mean(rows, "answer_score"),
        "exact_match": _mean(rows, "exact_match"),
        "token_f1": _mean(rows, "token_f1"),
        "numeric_match": _mean(rows, "numeric_match"),
        "list_f1": _mean(rows, "list_f1"),
        "page_hit": _mean(rows, "page_hit"),
        "page_precision": _mean(rows, "page_precision"),
        "page_recall": _mean(rows, "page_recall"),
        "page_f1": _mean(rows, "page_f1"),
        "missing_workspace_rate": sum(1 for row in rows if row.get("trace_error")) / len(rows) if rows else 0.0,
    }


def _aggregate_groups(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row.get(key) or "Unknown"].append(row)
    return {name: _aggregate(items) for name, items in sorted(groups.items())}


def _aggregate_multivalue_groups(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for row in rows:
        values = row.get(key) or ["None"]
        for value in values:
            groups[value or "None"].append(row)
    return {name: _aggregate(items) for name, items in sorted(groups.items())}


def _mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return mean(values) if values else None


def _match_gold_by_question(gold: dict, pred: dict) -> dict | None:
    doc_id = str(pred.get("doc_id", ""))
    question = str(pred.get("question", ""))
    for (gold_doc_id, _), sample in gold.items():
        if gold_doc_id == doc_id and sample.get("question") == question:
            return sample
    return None


def _parse_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except (SyntaxError, ValueError):
        return [text]


def _stringify_answer(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_stringify_answer(item) for item in value)
    if hasattr(value, "tolist"):
        return _stringify_answer(value.tolist())
    if value is None:
        return ""
    return str(value)


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none"} else text


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9.%/-]+", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _token_f1(gold: str, pred: str) -> float:
    gold_tokens = _normalize(gold).split()
    pred_tokens = _normalize(pred).split()
    if not gold_tokens or not pred_tokens:
        return 1.0 if gold_tokens == pred_tokens else 0.0
    common = 0
    pred_counts = defaultdict(int)
    for token in pred_tokens:
        pred_counts[token] += 1
    for token in gold_tokens:
        if pred_counts[token] > 0:
            common += 1
            pred_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _numeric_match(gold: str, pred: str) -> float:
    gold_nums = _numbers(gold)
    pred_nums = _numbers(pred)
    if not gold_nums or not pred_nums:
        return 0.0
    for gold_num in gold_nums:
        for pred_num in pred_nums:
            if _close_number(gold_num, pred_num):
                return 1.0
    return 0.0


def _numbers(text: str) -> list[float]:
    nums = []
    for raw in re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?", text):
        scale = 0.01 if raw.endswith("%") else 1.0
        nums.append(float(raw.rstrip("%").replace(",", "")) * scale)
    return nums


def _close_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-3, abs_tol=1e-3) or math.isclose(left, right * 0.01, rel_tol=1e-3, abs_tol=1e-3)


def _list_f1(gold: Any, pred: str) -> float:
    gold_items = [_normalize(_stringify_answer(item)) for item in _parse_list(gold)]
    pred_items = [_normalize(item) for item in re.split(r"[,;\n]|\band\b", pred) if _normalize(item)]
    if not gold_items:
        return 0.0
    matched = 0
    used = set()
    for gold_item in gold_items:
        for idx, pred_item in enumerate(pred_items):
            if idx in used:
                continue
            if gold_item == pred_item or gold_item in pred_item or pred_item in gold_item:
                matched += 1
                used.add(idx)
                break
    precision = matched / len(pred_items) if pred_items else 0.0
    recall = matched / len(gold_items)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MM-LeanRAG predictions on MMLongBench-Doc.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()
    output = score_mmlongbench_doc(args.dataset_dir, args.predictions_file, args.output_file)
    print(json.dumps(output["summary"]["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
