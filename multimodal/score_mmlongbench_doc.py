from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from math import isclose
from statistics import mean
from typing import Any

from .docbench_loader import load_docbench
from .io_utils import read_jsonl, write_json
from .mmlongbench_answer_extraction import ExtractedAnswer, make_answer_extractor


def score_mmlongbench_doc(
    dataset_dir: str,
    predictions_file: str,
    output_file: str,
    extract_answers: bool = False,
    evaluation_model_config: dict[str, Any] | None = None,
) -> dict:
    gold = {(row["doc_id"], row["question_id"]): row for row in load_docbench(dataset_dir) if row.get("question")}
    predictions = read_jsonl(predictions_file)
    answer_extractor = None
    if extract_answers:
        if not evaluation_model_config:
            raise ValueError("evaluation_model_config is required when extract_answers=True")
        answer_extractor = make_answer_extractor(evaluation_model_config)

    rows = []
    for pred in predictions:
        key = (str(pred.get("doc_id", "")), str(pred.get("question_id", "")))
        sample = gold.get(key)
        if sample is None:
            sample = _match_gold_by_question(gold, pred)
        rows.append(_score_row(sample, pred, answer_extractor))

    summary = {
        "overall": _aggregate(rows),
        "by_answer_format": _aggregate_groups(rows, "answer_format"),
        "by_extracted_answer_format": _aggregate_groups(rows, "extracted_answer_format"),
        "by_doc_type": _aggregate_groups(rows, "doc_type"),
        "by_evidence_source": _aggregate_multivalue_groups(rows, "evidence_sources"),
        "num_predictions": len(predictions),
        "num_gold": len(gold),
        "used_answer_extraction": bool(extract_answers),
    }
    output = {"summary": summary, "items": rows}
    write_json(output, output_file)
    return output


def _score_row(sample: dict | None, pred: dict, answer_extractor=None) -> dict:
    metadata = (sample or {}).get("metadata", {})
    gold_answer = (sample or {}).get("answer", pred.get("gold_answer", ""))
    prediction = str(pred.get("prediction", ""))
    answer_format = _clean_scalar(metadata.get("answer_format"))
    extracted = _extract_prediction_answer(sample, pred, prediction, answer_extractor)
    scored_prediction = extracted.answer if extracted else prediction
    extracted_format = extracted.answer_format if extracted else _clean_scalar(pred.get("extracted_answer_format"))
    evidence_pages = _parse_list(metadata.get("evidence_pages"))
    evidence_sources = [_clean_scalar(item) for item in _parse_list(metadata.get("evidence_sources"))]
    retrieved_pages = sorted(_extract_pages(pred))
    answer_metrics = _answer_metrics(gold_answer, scored_prediction, answer_format)
    evidence_metrics = _evidence_metrics(evidence_pages, retrieved_pages)
    return {
        "doc_id": pred.get("doc_id", (sample or {}).get("doc_id", "")),
        "question_id": pred.get("question_id", (sample or {}).get("question_id", "")),
        "question": pred.get("question", (sample or {}).get("question", "")),
        "gold_answer": gold_answer,
        "prediction": prediction,
        "extracted_answer": scored_prediction if extracted else None,
        "extracted_answer_format": extracted_format,
        "answer_extraction_raw": extracted.raw_response if extracted else None,
        "answer_extraction_error": extracted.error if extracted else None,
        "scored_prediction": scored_prediction,
        "answer_format": answer_format or "Unknown",
        "doc_type": _clean_scalar(metadata.get("doc_type")) or "Unknown",
        "evidence_pages": evidence_pages,
        "retrieved_pages": retrieved_pages,
        "evidence_sources": evidence_sources,
        **answer_metrics,
        **evidence_metrics,
        "trace_error": (pred.get("trace") or {}).get("error"),
    }


def _extract_prediction_answer(sample: dict | None, pred: dict, prediction: str, answer_extractor) -> ExtractedAnswer | None:
    if pred.get("extracted_answer") is not None:
        return ExtractedAnswer(
            answer=str(pred.get("extracted_answer")),
            answer_format=_clean_scalar(pred.get("extracted_answer_format")) or "Str",
            raw_response=str(pred.get("answer_extraction_raw") or ""),
            error=_clean_scalar(pred.get("answer_extraction_error")),
        )
    if answer_extractor is None:
        return None
    question = str(pred.get("question") or (sample or {}).get("question") or "")
    return answer_extractor(question, prediction)


def _answer_metrics(gold: Any, pred: Any, answer_format: str | None) -> dict:
    gold_text = _stringify_answer(gold)
    pred_text = _stringify_answer(pred)
    official_format = _official_answer_format(answer_format)
    score = _mmlongbench_eval_score(gold, pred_text, official_format)
    exact = 1.0 if _normalize(gold_text) == _normalize(pred_text) else 0.0
    token_f1 = _token_f1(gold_text, pred_text)
    return {
        "answer_score": score,
        "exact_match": exact,
        "token_f1": token_f1,
        "numeric_match": score if official_format in {"Int", "Float"} else None,
        "list_f1": score if official_format == "List" else None,
        "anls": score if official_format in {"Str", "None"} else None,
        "official_answer_format": official_format,
    }


def _mmlongbench_eval_score(gt: Any, pred: Any, answer_type: str) -> float:
    if answer_type == "Int":
        try:
            gt_value = int(float(str(gt).strip().rstrip("%").strip()))
            pred_value = int(float(str(pred).strip().rstrip("%").strip()))
            return float(gt_value == pred_value)
        except Exception:
            return 0.0
    if answer_type == "Float":
        try:
            gt_value = float(get_clean_string(str(gt)))
            pred_value = float(get_clean_string(str(pred)))
        except Exception:
            return 0.0
        return float(is_float_equal(gt_value, pred_value, include_percentage=True, is_close=True))
    if answer_type in {"Str", "None"}:
        gt_clean = get_clean_string(gt)
        pred_clean = get_clean_string(pred)
        if is_exact_match(gt_clean):
            return float(gt_clean == pred_clean)
        return float(anls_compute(gt_clean, pred_clean))
    return _mmlongbench_list_score(gt, pred)


def _mmlongbench_list_score(gt: Any, pred: Any) -> float:
    gt_items = _parse_list_literal(gt)
    pred_items = _parse_list_literal(pred)
    if not isinstance(gt_items, list):
        gt_items = [gt_items]
    if not isinstance(pred_items, list):
        pred_items = [pred_items]
    if len(gt_items) != len(pred_items):
        return 0.0
    if not gt_items:
        return 0.0
    gt_clean = sorted([get_clean_string(item) for item in gt_items])
    pred_clean = sorted([get_clean_string(item) for item in pred_items])
    if isfloat(gt_clean[0]) or is_exact_match(gt_clean[0]):
        return float("-".join(gt_clean) == "-".join(pred_clean))
    return float(min(anls_compute(gt_value, pred_value) for gt_value, pred_value in zip(gt_clean, pred_clean)))


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    dist = levenshtein_distance(str(groundtruth), str(prediction))
    length = max(len(str(groundtruth).upper()), len(str(prediction).upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    return 0.0 if anls <= threshold else anls


def is_float_equal(reference: float, prediction: float, include_percentage: bool = False, is_close: bool = False) -> bool:
    def get_precision(value: float) -> int:
        precision = 3
        if "." in str(value):
            precision = len(str(value).split(".")[-1])
        return precision

    candidates = [reference / 100, reference, reference * 100] if include_percentage else [reference]
    for item in candidates:
        try:
            if is_close and isclose(item, prediction, rel_tol=0.01):
                return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(value: Any) -> str:
    text = str(value).lower().strip()
    for suffix in ("miles", "mile", "million"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    text = re.sub(r"\s*\([^)]*\)", "", text).strip()
    text = re.sub(r"^['\"]|['\"]$", "", text).strip()
    return text.strip().lstrip("$").strip().rstrip("%").strip()


def is_exact_match(value: str) -> bool:
    text = str(value)
    if "https://" in text:
        return True
    if text.endswith(".py") or text.endswith("ipynb"):
        return True
    if text.startswith("page"):
        return True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", text):
        return True
    if "a.m." in text or "p.m." in text:
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", text):
        return True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text):
        return True
    return False


def isfloat(value: Any) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _official_answer_format(answer_format: str | None) -> str:
    text = str(answer_format or "").strip().lower()
    if text in {"int", "integer"}:
        return "Int"
    if text in {"float", "number"}:
        return "Float"
    if text in {"list", "array"}:
        return "List"
    if text in {"none"}:
        return "None"
    return "Str"


def _evidence_metrics(gold_pages: list[int], pred_pages: list[int]) -> dict:
    if not gold_pages:
        return {
            "page_hit": None,
            "page_precision": None,
            "page_recall": None,
            "page_f1": None,
            "page_hit_near": None,
            "page_precision_near": None,
            "page_recall_near": None,
            "page_f1_near": None,
            "page_min_abs_delta": None,
        }
    gold_set = set(gold_pages)
    pred_set = set(pred_pages)
    hit = 1.0 if gold_set & pred_set else 0.0
    precision = len(gold_set & pred_set) / len(pred_set) if pred_set else 0.0
    recall = len(gold_set & pred_set) / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    near_pred_set = _expand_pages(pred_set, tolerance=1)
    near_gold_set = _expand_pages(gold_set, tolerance=1)
    near_hit = 1.0 if gold_set & near_pred_set else 0.0
    near_precision = len(pred_set & near_gold_set) / len(pred_set) if pred_set else 0.0
    near_recall = len(gold_set & near_pred_set) / len(gold_set)
    near_f1 = 2 * near_precision * near_recall / (near_precision + near_recall) if near_precision + near_recall else 0.0
    min_delta = min((abs(gold_page - pred_page) for gold_page in gold_set for pred_page in pred_set), default=None)
    return {
        "page_hit": hit,
        "page_precision": precision,
        "page_recall": recall,
        "page_f1": f1,
        "page_hit_near": near_hit,
        "page_precision_near": near_precision,
        "page_recall_near": near_recall,
        "page_f1_near": near_f1,
        "page_min_abs_delta": min_delta,
    }


def _expand_pages(pages: set[int], tolerance: int) -> set[int]:
    expanded = set()
    for page in pages:
        expanded.update(range(max(1, page - tolerance), page + tolerance + 1))
    return expanded


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
        "anls": _mean(rows, "anls"),
        "page_hit": _mean(rows, "page_hit"),
        "page_precision": _mean(rows, "page_precision"),
        "page_recall": _mean(rows, "page_recall"),
        "page_f1": _mean(rows, "page_f1"),
        "page_hit_near": _mean(rows, "page_hit_near"),
        "page_precision_near": _mean(rows, "page_precision_near"),
        "page_recall_near": _mean(rows, "page_recall_near"),
        "page_f1_near": _mean(rows, "page_f1_near"),
        "page_min_abs_delta": _mean(rows, "page_min_abs_delta"),
        "missing_workspace_rate": sum(1 for row in rows if row.get("trace_error")) / len(rows) if rows else 0.0,
        "answer_extraction_error_rate": sum(1 for row in rows if row.get("answer_extraction_error")) / len(rows) if rows else 0.0,
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
    parsed = _parse_list_literal(value)
    return parsed if isinstance(parsed, list) else [parsed]


def _parse_list_literal(value: Any) -> Any:
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
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [item.strip() for item in text.split(",") if item.strip()] if "," in text else text


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


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _evaluation_model_config_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.extract_answers:
        return None
    config = {
        "model": args.evaluation_model,
        "base_url": args.evaluation_base_url,
        "api_key": args.evaluation_api_key,
        "api_key_env": args.evaluation_api_key_env,
        "temperature": args.evaluation_temperature,
        "max_tokens": args.evaluation_max_tokens,
    }
    if not config["model"] or not config["base_url"]:
        raise ValueError("--evaluation_model and --evaluation_base_url are required with --extract_answers")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Score MM-LeanRAG predictions on MMLongBench-Doc.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--extract_answers", action="store_true", help="Use an evaluation model to extract canonical answers before scoring.")
    parser.add_argument("--evaluation_model", default=None)
    parser.add_argument("--evaluation_base_url", default=None)
    parser.add_argument("--evaluation_api_key", default="")
    parser.add_argument("--evaluation_api_key_env", default=None)
    parser.add_argument("--evaluation_temperature", type=float, default=0.0)
    parser.add_argument("--evaluation_max_tokens", type=int, default=256)
    args = parser.parse_args()
    output = score_mmlongbench_doc(
        args.dataset_dir,
        args.predictions_file,
        args.output_file,
        extract_answers=args.extract_answers,
        evaluation_model_config=_evaluation_model_config_from_args(args),
    )
    print(json.dumps(output["summary"]["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
