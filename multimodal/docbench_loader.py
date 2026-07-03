from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# 统一加载 DocBench/MMLongBench-Doc 风格的数据目录，输出后续构建和评测共用的 sample 列表。
def load_docbench(dataset_dir: str) -> list[dict]:
    """
    Load a flexible DocBench-style directory.

    The loader accepts common layouts:
    - metadata/qa files: *.jsonl, *.json, *.csv, *.parquet under dataset_dir
    - PDF files under pdfs/ or anywhere below dataset_dir
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"DocBench directory not found: {dataset_dir}")

    pdfs = _find_pdfs(root)
    qa_rows = _find_qa_rows(root)
    if not qa_rows:
        return [
            {
                "doc_id": doc_id,
                "pdf_path": str(pdf_path),
                "question_id": "",
                "question": "",
                "answer": "",
                "metadata": {},
            }
            for doc_id, pdf_path in sorted(pdfs.items())
        ]

    samples = []
    for index, row in enumerate(qa_rows):
        # 兼容不同数据集常见字段名，将 doc_id/pdf_path/question/answer 归一化。
        doc_id = _first(row, ["doc_id", "document_id", "pdf_id", "file_id", "id"])
        pdf_path = _first(row, ["pdf_path", "file_path", "document_path", "path"])
        if pdf_path:
            resolved_pdf = _resolve_path(root, pdf_path)
            doc_id = doc_id or Path(pdf_path).stem
        elif doc_id and doc_id in pdfs:
            resolved_pdf = pdfs[doc_id]
        else:
            resolved_pdf = _match_pdf_by_name(pdfs, doc_id)
            doc_id = doc_id or (resolved_pdf.stem if resolved_pdf else f"doc_{index:06d}")
        if resolved_pdf is None:
            continue
        question_id = _first(row, ["question_id", "qid", "qa_id"], f"q_{index:06d}")
        samples.append(
            {
                "doc_id": str(doc_id),
                "pdf_path": str(resolved_pdf),
                "question_id": str(question_id),
                "question": str(_first(row, ["question", "query", "input"], "")),
                "answer": str(_first(row, ["answer", "gold_answer", "output", "label"], "")),
                "metadata": {key: value for key, value in row.items() if key not in _KNOWN_KEYS},
            }
        )
    return samples


_KNOWN_KEYS = {
    "doc_id", "document_id", "pdf_id", "file_id", "id", "pdf_path", "file_path",
    "document_path", "path", "question_id", "qid", "qa_id", "question", "query",
    "input", "answer", "gold_answer", "output", "label",
}


def _find_pdfs(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in root.rglob("*.pdf")}


def _find_qa_rows(root: Path) -> list[dict[str, Any]]:
    # 只读取候选 QA 文件，避免把结果文件、chunk 文件误识别为评测样本。
    candidates = _qa_candidates(root)
    rows = []
    for path in sorted(candidates):
        loaded = _load_records(path)
        if any(_has_qa_shape(row) for row in loaded):
            rows.extend(row for row in loaded if isinstance(row, dict))
    return rows


def _qa_candidates(root: Path) -> list[Path]:
    # 如果存在标准 qa 文件，优先只读它；否则 samples.json 和 qa.jsonl 会导致样本重复。
    preferred_names = (
        "qa.jsonl",
        "qa.json",
        "qa.csv",
        "qa.parquet",
        "questions.jsonl",
        "questions.json",
        "samples.jsonl",
        "samples.parquet",
    )
    for name in preferred_names:
        path = root / name
        if path.exists():
            return [path]

    excluded_names = {"manifest.json", "mmlongbench_doc_scores.json"}
    excluded_suffixes = ("_chunk.json", "mm_chunk.json", "mm_media.json", "leanrag_chunk.json")
    candidates = []
    for path in root.rglob("*"):
        # 兜底扫描时排除构建产物和结果目录，只保留可能的原始 QA 元数据。
        if path.suffix.lower() not in {".jsonl", ".json", ".csv", ".parquet"}:
            continue
        if path.name in excluded_names or path.name.endswith(excluded_suffixes):
            continue
        if "result" in {part.lower() for part in path.parts}:
            continue
        candidates.append(path)
    return candidates


def _load_records(path: Path) -> list[dict[str, Any]]:
    # 兼容 JSONL/CSV/Parquet/JSON；无法读取时返回空列表，让上层继续尝试其它文件。
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as f:
                return [json.loads(line) for line in f if line.strip()]
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                return list(csv.DictReader(f))
        if path.suffix.lower() == ".parquet":
            try:
                import pandas as pd
            except ModuleNotFoundError:
                return []
            return pd.read_parquet(path).to_dict(orient="records")
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        for key in ("data", "samples", "questions", "qas", "items"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    except Exception:
        return []
    return []


def _has_qa_shape(row: dict[str, Any]) -> bool:
    keys = set(row)
    return bool(keys & {"question", "query", "input"}) or bool(keys & {"pdf_path", "doc_id", "document_id"})


def _first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _resolve_path(root: Path, raw_path: str) -> Path | None:
    # 支持绝对路径、相对数据集根目录路径，以及仅给文件名时的递归匹配。
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    matches = list(root.rglob(path.name))
    return matches[0] if matches else None


def _match_pdf_by_name(pdfs: dict[str, Path], doc_id: str | None) -> Path | None:
    if not doc_id:
        return None
    if doc_id in pdfs:
        return pdfs[doc_id]
    doc_id_lower = str(doc_id).lower()
    for stem, path in pdfs.items():
        if stem.lower() == doc_id_lower or stem.lower() in doc_id_lower or doc_id_lower in stem.lower():
            return path
    return None
