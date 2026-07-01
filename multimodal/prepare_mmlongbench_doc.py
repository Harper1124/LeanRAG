from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from urllib.parse import quote

import requests

from .io_utils import write_jsonl


HF_API_TREE = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}?recursive=true"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def prepare_mmlongbench_doc(
    output_dir: str,
    repo_id: str = "yubo2333/MMLongBench-Doc",
    max_docs: int | None = None,
    skip_existing: bool = True,
) -> dict:
    """
    Download MMLongBench-Doc metadata and PDFs into a DocBench-style directory.

    Output layout:
    - qa.jsonl
    - pdfs/*.pdf
    - manifest.json
    """
    root = Path(output_dir)
    pdf_dir = root / "pdfs"
    root.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    data_files = _list_hf_files(repo_id, "data")
    parquet_files = [item for item in data_files if item["path"].endswith(".parquet")]
    if not parquet_files:
        raise RuntimeError(f"No parquet metadata files found in {repo_id}/data")

    metadata_path = root / Path(parquet_files[0]["path"]).name
    _download(repo_id, parquet_files[0]["path"], metadata_path, skip_existing=skip_existing)

    rows = _read_parquet(metadata_path)
    if max_docs is not None:
        selected_doc_ids = []
        selected = []
        for row in rows:
            doc_id = str(row.get("doc_id", ""))
            if doc_id not in selected_doc_ids:
                if len(selected_doc_ids) >= max_docs:
                    continue
                selected_doc_ids.append(doc_id)
            selected.append(row)
        rows = selected

    doc_ids = sorted({str(row["doc_id"]) for row in rows if row.get("doc_id")})
    remote_docs = {Path(item["path"]).name: item["path"] for item in _list_hf_files(repo_id, "documents")}
    missing = []
    for doc_id in doc_ids:
        remote_path = remote_docs.get(doc_id)
        if remote_path is None:
            missing.append(doc_id)
            continue
        _download(repo_id, remote_path, pdf_dir / doc_id, skip_existing=skip_existing)

    qa_rows = []
    for idx, row in enumerate(rows):
        doc_id = str(row.get("doc_id", ""))
        qa_rows.append(
            {
                "question_id": f"{Path(doc_id).stem}_{idx:05d}",
                "doc_id": doc_id,
                "pdf_path": str(Path("pdfs") / doc_id),
                "doc_type": _jsonable(row.get("doc_type")),
                "question": _jsonable(row.get("question")),
                "answer": _jsonable(row.get("answer")),
                "evidence_pages": _jsonable(row.get("evidence_pages")),
                "evidence_sources": _jsonable(row.get("evidence_sources")),
                "answer_format": _jsonable(row.get("answer_format")),
            }
        )
    write_jsonl(qa_rows, root / "qa.jsonl")

    manifest = {
        "repo_id": repo_id,
        "metadata_file": str(metadata_path),
        "qa_file": str(root / "qa.jsonl"),
        "pdf_dir": str(pdf_dir),
        "num_questions": len(qa_rows),
        "num_docs": len(doc_ids),
        "missing_docs": missing,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _list_hf_files(repo_id: str, path: str) -> list[dict]:
    url = HF_API_TREE.format(repo=repo_id, path=quote(path))
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return [item for item in response.json() if item.get("type") == "file"]


def _download(repo_id: str, remote_path: str, output_path: Path, skip_existing: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        return
    url = HF_RESOLVE.format(repo=repo_id, path="/".join(quote(part) for part in remote_path.split("/")))
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(response.raw, f)
    tmp_path.replace(output_path)


def _read_parquet(path: Path) -> list[dict]:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Please install pandas with a parquet engine, e.g. `pip install pyarrow`.") from exc
    try:
        return pd.read_parquet(path).to_dict(orient="records")
    except ImportError as exc:
        raise RuntimeError("Reading parquet requires pyarrow or fastparquet, e.g. `pip install pyarrow`.") from exc


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare MMLongBench-Doc as a local DocBench-style directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_id", default="yubo2333/MMLongBench-Doc")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = prepare_mmlongbench_doc(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        max_docs=args.max_docs,
        skip_existing=not args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
