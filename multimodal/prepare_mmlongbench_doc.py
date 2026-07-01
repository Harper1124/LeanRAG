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
GITHUB_API_TREE = "https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/main/{path}"


def prepare_mmlongbench_doc(
    output_dir: str,
    source: str = "github",
    repo_id: str = "yubo2333/MMLongBench-Doc",
    github_repo: str = "mayubo2333/MMLongBench-Doc",
    max_docs: int | None = None,
    skip_existing: bool = True,
    local_data_file: str | None = None,
    local_documents_dir: str | None = None,
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

    if local_data_file:
        metadata_path = root / Path(local_data_file).name
        if Path(local_data_file).resolve() != metadata_path.resolve():
            shutil.copy2(local_data_file, metadata_path)
    elif source == "github":
        metadata_path = root / "samples.json"
        _download_github(github_repo, "data/samples.json", metadata_path, skip_existing=skip_existing)
    else:
        data_files = _list_hf_files(repo_id, "data")
        parquet_files = [item for item in data_files if item["path"].endswith(".parquet")]
        if not parquet_files:
            raise RuntimeError(f"No parquet metadata files found in {repo_id}/data")
        metadata_path = root / Path(parquet_files[0]["path"]).name
        _download_hf(repo_id, parquet_files[0]["path"], metadata_path, skip_existing=skip_existing)

    rows = _read_records(metadata_path)
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
    if local_documents_dir:
        remote_docs = _copy_local_documents(local_documents_dir, pdf_dir, skip_existing=skip_existing)
    elif source == "github":
        remote_docs = {Path(path).name: path for path in _list_github_files(github_repo, "data/documents")}
    else:
        remote_docs = {Path(item["path"]).name: item["path"] for item in _list_hf_files(repo_id, "documents")}
    missing = []
    for doc_id in doc_ids:
        remote_path = remote_docs.get(doc_id)
        if remote_path is None:
            missing.append(doc_id)
            continue
        if not local_documents_dir:
            if source == "github":
                _download_github(github_repo, remote_path, pdf_dir / doc_id, skip_existing=skip_existing)
            else:
                _download_hf(repo_id, remote_path, pdf_dir / doc_id, skip_existing=skip_existing)

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
        "source": source,
        "repo_id": repo_id,
        "github_repo": github_repo,
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


def _list_github_files(github_repo: str, path_prefix: str) -> list[str]:
    url = GITHUB_API_TREE.format(repo=github_repo)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    prefix = path_prefix.rstrip("/") + "/"
    return [
        item["path"]
        for item in response.json().get("tree", [])
        if item.get("type") == "blob" and item.get("path", "").startswith(prefix)
    ]


def _download_hf(repo_id: str, remote_path: str, output_path: Path, skip_existing: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        return
    url = HF_RESOLVE.format(repo=repo_id, path="/".join(quote(part) for part in remote_path.split("/")))
    _download_url(url, output_path)


def _download_github(github_repo: str, remote_path: str, output_path: Path, skip_existing: bool = True) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        return
    url = GITHUB_RAW.format(repo=github_repo, path="/".join(quote(part) for part in remote_path.split("/")))
    _download_url(url, output_path)


def _download_url(url: str, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            shutil.copyfileobj(response.raw, f)
    tmp_path.replace(output_path)


def _copy_local_documents(local_documents_dir: str, pdf_dir: Path, skip_existing: bool = True) -> dict[str, str]:
    source_dir = Path(local_documents_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Local documents directory not found: {local_documents_dir}")
    copied = {}
    for source in source_dir.rglob("*.pdf"):
        target = pdf_dir / source.name
        if not (skip_existing and target.exists() and target.stat().st_size > 0):
            shutil.copy2(source, target)
        copied[source.name] = str(source)
    return copied


def _read_records(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        for key in ("data", "samples", "questions", "items"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
        raise RuntimeError(f"No records found in JSON file: {path}")
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as f:
            return [json.loads(line) for line in f if line.strip()]
    if suffix != ".parquet":
        raise RuntimeError(f"Unsupported metadata file type: {path}")
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
    parser.add_argument("--source", choices=["github", "hf"], default="github")
    parser.add_argument("--repo_id", default="yubo2333/MMLongBench-Doc")
    parser.add_argument("--github_repo", default="mayubo2333/MMLongBench-Doc")
    parser.add_argument("--max_docs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local_data_file", default=None, help="Offline metadata file, such as samples.json or parquet.")
    parser.add_argument("--local_documents_dir", default=None, help="Offline directory containing MMLongBench-Doc PDFs.")
    args = parser.parse_args()
    manifest = prepare_mmlongbench_doc(
        output_dir=args.output_dir,
        source=args.source,
        repo_id=args.repo_id,
        github_repo=args.github_repo,
        max_docs=args.max_docs,
        skip_existing=not args.force,
        local_data_file=args.local_data_file,
        local_documents_dir=args.local_documents_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
