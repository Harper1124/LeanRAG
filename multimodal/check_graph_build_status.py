from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .docbench_loader import load_docbench


def check_graph_build_status(dataset_dir: str, working_root: str, output_file: str | None = None) -> dict[str, Any]:
    docs = _dataset_docs(dataset_dir)
    rows = []
    for doc_id in docs:
        doc_dir = Path(working_root) / doc_id
        rows.append(_inspect_doc(doc_id, doc_dir))

    missing = [row for row in rows if not row["complete"]]
    summary = {
        "total_docs": len(rows),
        "complete_docs": len(rows) - len(missing),
        "incomplete_docs": len(missing),
        "by_reason": _count_reasons(missing),
    }
    result = {"summary": summary, "missing_or_incomplete": missing, "items": rows}
    if output_file:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _dataset_docs(dataset_dir: str) -> list[str]:
    docs = []
    seen = set()
    for sample in load_docbench(dataset_dir):
        doc_id = str(sample.get("doc_id", ""))
        if doc_id and doc_id not in seen:
            docs.append(doc_id)
            seen.add(doc_id)
    return docs


def _inspect_doc(doc_id: str, doc_dir: Path) -> dict[str, Any]:
    manifest = _read_json(doc_dir / "manifest.json")
    error = _read_json(doc_dir / "graph_build_error.json")
    files = {
        "manifest": (doc_dir / "manifest.json").exists(),
        "all_entities": (doc_dir / "all_entities.json").exists(),
        "community": (doc_dir / "community.json").exists(),
        "generate_relations": (doc_dir / "generate_relations.json").exists(),
        "milvus": (doc_dir / "milvus_demo.db").exists(),
        "entity": (doc_dir / "entity.jsonl").exists(),
        "relation": (doc_dir / "relation.jsonl").exists(),
    }
    graph_status = manifest.get("graph_status")
    reasons = []
    if not doc_dir.exists():
        reasons.append("missing_doc_dir")
    if not files["manifest"]:
        reasons.append("missing_manifest")
    if graph_status != "built":
        reasons.append(f"graph_status={graph_status or 'none'}")
    for key in ("all_entities", "community", "generate_relations", "milvus"):
        if not files[key]:
            reasons.append(f"missing_{key}")
    if error:
        reasons.append(f"graph_build_error={error.get('error', 'unknown')}")

    complete = not reasons
    return {
        "doc_id": doc_id,
        "complete": complete,
        "graph_status": graph_status,
        "reasons": reasons,
        "error": error.get("error") if error else None,
        "working_dir": str(doc_dir),
        "files": files,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _count_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in row["reasons"]:
            if reason.startswith("graph_build_error="):
                reason = "graph_build_error"
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _print_report(result: dict[str, Any], limit: int) -> None:
    summary = result["summary"]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    missing = result["missing_or_incomplete"]
    if not missing:
        print("All documents have complete graph artifacts.")
        return
    print(f"\nMissing/incomplete documents, showing {min(limit, len(missing))}/{len(missing)}:")
    for row in missing[:limit]:
        print(f"- {row['doc_id']}")
        print(f"  graph_status: {row['graph_status']}")
        print(f"  reasons: {'; '.join(row['reasons'])}")
        print(f"  working_dir: {row['working_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check which MMLongBench-Doc workspaces lack complete hierarchical graph artifacts.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_root", required=True)
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    result = check_graph_build_status(args.dataset_dir, args.working_root, args.output_file)
    _print_report(result, args.limit)


if __name__ == "__main__":
    main()
