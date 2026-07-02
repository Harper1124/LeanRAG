from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .docbench_loader import load_docbench


def monitor_progress(
    dataset_dir: str,
    working_root: str,
    output_dir: str | None = None,
    predictions_file: str | None = None,
    interval: int = 0,
) -> None:
    while True:
        snapshot = collect_progress(dataset_dir, working_root, output_dir, predictions_file)
        print(format_snapshot(snapshot), flush=True)
        if interval <= 0:
            break
        time.sleep(interval)


def collect_progress(
    dataset_dir: str,
    working_root: str,
    output_dir: str | None = None,
    predictions_file: str | None = None,
) -> dict[str, Any]:
    samples = load_docbench(dataset_dir)
    docs = defaultdict(int)
    for sample in samples:
        if sample.get("question"):
            docs[str(sample["doc_id"])] += 1
        elif sample.get("doc_id"):
            docs.setdefault(str(sample["doc_id"]), 0)

    working = Path(working_root)
    doc_statuses = []
    for doc_id, qa_count in sorted(docs.items()):
        doc_dir = working / doc_id
        status = _doc_status(doc_dir)
        status["doc_id"] = doc_id
        status["qa_count"] = qa_count
        doc_statuses.append(status)

    pred_path = _resolve_predictions_file(output_dir, predictions_file)
    pred_count, pred_docs = _prediction_counts(pred_path)

    return {
        "dataset_dir": dataset_dir,
        "working_root": working_root,
        "predictions_file": str(pred_path) if pred_path else None,
        "total_docs": len(docs),
        "total_questions": sum(docs.values()),
        "prediction_count": pred_count,
        "prediction_docs": dict(pred_docs),
        "doc_statuses": doc_statuses,
        "status_counts": Counter(item["stage"] for item in doc_statuses),
    }


def _doc_status(doc_dir: Path) -> dict[str, Any]:
    if not doc_dir.exists():
        return {"stage": "not_started", "graph_status": None, "updated_at": None}

    manifest = _read_json(doc_dir / "manifest.json")
    error = _read_json(doc_dir / "graph_build_error.json")
    files = {
        "mineru": any((doc_dir / "mineru_output").rglob("*_content_list*.json")) if (doc_dir / "mineru_output").exists() else False,
        "mm_chunk": (doc_dir / "mm_chunk.json").exists(),
        "mm_media": (doc_dir / "mm_media.json").exists(),
        "leanrag_chunk": (doc_dir / "leanrag_chunk.json").exists(),
        "entity": (doc_dir / "entity.jsonl").exists(),
        "relation": (doc_dir / "relation.jsonl").exists(),
        "all_entities": (doc_dir / "all_entities.json").exists(),
        "community": (doc_dir / "community.json").exists(),
        "manifest": (doc_dir / "manifest.json").exists(),
    }
    if manifest:
        stage = "built"
    elif files["all_entities"] or files["community"]:
        stage = "graph_building_or_partial"
    elif files["entity"] and files["relation"]:
        stage = "graph_pending"
    elif files["mm_chunk"] and files["leanrag_chunk"]:
        stage = "chunked"
    elif files["mineru"]:
        stage = "parsed"
    else:
        stage = "started"

    updated_at = _latest_mtime(doc_dir)
    graph_status = manifest.get("graph_status") if manifest else None
    show_error = graph_status != "built"
    return {
        "stage": stage,
        "graph_status": graph_status,
        "error": error.get("error") if show_error and error else None,
        "stale_error": error.get("error") if graph_status == "built" and error else None,
        "updated_at": updated_at,
        "files": files,
    }


def _prediction_counts(path: Path | None) -> tuple[int, Counter]:
    if not path or not path.exists():
        return 0, Counter()
    count = 0
    docs = Counter()
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            docs[str(item.get("doc_id", ""))] += 1
    return count, docs


def _resolve_predictions_file(output_dir: str | None, predictions_file: str | None) -> Path | None:
    if predictions_file:
        return Path(predictions_file)
    if output_dir:
        return Path(output_dir) / "mmlongbench_doc_predictions.jsonl"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _latest_mtime(path: Path) -> str | None:
    latest = None
    for item in path.rglob("*"):
        try:
            mtime = item.stat().st_mtime
        except OSError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest)) if latest else None


def format_snapshot(snapshot: dict[str, Any]) -> str:
    total_docs = snapshot["total_docs"]
    total_questions = snapshot["total_questions"]
    status_counts = snapshot["status_counts"]
    built = status_counts.get("built", 0)
    partial = status_counts.get("graph_building_or_partial", 0)
    started = total_docs - status_counts.get("not_started", 0)
    lines = [
        "=" * 80,
        time.strftime("Progress snapshot: %Y-%m-%d %H:%M:%S"),
        f"Docs: started {started}/{total_docs}, built {built}/{total_docs}, graph_partial {partial}/{total_docs}",
        f"Questions: predictions {snapshot['prediction_count']}/{total_questions}",
        "Stage counts: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())),
    ]
    current = _recent_docs(snapshot["doc_statuses"], limit=8)
    if current:
        lines.append("Recent/active docs:")
        for item in current:
            extra = f", graph_status={item['graph_status']}" if item.get("graph_status") else ""
            error = f", error={item['error']}" if item.get("error") else ""
            lines.append(f"  - {item['doc_id']}: {item['stage']}{extra}{error}, updated={item.get('updated_at')}")
    if snapshot.get("predictions_file"):
        lines.append(f"Predictions file: {snapshot['predictions_file']}")
    return "\n".join(lines)


def _recent_docs(doc_statuses: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    active = [item for item in doc_statuses if item["stage"] not in {"not_started", "built"}]
    built = [item for item in doc_statuses if item["stage"] == "built"]
    active.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    built.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    return (active + built)[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor MMLongBench-Doc MM-LeanRAG evaluation progress.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--working_root", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--predictions_file", default=None)
    parser.add_argument("--interval", type=int, default=0, help="Refresh interval in seconds. 0 prints once.")
    args = parser.parse_args()
    monitor_progress(
        dataset_dir=args.dataset_dir,
        working_root=args.working_root,
        output_dir=args.output_dir,
        predictions_file=args.predictions_file,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()
