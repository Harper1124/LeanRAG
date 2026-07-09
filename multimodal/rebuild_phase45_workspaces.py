from __future__ import annotations

import argparse
import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from .graph.mm_edge_builder import build_phase4_edges
from .mm_node_builder import build_phase1_mm_graph


def rebuild_workspaces(
    working_root: str | Path,
    import_mysql: bool = False,
    skip_mm_graph: bool = False,
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(working_root).expanduser().resolve()
    result: dict[str, Any] = {
        "working_root": str(root),
        "num_workspaces": 0,
        "rebuilt_mm_graph": 0,
        "imported_mysql": 0,
        "skipped": [],
        "errors": [],
        "items": [],
    }
    if not root.exists():
        raise FileNotFoundError(f"working_root not found: {root}")

    for workspace in sorted(path for path in root.iterdir() if path.is_dir()):
        result["num_workspaces"] += 1
        item: dict[str, Any] = {
            "working_dir": str(workspace),
            "mm_graph_status": "skipped",
            "mysql_status": "skipped",
            "warnings": [],
            "errors": [],
        }
        print(f"[workspace] {workspace}", flush=True)
        try:
            if not skip_mm_graph:
                _require_files(workspace, ["mm_chunk.json", "mm_media.json", "leanrag_chunk.json"])
                trace = build_phase1_mm_graph(str(workspace), validate=False)
                edges = build_phase4_edges(workspace)
                edge_file = workspace / "mm_edges.jsonl"
                item["mm_graph_status"] = "rebuilt"
                item["mm_node_trace"] = trace
                item["mm_edge_file"] = str(edge_file)
                item["mm_edge_counts"] = dict(Counter(edge.get("edge_type") for edge in edges))
                result["rebuilt_mm_graph"] += 1
                print(f"  rebuilt mm graph: {edge_file}", flush=True)

            if import_mysql:
                _import_mysql(workspace, item)
                result["imported_mysql"] += 1
                print(f"  imported mysql: {item.get('mysql_database')}", flush=True)
        except Exception as exc:
            item["errors"].append(str(exc))
            result["errors"].append({"working_dir": str(workspace), "error": str(exc), "traceback": traceback.format_exc()})
            print(f"  ERROR: {exc}", flush=True)
        result["items"].append(item)

    if output_file:
        output_path = Path(output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {output_path}", flush=True)
    return result


def _require_files(workspace: Path, names: list[str]) -> None:
    missing = [name for name in names if not (workspace / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing required files under {workspace}: {missing}")


def _import_mysql(workspace: Path, item: dict[str, Any]) -> None:
    _require_files(workspace, ["all_entities.json", "generate_relations.json", "community.json"])
    from database_utils import _mysql_db_name, create_db_table_mysql, insert_data_to_mysql

    item["mysql_database"] = _mysql_db_name(str(workspace))
    create_db_table_mysql(str(workspace))
    insert_data_to_mysql(str(workspace))
    item["mysql_status"] = "imported"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild Phase 4/5 workspace artifacts for a directory of documents.")
    parser.add_argument("--working_root", required=True, help="Directory containing per-document workspaces.")
    parser.add_argument("--import_mysql", action="store_true", help="Recreate and import original LeanRAG MySQL tables from existing graph files.")
    parser.add_argument("--skip_mm_graph", action="store_true", help="Skip rebuilding mm_nodes.jsonl/mm_edges.jsonl.")
    parser.add_argument("--output_file", default=None, help="Optional JSON summary path.")
    args = parser.parse_args()

    result = rebuild_workspaces(
        working_root=args.working_root,
        import_mysql=args.import_mysql,
        skip_mm_graph=args.skip_mm_graph,
        output_file=args.output_file,
    )
    print(
        json.dumps(
            {
                "working_root": result["working_root"],
                "num_workspaces": result["num_workspaces"],
                "rebuilt_mm_graph": result["rebuilt_mm_graph"],
                "imported_mysql": result["imported_mysql"],
                "num_errors": len(result["errors"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
