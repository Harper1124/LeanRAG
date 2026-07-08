from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from multimodal.graph.mm_graph_loader import MMGraph, load_mm_graph


ALLOWED_EDGE_TYPES = {
    "text_caption_of_media",
    "text_refers_to_media",
    "entity_link_media",
    "media_near_text",
    "media_same_page_media",
    "page_contains_node",
    "page_next_page",
    "page_prev_page",
}
EDGE_PRIORITY = [
    "text_caption_of_media",
    "text_refers_to_media",
    "entity_link_media",
    "media_near_text",
    "page_contains_node",
    "media_same_page_media",
    "page_next_page",
    "page_prev_page",
]
EDGE_PRIORITY_RANK = {edge_type: idx for idx, edge_type in enumerate(EDGE_PRIORITY)}
DEFAULT_GRAPH_EXPANSION = {
    "enabled": True,
    "max_graph_hops": 2,
    "max_neighbors_per_edge_type": {
        "text_caption_of_media": 3,
        "text_refers_to_media": 5,
        "entity_link_media": 8,
        "media_near_text": 5,
        "page_contains_node": 20,
        "media_same_page_media": 5,
        "page_next_page": 2,
        "page_prev_page": 2,
    },
}
DEFAULT_HOP_DECAY = {"hop_1": 0.85, "hop_2": 0.70}


def expand_graph(
    anchors: list[dict[str, Any]],
    graph: MMGraph,
    query_info: dict[str, Any] | None,
    config: dict[str, Any] | None,
    doc_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del query_info
    graph_config = _graph_expansion_config(config)
    if not graph_config.get("enabled", True):
        return [], []

    max_hops = max(0, int(graph_config.get("max_graph_hops", 2)))
    limits = graph_config.get("max_neighbors_per_edge_type") or {}
    hop_decay = _hop_decay(config)
    anchor_ids = {item.get("node_id") for item in anchors if item.get("node_id")}
    frontier = [(item, item, 0) for item in anchors if item.get("node_id")]
    expanded_by_id: dict[str, dict[str, Any]] = {}
    expanded_edges: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, str, str, int]] = set()
    seen_expanded_edges: set[tuple[str, str, int]] = set()

    for hop in range(1, max_hops + 1):
        next_frontier: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        for current, anchor, _ in frontier:
            current_id = current.get("node_id")
            if not current_id:
                continue
            grouped = _group_neighbor_edges(graph, current_id)
            for edge_type in EDGE_PRIORITY:
                edge_pairs = grouped.get(edge_type, [])
                if not edge_pairs:
                    continue
                limit = int(limits.get(edge_type, 0))
                if limit <= 0:
                    continue
                edge_pairs.sort(key=lambda pair: float(pair[0].get("weight", 1.0) or 1.0), reverse=True)
                for edge, neighbor_id in edge_pairs[:limit]:
                    neighbor = graph.get_node(neighbor_id)
                    if not neighbor:
                        continue
                    if doc_id is not None and neighbor.get("doc_id") != doc_id:
                        continue
                    path_key = (anchor.get("node_id"), current_id, neighbor_id, hop)
                    if path_key in seen_paths:
                        continue
                    seen_paths.add(path_key)
                    edge_weight = float(edge.get("weight", 1.0) or 1.0)
                    score = float(anchor.get("score") or current.get("score") or 0.0)
                    score *= float(hop_decay.get(f"hop_{hop}", hop_decay.get("hop_2", 0.70)))
                    score *= edge_weight
                    expanded_edge = {
                        "edge_id": edge.get("edge_id"),
                        "src": edge.get("src"),
                        "dst": edge.get("dst"),
                        "edge_type": edge_type,
                        "weight": edge_weight,
                        "hop": hop,
                        "from_anchor": anchor.get("node_id"),
                    }
                    expanded_edge_key = (anchor.get("node_id"), edge.get("edge_id"), hop)
                    if expanded_edge_key not in seen_expanded_edges:
                        seen_expanded_edges.add(expanded_edge_key)
                        expanded_edges.append(expanded_edge)
                    if neighbor_id not in anchor_ids:
                        candidate = _candidate_from_node(neighbor, score, anchor, hop, edge, edge_type)
                        previous = expanded_by_id.get(neighbor_id)
                        if previous is None:
                            expanded_by_id[neighbor_id] = candidate
                        else:
                            expanded_by_id[neighbor_id] = _merge_expanded_candidate(previous, candidate)
                        next_frontier.append((candidate, anchor, hop))
        frontier = next_frontier

    return sorted(expanded_by_id.values(), key=lambda item: item["score"], reverse=True), expanded_edges


def _candidate_from_node(
    node: dict[str, Any],
    score: float,
    anchor: dict[str, Any],
    hop: int,
    edge: dict[str, Any],
    edge_type: str,
) -> dict[str, Any]:
    debug = {
        "from_anchor": anchor.get("node_id"),
        "hop": hop,
        "edge_type": edge_type,
        "edge_id": edge.get("edge_id"),
        "edge_weight": float(edge.get("weight", 1.0) or 1.0),
    }
    candidate = {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "doc_id": node.get("doc_id"),
        "page_id": node.get("page_id"),
        "score": float(score),
        "retrievers": ["graph_expansion"],
        "source": "graph_expansion",
        "raw_ref": node.get("raw_ref") or {},
        "metadata": node.get("metadata") or {},
        "debug": debug,
    }
    for field in ("text_for_embedding", "caption", "ocr_text", "summary"):
        if node.get(field):
            candidate[field] = node.get(field)
    return candidate


def _merge_expanded_candidate(previous: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    previous_paths = _all_paths(previous)
    candidate_paths = _all_paths(candidate)
    all_paths = previous_paths + [path for path in candidate_paths if path not in previous_paths]
    best_path = min(all_paths, key=_path_sort_key)
    winner = candidate if _path_sort_key(candidate.get("debug") or {}) == _path_sort_key(best_path) else previous
    merged = dict(winner)
    merged["score"] = max(float(previous.get("score") or 0.0), float(candidate.get("score") or 0.0))
    debug = dict(best_path)
    debug["expansion_paths"] = [path for path in all_paths if path != best_path]
    merged["debug"] = debug
    return merged


def _all_paths(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    debug = candidate.get("debug") or {}
    paths = [
        {
            "from_anchor": debug.get("from_anchor"),
            "hop": debug.get("hop"),
            "edge_type": debug.get("edge_type"),
            "edge_id": debug.get("edge_id"),
            "edge_weight": debug.get("edge_weight"),
        }
    ]
    for path in debug.get("expansion_paths") or []:
        if isinstance(path, dict):
            paths.append(
                {
                    "from_anchor": path.get("from_anchor"),
                    "hop": path.get("hop"),
                    "edge_type": path.get("edge_type"),
                    "edge_id": path.get("edge_id"),
                    "edge_weight": path.get("edge_weight"),
                }
            )
    return [path for path in paths if path.get("edge_type")]


def _path_sort_key(path: dict[str, Any]) -> tuple[int, int, float]:
    hop = int(path.get("hop") or 999)
    edge_type = path.get("edge_type")
    priority = EDGE_PRIORITY_RANK.get(edge_type, len(EDGE_PRIORITY))
    weight = float(path.get("edge_weight", 1.0) or 1.0)
    return hop, priority, -weight


def _group_neighbor_edges(graph: MMGraph, node_id: str) -> dict[str, list[tuple[dict[str, Any], str]]]:
    grouped: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for edge, neighbor_id in graph.get_neighbor_edges(node_id, edge_types=ALLOWED_EDGE_TYPES, direction="both"):
        edge_type = edge.get("edge_type")
        if edge_type == "node_semantic_similar_node":
            continue
        grouped[edge_type].append((edge, neighbor_id))
    return grouped


def _graph_expansion_config(config: dict[str, Any] | None) -> dict[str, Any]:
    section = dict(DEFAULT_GRAPH_EXPANSION)
    section["max_neighbors_per_edge_type"] = dict(DEFAULT_GRAPH_EXPANSION["max_neighbors_per_edge_type"])
    value = _multimodal_section(config).get("graph_expansion", {}) if isinstance(_multimodal_section(config), dict) else {}
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "max_neighbors_per_edge_type" and isinstance(item, dict):
                section[key].update(item)
            else:
                section[key] = item
    return section


def _hop_decay(config: dict[str, Any] | None) -> dict[str, float]:
    value = dict(DEFAULT_HOP_DECAY)
    fusion = _multimodal_section(config).get("fusion", {}) if isinstance(_multimodal_section(config), dict) else {}
    if isinstance(fusion, dict) and isinstance(fusion.get("hop_decay"), dict):
        value.update(fusion["hop_decay"])
    return {key: float(item) for key, item in value.items()}


def _multimodal_section(config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    return config.get("multimodal") if isinstance(config.get("multimodal"), dict) else config


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run Phase 4 typed graph expansion for one anchor node.")
    parser.add_argument("--nodes", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--anchor_node_id", required=True)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("--output_file", default=None, help="Optional JSONL output path for expanded nodes/edges.")
    parser.add_argument("--print_full", action="store_true", help="Print full expanded nodes/edges JSON to the terminal.")
    args = parser.parse_args()
    nodes_path = Path(args.nodes)
    graph = load_mm_graph(nodes_path.parent, node_file=nodes_path.name, edge_file=str(Path(args.edges)))
    anchor_node = graph.get_node(args.anchor_node_id)
    if not anchor_node:
        raise SystemExit(f"anchor not found: {args.anchor_node_id}")
    anchor = dict(anchor_node)
    anchor.update({"score": 1.0, "retrievers": ["manual_anchor"], "source": "direct_recall"})
    expanded, edges = expand_graph([anchor], graph, {}, {"graph_expansion": {"enabled": True}}, doc_id=args.doc_id)
    payload = {"expanded_nodes": expanded, "expanded_edges": edges, "warnings": graph.warnings}
    if args.output_file:
        _write_expansion_jsonl(Path(args.output_file), expanded, edges, graph.warnings)
        print(json.dumps({"output_file": args.output_file, "num_expanded_nodes": len(expanded), "num_expanded_edges": len(edges)}, ensure_ascii=False, indent=2))
    elif not args.print_full:
        print(
            json.dumps(
                {
                    "num_expanded_nodes": len(expanded),
                    "num_expanded_edges": len(edges),
                    "expanded_node_types": _count_by_key(expanded, "node_type"),
                    "expanded_edge_types": _count_by_key(edges, "edge_type"),
                    "sample_expanded_nodes": [_slim_cli_node(item) for item in expanded[:10]],
                    "sample_expanded_edges": edges[:10],
                    "warnings": graph.warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_expansion_jsonl(
    output_file: Path,
    expanded_nodes: list[dict[str, Any]],
    expanded_edges: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for node in expanded_nodes:
            f.write(json.dumps({"record_type": "expanded_node", **node}, ensure_ascii=False) + "\n")
        for edge in expanded_edges:
            f.write(json.dumps({"record_type": "expanded_edge", **edge}, ensure_ascii=False) + "\n")
        for warning in warnings:
            f.write(json.dumps({"record_type": "warning", "message": warning}, ensure_ascii=False) + "\n")


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _slim_cli_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "page_id": node.get("page_id"),
        "score": node.get("score"),
        "source": node.get("source"),
        "debug": node.get("debug"),
    }


if __name__ == "__main__":
    main()
