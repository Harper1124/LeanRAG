from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from multimodal.io_utils import read_jsonl


class MMGraph:
    def __init__(self, nodes: list[dict[str, Any]] | None = None, edges: list[dict[str, Any]] | None = None):
        nodes = list(nodes or [])
        _ensure_document_nodes(nodes, edges or [])
        self.nodes = {node.get("node_id"): node for node in nodes if node.get("node_id")}
        self.out_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.in_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.warnings: list[str] = []
        for edge in edges or []:
            src = edge.get("src")
            dst = edge.get("dst")
            if src not in self.nodes or dst not in self.nodes:
                self.warnings.append(f"skip edge {edge.get('edge_id')}: missing node {src if src not in self.nodes else dst}")
                continue
            self.out_edges[src].append(edge)
            self.in_edges[dst].append(edge)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        return self.nodes.get(node_id)

    def get_out_edges(self, node_id: str, edge_types: set[str] | list[str] | None = None) -> list[dict[str, Any]]:
        return _filter_edges(self.out_edges.get(node_id, []), edge_types)

    def get_in_edges(self, node_id: str, edge_types: set[str] | list[str] | None = None) -> list[dict[str, Any]]:
        return _filter_edges(self.in_edges.get(node_id, []), edge_types)

    def get_neighbor_edges(
        self,
        node_id: str,
        edge_types: set[str] | list[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[dict[str, Any], str]]:
        pairs: list[tuple[dict[str, Any], str]] = []
        if direction in {"out", "both"}:
            for edge in self.get_out_edges(node_id, edge_types):
                pairs.append((edge, edge.get("dst")))
        if direction in {"in", "both"}:
            for edge in self.get_in_edges(node_id, edge_types):
                pairs.append((edge, edge.get("src")))
        if direction == "both":
            for edge in self.get_out_edges(node_id, edge_types):
                if edge.get("direction") == "undirected":
                    pairs.append((edge, edge.get("dst")))
            for edge in self.get_in_edges(node_id, edge_types):
                if edge.get("direction") == "undirected":
                    pairs.append((edge, edge.get("src")))
        seen = set()
        unique = []
        for edge, neighbor_id in pairs:
            key = (edge.get("edge_id"), neighbor_id)
            if not neighbor_id or key in seen:
                continue
            seen.add(key)
            unique.append((edge, neighbor_id))
        return unique

    def get_neighbors(
        self,
        node_id: str,
        edge_types: set[str] | list[str] | None = None,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        return [
            node
            for _, neighbor_id in self.get_neighbor_edges(node_id, edge_types=edge_types, direction=direction)
            for node in [self.get_node(neighbor_id)]
            if node is not None
        ]


def load_mm_graph(
    working_dir: str | Path,
    node_file: str = "mm_nodes.jsonl",
    edge_file: str = "mm_edges.jsonl",
    edge_seed_file: str = "mm_edges_seed.jsonl",
) -> MMGraph:
    working = Path(working_dir)
    node_path = _resolve_artifact_path(working, node_file)
    if not node_path.exists():
        return MMGraph([], [])

    edge_path = _resolve_artifact_path(working, edge_file)
    seed_path = _resolve_artifact_path(working, edge_seed_file)
    selected_edge_path = edge_path if edge_path.exists() else seed_path

    nodes = read_jsonl(node_path)
    edges = read_jsonl(selected_edge_path) if selected_edge_path.exists() else []
    return MMGraph(nodes, edges)


def _filter_edges(edges: list[dict[str, Any]], edge_types: set[str] | list[str] | None) -> list[dict[str, Any]]:
    if edge_types is None:
        return list(edges)
    allowed = set(edge_types)
    return [edge for edge in edges if edge.get("edge_type") in allowed]


def _resolve_artifact_path(working_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return working_dir / path


def _ensure_document_nodes(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    existing = {node.get("node_id") for node in nodes}
    doc_ids = {
        edge.get("src")
        for edge in edges
        if edge.get("edge_type") == "document_contains_page" and edge.get("src") not in existing
    }
    for node_id in sorted(doc_id for doc_id in doc_ids if doc_id):
        doc_id = str(node_id).removesuffix("::document")
        nodes.append(
            {
                "node_id": node_id,
                "doc_id": doc_id,
                "node_type": "document",
                "page_id": None,
                "text_for_embedding": doc_id,
                "raw_ref": {"doc_id": doc_id},
                "metadata": {"synthesized_by": "graph_loader"},
            }
        )
