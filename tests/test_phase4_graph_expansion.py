from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal.generation.evidence_package import build_evidence_package
from multimodal.graph.graph_expander import expand_graph
from multimodal.graph.mm_graph_loader import load_mm_graph
from multimodal.retrieval.candidate_merge import merge_candidates


def _node(node_id: str, node_type: str, page_id: int = 1, **extra):
    item = {
        "node_id": node_id,
        "doc_id": "doc",
        "node_type": node_type,
        "page_id": page_id,
        "text_for_embedding": extra.pop("text_for_embedding", node_id),
        "raw_ref": extra.pop("raw_ref", {}),
        "metadata": extra.pop("metadata", {}),
    }
    item.update(extra)
    return item


def _edge(edge_id: str, src: str, dst: str, edge_type: str, weight: float = 1.0):
    return {
        "edge_id": edge_id,
        "src": src,
        "dst": dst,
        "src_type": src.split("::")[1],
        "dst_type": dst.split("::")[1],
        "edge_type": edge_type,
        "weight": weight,
        "direction": "directed",
        "evidence": {},
        "metadata": {},
    }


class Phase4GraphExpansionTest(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            _node("doc::text::t1", "text", text_for_embedding="see Figure 1"),
            _node("doc::media::m1", "media", raw_ref={"path": "m1.png"}, metadata={"media_type": "image"}),
            _node("doc::media::m2", "media", raw_ref={"path": "m2.png"}, metadata={"media_type": "image"}),
            _node("doc::text::t2", "text", text_for_embedding="nearby explanation"),
            _node("doc::entity::e1", "entity", text_for_embedding="Entity One"),
            _node("doc::page::1", "page", text_for_embedding="page one"),
            _node("doc::text::t3", "text", text_for_embedding="page text 3"),
        ]
        self.edges = [
            _edge("e_ref", "doc::text::t1", "doc::media::m1", "text_refers_to_media", 0.9),
            _edge("e_near", "doc::media::m1", "doc::text::t2", "media_near_text", 1.0),
            _edge("e_ent", "doc::entity::e1", "doc::media::m1", "entity_link_media", 1.0),
            _edge("e_page_t1", "doc::page::1", "doc::text::t1", "page_contains_node", 1.0),
            _edge("e_page_m1", "doc::page::1", "doc::media::m1", "page_contains_node", 1.0),
            _edge("e_page_m2", "doc::page::1", "doc::media::m2", "page_contains_node", 1.0),
            _edge("e_page_t3", "doc::page::1", "doc::text::t3", "page_contains_node", 1.0),
        ]
        self.graph = self._graph(self.nodes, self.edges)

    def test_text_refers_to_media_expands_text_to_media(self):
        expanded, edges = expand_graph([self._anchor("doc::text::t1", "text")], self.graph, {}, self._config())
        self.assertTrue(any(item["node_id"] == "doc::media::m1" for item in expanded))
        self.assertTrue(any(edge["edge_type"] == "text_refers_to_media" for edge in edges))

    def test_media_near_text_expands_media_to_text(self):
        expanded, edges = expand_graph([self._anchor("doc::media::m1", "media")], self.graph, {}, self._config())
        self.assertTrue(any(item["node_id"] == "doc::text::t2" for item in expanded))
        self.assertTrue(any(edge["edge_type"] == "media_near_text" for edge in edges))

    def test_entity_link_media_expands_entity_to_media(self):
        expanded, edges = expand_graph([self._anchor("doc::entity::e1", "entity")], self.graph, {}, self._config())
        self.assertTrue(any(item["node_id"] == "doc::media::m1" for item in expanded))
        self.assertTrue(any(edge["edge_type"] == "entity_link_media" for edge in edges))

    def test_page_contains_node_respects_budget(self):
        config = self._config({"page_contains_node": 2})
        expanded, edges = expand_graph([self._anchor("doc::page::1", "page")], self.graph, {}, config)
        page_nodes = [item for item in expanded if item["debug"]["edge_type"] == "page_contains_node"]
        page_edges = [edge for edge in edges if edge["edge_type"] == "page_contains_node"]
        self.assertEqual(len(page_nodes), 2)
        self.assertEqual(len(page_edges), 2)

    def test_candidate_merge_deduplicates_expanded_node(self):
        merged = merge_candidates(
            [
                self._anchor("doc::media::m1", "media", score=0.8),
                {
                    **self._anchor("doc::media::m1", "media", score=0.5),
                    "retrievers": ["graph_expansion"],
                    "source": "graph_expansion",
                    "debug": {"from_anchor": "doc::text::t1", "hop": 1, "edge_type": "text_refers_to_media"},
                },
            ],
            multi_hit_bonus=0.1,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "direct_recall+graph_expansion")
        self.assertIn("expansion_paths", merged[0]["debug"])

    def test_enabled_false_does_not_expand(self):
        expanded, edges = expand_graph([self._anchor("doc::text::t1", "text")], self.graph, {}, {"graph_expansion": {"enabled": False}})
        self.assertEqual(expanded, [])
        self.assertEqual(edges, [])

    def test_missing_edge_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_jsonl(tmp_path / "mm_nodes.jsonl", self.nodes)
            graph = load_mm_graph(tmp_path, edge_file="missing.jsonl", edge_seed_file="also_missing.jsonl")
            expanded, edges = expand_graph([self._anchor("doc::text::t1", "text")], graph, {}, self._config())
            self.assertEqual(expanded, [])
            self.assertEqual(edges, [])

    def test_evidence_package_accepts_expanded_nodes(self):
        expanded = [
            {
                "node_id": "doc::media::m1",
                "node_type": "media",
                "doc_id": "doc",
                "page_id": 1,
                "score": 0.7,
                "retrievers": ["graph_expansion"],
                "source": "graph_expansion",
                "raw_ref": {"path": "m1.png"},
                "metadata": {"media_type": "image"},
                "debug": {"from_anchor": "doc::text::t1", "hop": 1, "edge_type": "text_refers_to_media"},
            }
        ]
        package = build_evidence_package(expanded, {"media_refs": [{"type": "figure", "number": 9}]}, {})
        self.assertTrue(package["visual_evidence"])
        self.assertEqual(package["visual_evidence"][0]["source"], "graph_expansion")
        self.assertIn("debug", package["visual_evidence"][0])

    def _anchor(self, node_id: str, node_type: str, score: float = 1.0):
        node = next(item for item in self.nodes if item["node_id"] == node_id)
        return {
            **node,
            "score": score,
            "retriever": f"{node_type}_retriever",
            "retrievers": [f"{node_type}_retriever"],
            "source": "direct_recall",
        }

    def _config(self, page_limit: dict[str, int] | None = None):
        limits = {
            "text_refers_to_media": 5,
            "media_near_text": 5,
            "entity_link_media": 8,
            "page_contains_node": 20,
            "media_same_page_media": 5,
            "text_caption_of_media": 3,
            "page_next_page": 2,
            "page_prev_page": 2,
        }
        if page_limit:
            limits.update(page_limit)
        return {"graph_expansion": {"enabled": True, "max_graph_hops": 1, "max_neighbors_per_edge_type": limits}}

    def _graph(self, nodes, edges):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            self._write_jsonl(tmp_path / "mm_nodes.jsonl", nodes)
            self._write_jsonl(tmp_path / "mm_edges.jsonl", edges)
            return load_mm_graph(tmp_path)

    def _write_jsonl(self, path: Path, rows):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()
