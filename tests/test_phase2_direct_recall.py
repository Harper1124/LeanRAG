from __future__ import annotations

import unittest
from pathlib import Path

from multimodal.retrieval.candidate_merge import merge_candidates
from multimodal.retrieval.mm_retriever import mm_hybrid_retrieve


WORKSPACE = Path("exp/mm_leanrag_leanrag_pdf/LeanRAG")
NODES = WORKSPACE / "mm_nodes.jsonl"


@unittest.skipUnless(NODES.exists(), "Phase 1 mm_nodes.jsonl fixture is not available")
class Phase2DirectRecallTest(unittest.TestCase):
    def _retrieve(self, query: str):
        return mm_hybrid_retrieve(
            query,
            {"working_dir": str(WORKSPACE), "nodes_file": str(NODES)},
            doc_id="LeanRAG",
        )

    def test_media_node_can_be_recalled(self):
        _, _, trace = self._retrieve("What does Figure 2 show on page 3?")
        self.assertTrue(trace["direct_recall"]["media"])
        self.assertTrue(trace["retrieved_nodes_by_type"]["media"])

    def test_table_media_can_be_recalled(self):
        _, _, trace = self._retrieve("According to the table, which metric has the highest F1 score?")
        media = trace["direct_recall"]["media"]
        self.assertTrue(media)
        self.assertEqual(media[0]["metadata"].get("media_type"), "table")

    def test_page_hint_recalls_page_node(self):
        _, _, trace = self._retrieve("What is shown on page 6?")
        pages = trace["direct_recall"]["page"]
        self.assertTrue(any(item.get("page_id") == 6 for item in pages))

    def test_candidate_merge_deduplicates(self):
        merged = merge_candidates(
            [
                {"node_id": "n1", "node_type": "media", "score": 0.4, "retriever": "media_retriever"},
                {"node_id": "n1", "node_type": "media", "score": 0.8, "retriever": "page_retriever"},
            ],
            multi_hit_bonus=0.1,
        )
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0]["score"], 0.9)
        self.assertEqual(set(merged[0]["retrievers"]), {"media_retriever", "page_retriever"})

    def test_trace_contains_direct_recall_sections(self):
        _, _, trace = self._retrieve("What does the chart show on page 6?")
        self.assertIn("query_info", trace)
        self.assertIn("media", trace["direct_recall"])
        self.assertIn("page", trace["direct_recall"])
        self.assertIn("merged_candidates", trace)


if __name__ == "__main__":
    unittest.main()
