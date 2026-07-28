from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal.io_utils import save_dataclasses, write_jsonl
from multimodal.processing.context_builder import MultimodalContextBuilder
from multimodal.processing.pipeline import EvidenceAwareMultimodalProcessor, process_workspace
from multimodal.processing.processors import ChartProcessor, ImageProcessor, TableProcessor
from multimodal.schema import MMChunk, MMMedia


class _SpyProcessor:
    def __init__(self):
        self.calls = []

    def process(self, media, context):
        self.calls.append(media.media_id)
        return {"processor": "spy"}, {"confidence": 1.0}, {"overall": 1.0}


class Phase2MultimodalProcessorTest(unittest.TestCase):
    def _fixture(self):
        chunk = MMChunk(
            chunk_id="doc_chunk_000001",
            hash_code="hash-1",
            doc_id="doc",
            text="As shown in Figure 2, the system contains two modules.",
            page_start=7,
            page_end=7,
            section_title="Architecture",
        )
        media = MMMedia(
            media_id="doc_image_1",
            doc_id="doc",
            modality="image",
            mapped_type="image",
            type="image",
            original_type="image",
            page=1,
            path="missing.png",
            caption="Figure 2: System architecture",
            ocr_text="Module A Module B",
            bbox=[10, 20, 300, 400],
            nearby_chunk_ids=[chunk.chunk_id],
        )
        media_node_id = "doc::media::doc_image_1"
        text_node_id = "doc::text::doc_chunk_000001"
        page_node_id = "doc::page::7"
        nodes = [
            {"node_id": media_node_id, "node_type": "media", "page_id": 1, "raw_ref": {"media_id": media.media_id}},
            {"node_id": text_node_id, "node_type": "text", "page_id": 7, "raw_ref": {"chunk_id": chunk.chunk_id}},
            {"node_id": page_node_id, "node_type": "page", "page_id": 7, "raw_ref": {"page": 7}},
        ]
        edges = [
            {"edge_id": "near", "src": media_node_id, "dst": text_node_id, "edge_type": "media_near_text"},
            {"edge_id": "page", "src": page_node_id, "dst": media_node_id, "edge_type": "page_contains_node"},
        ]
        return chunk, media, nodes, edges

    def test_chart_never_uses_image_processor(self):
        chunk, _, nodes, edges = self._fixture()
        chart = MMMedia(
            media_id="doc_chart_1", doc_id="doc", modality="chart", mapped_type="chart", type="chart",
            page=7, path="chart.png", caption="Chart 1", ocr_text="Series A 10 20",
        )
        chart_node = {"node_id": "doc::media::doc_chart_1", "node_type": "media", "page_id": 7, "raw_ref": {"media_id": chart.media_id}}
        builder = MultimodalContextBuilder([chunk], nodes + [chart_node], edges)
        image_spy, chart_spy, table_spy = _SpyProcessor(), _SpyProcessor(), _SpyProcessor()
        pipeline = EvidenceAwareMultimodalProcessor(builder, image_spy, chart_spy, table_spy)

        records = pipeline.process([chart])

        self.assertEqual(image_spy.calls, [])
        self.assertEqual(chart_spy.calls, [chart.media_id])
        self.assertEqual(records[0]["media_type"], "chart")

    def test_table_numbers_are_traceable_to_source_cells(self):
        table = MMMedia(
            media_id="doc_table_1", doc_id="doc", modality="table", mapped_type="table", type="table",
            page=2, path="", caption="Results",
            table_markdown="| Model | Score |\n|---|---|\n| A | 10 |\n| B | 20 |",
        )

        def fake_llm(**kwargs):
            return {
                "important_cells": [
                    {"row": 1, "col": 1, "value": "10", "reason": "reported score"},
                    {"row": 2, "col": 1, "value": "999", "reason": "fabricated"},
                ],
                "comparisons": [
                    {"statement": "B has 20 versus A at 10", "source_cells": [[1, 1], [2, 1]]},
                    {"statement": "B has 999", "source_cells": [[2, 1]]},
                ],
                "grounded_summary": "Scores are 10 and 20.",
                "confidence": 0.9,
            }

        structured, semantic, confidence = TableProcessor(fake_llm).process(table, {})

        self.assertEqual({cell["text"] for cell in structured["numeric_cells"]}, {"10", "20"})
        self.assertEqual(semantic["important_cells"], [{"row": 1, "col": 1, "value": "10", "reason": "reported score"}])
        self.assertEqual(len(semantic["comparisons"]), 1)
        self.assertNotIn("999", json.dumps(semantic))
        self.assertTrue(confidence["numeric_provenance_complete"])

    def test_image_keeps_visual_and_caption_evidence_separate(self):
        chunk, media, nodes, edges = self._fixture()
        context = MultimodalContextBuilder([chunk], nodes, edges).build(media)

        def fake_vlm(**kwargs):
            return {
                "visual_facts": ["A blue rectangular block is visible."],
                "visible_text": ["Module A"],
                "objects": [{"name": "blue block"}],
                "spatial_relations": [],
                "image_type": "diagram",
                "caption_consistency": "The caption calls this a system architecture.",
                "grounded_summary": "A blue block labeled Module A is visible.",
                "uncertain_items": [],
                "confidence": 0.8,
            }

        _, semantic, confidence = ImageProcessor(fake_vlm).process(media, context)

        self.assertEqual(context["direct_evidence"]["caption"], "Figure 2: System architecture")
        self.assertEqual(semantic["visual_facts"], ["A blue rectangular block is visible."])
        self.assertNotIn(context["direct_evidence"]["caption"], semantic["visual_facts"])
        self.assertIn("remain in media_context", confidence["evidence_separation"])

    def test_context_builder_uses_page_and_media_near_text_relations(self):
        chunk, media, nodes, edges = self._fixture()

        context = MultimodalContextBuilder([chunk], nodes, edges).build(media)

        self.assertEqual(context["layout_context"]["page"], 7)
        self.assertEqual(context["layout_context"]["section"], "Architecture")
        self.assertIn("system contains two modules", context["layout_context"]["nearby_text"])
        self.assertIn("Figure 2", context["reference_context"])

    def test_noise_does_not_enter_any_processor(self):
        chunk, _, nodes, edges = self._fixture()
        noise = MMMedia(
            media_id="doc_noise_1", doc_id="doc", modality="noise", mapped_type="noise", type="noise",
            page=7, path="", original_type="page_number", ocr_text="7",
        )
        spies = [_SpyProcessor(), _SpyProcessor(), _SpyProcessor()]
        pipeline = EvidenceAwareMultimodalProcessor(MultimodalContextBuilder([chunk], nodes, edges), *spies)

        self.assertEqual(pipeline.process([noise]), [])
        self.assertTrue(all(spy.calls == [] for spy in spies))

    def test_processing_does_not_create_or_modify_graph_edges(self):
        chunk, media, nodes, edges = self._fixture()
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            save_dataclasses([chunk], working / "mm_chunk.json")
            save_dataclasses([media], working / "mm_media.json")
            write_jsonl(nodes, working / "mm_nodes.jsonl")
            write_jsonl(edges, working / "mm_edges_seed.jsonl")
            edge_bytes_before = (working / "mm_edges_seed.jsonl").read_bytes()

            records = process_workspace(working, vlm_func=None, llm_func=None)

            self.assertEqual((working / "mm_edges_seed.jsonl").read_bytes(), edge_bytes_before)
            self.assertFalse((working / "mm_edges.jsonl").exists())
            self.assertTrue((working / "processed_media.json").exists())
            serialized = json.dumps(records)
            for forbidden in ("graph_text", "retrieval_text", "entity_info", "graph_knowledge"):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
