from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal.io_utils import save_dataclasses, write_jsonl
from multimodal.processing.chart_ocr import parse_chart_layout
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

    def test_chart_program_ocr_populates_axes_legends_and_values(self):
        chart = MMMedia(
            media_id="doc_chart_ocr", doc_id="doc", modality="chart", mapped_type="chart", type="chart",
            page=7, path="chart.png", caption="Training output norms",
        )

        def fake_ocr(path):
            del path
            return {
                "backend": "test_ocr",
                "status": "ok",
                "width": 400,
                "height": 300,
                "items": [
                    {"text": "Output", "bbox": None, "region": "y_axis_rotated", "line_id": "y-label", "confidence": 0.98},
                    {"text": "Norm", "bbox": None, "region": "y_axis_rotated", "line_id": "y-label", "confidence": 0.98},
                    {"text": "Step", "bbox": [180, 280, 40, 12], "region": "full", "line_id": "x-label", "confidence": 0.99},
                    {"text": "0k", "bbox": [55, 260, 18, 10], "region": "full", "line_id": "x-ticks", "confidence": 0.95},
                    {"text": "30k", "bbox": [360, 260, 25, 10], "region": "full", "line_id": "x-ticks", "confidence": 0.95},
                    {"text": "0.0", "bbox": [8, 235, 22, 10], "region": "full", "line_id": "y0", "confidence": 0.96},
                    {"text": "35.0", "bbox": [4, 18, 28, 10], "region": "full", "line_id": "y35", "confidence": 0.96},
                    {"text": "with dropout", "bbox": [100, 35, 90, 12], "region": "full", "line_id": "legend1", "confidence": 0.97},
                    {"text": "without dropout", "bbox": [100, 52, 105, 12], "region": "full", "line_id": "legend2", "confidence": 0.97},
                    {"text": "20", "bbox": [240, 120, 18, 10], "region": "full", "line_id": "point", "confidence": 0.91},
                ],
            }

        def fake_vlm(**kwargs):
            return {
                "chart_type": "line chart",
                "x_axis": {"label": "Optimization step", "meaning": "training progress", "unit": "step", "confidence": 0.9},
                "y_axis": {"label": "Output norm", "meaning": "model output magnitude", "unit": "", "confidence": 0.9},
                "legends": ["with dropout", "without dropout"],
                "series": ["with dropout", "without dropout"],
                "qualitative_trends": [
                    {"series": "without dropout", "trend": "increases", "evidence": "the line reaches 20"}
                ],
                "chart_grounding": {
                    "visual_evidence": ["Two lines are visible."],
                    "ocr_evidence": ["Step", "Output Norm"],
                    "context_evidence": [],
                },
                "semantic_confidence": 0.88,
                "confidence": 0.9,
            }

        structured, semantic, confidence = ChartProcessor(fake_vlm, ocr_func=fake_ocr).process(chart, {})

        self.assertEqual(structured["ocr_status"], "ok")
        self.assertEqual(structured["x_axis"]["label"], "Step")
        self.assertEqual(structured["y_axis"]["label"], "Output Norm")
        self.assertEqual({item["text"] for item in structured["x_axis"]["tick_labels"]}, {"0k", "30k"})
        self.assertEqual({item["text"] for item in structured["y_axis"]["tick_labels"]}, {"0.0", "35.0"})
        self.assertIn("with dropout", {item["text"] for item in structured["legends"]})
        self.assertIn("20", {item["text"] for item in structured["readable_data_points"]})
        self.assertEqual(semantic["x_axis"]["label"], "Optimization step")
        self.assertNotEqual(semantic["x_axis"], structured["x_axis"])
        self.assertEqual(semantic["chart_grounding"]["ocr_evidence"], ["Step", "Output Norm"])
        self.assertEqual(semantic["semantic_confidence"], 0.88)
        self.assertEqual(len(semantic["qualitative_trends"]), 1)
        self.assertGreater(confidence["programmatic_parse"], 0.5)

    def test_required_chart_ocr_fails_with_diagnostic(self):
        chart = MMMedia(
            media_id="doc_chart_missing_ocr", doc_id="doc", modality="chart", mapped_type="chart", type="chart",
            page=1, path="missing.png",
        )

        with self.assertRaisesRegex(RuntimeError, "Chart OCR unavailable.*chart image not found"):
            ChartProcessor(require_ocr_backend=True).process(chart, {})

    def test_chart_unknown_semantics_do_not_fall_back_to_rule_roles(self):
        chart = MMMedia(
            media_id="doc_chart_unknown", doc_id="doc", modality="chart", mapped_type="chart", type="chart",
            page=1, path="chart.png",
        )

        def fake_ocr(path):
            del path
            return {
                "status": "ok", "width": 400, "height": 300,
                "items": [
                    {"text": "Step", "bbox": [180, 280, 40, 12], "region": "full", "line_id": "x", "confidence": 0.99},
                    {"text": "Loss", "bbox": None, "region": "y_axis_rotated", "line_id": "y", "confidence": 0.99},
                ],
            }

        def fake_vlm(**kwargs):
            return {
                "chart_type": "unknown", "title": "unknown",
                "x_axis": {"label": "unknown"}, "y_axis": {"label": "unknown"},
                "legends": ["unknown"], "series": ["unknown"],
                "semantic_confidence": 0.2, "confidence": 0.2,
            }

        structured, semantic, _ = ChartProcessor(fake_vlm, ocr_func=fake_ocr).process(chart, {})

        self.assertEqual(structured["axis_candidates"]["x"]["label"], "Step")
        self.assertEqual(semantic["x_axis"]["label"], "unknown")
        self.assertEqual(semantic["y_axis"]["label"], "unknown")
        self.assertEqual(semantic["semantic_confidence"], 0.2)

    def test_chart_layout_separates_stacked_bar_axes_legends_and_values(self):
        items = [
            {"text": "Wins", "bbox": [238, 9, 34, 18], "region": "full", "line_id": "legend-1", "confidence": 0.9},
            {"text": "Ties", "bbox": [340, 9, 30, 18], "region": "full", "line_id": "legend-2", "confidence": 0.9},
            {"text": "Loses", "bbox": [438, 9, 39, 18], "region": "full", "line_id": "legend-3", "confidence": 0.9},
            {"text": "Gemini+", "bbox": [12, 83, 61, 12], "region": "full", "line_id": "row-1", "confidence": 0.9},
            {"text": "GPT-4V+", "bbox": [11, 178, 62, 12], "region": "full", "line_id": "row-2", "confidence": 0.9},
            {"text": "41.5", "bbox": [168, 82, 28, 12], "region": "plot_crop", "line_id": "value-1", "confidence": 0.94},
            {"text": "34.5", "bbox": [351, 82, 28, 12], "region": "plot_crop", "line_id": "value-2", "confidence": 0.94},
            {"text": "46.0", "bbox": [180, 366, 30, 11], "region": "plot_crop", "line_id": "value-bottom", "confidence": 0.95},
            {"text": "20", "bbox": [172, 430, 18, 12], "region": "full", "line_id": "tick-20", "confidence": 0.96},
            {"text": "100", "bbox": [555, 430, 26, 12], "region": "full", "line_id": "tick-100", "confidence": 0.96},
            {"text": "Percent", "bbox": [289, 450, 64, 14], "region": "full", "line_id": "x-label", "confidence": 0.95},
            {"text": "(%)", "bbox": [359, 449, 28, 17], "region": "full", "line_id": "x-label", "confidence": 0.95},
        ]

        parsed = parse_chart_layout(
            {"backend": "test", "status": "ok", "width": 600, "height": 475, "items": items}
        )

        self.assertEqual(parsed["title"], "")
        self.assertEqual(parsed["x_axis"]["label"], "Percent (%)")
        self.assertEqual([item["text"] for item in parsed["y_axis"]["tick_labels"]], ["Gemini+", "GPT-4V+"])
        self.assertEqual({item["text"] for item in parsed["legends"]}, {"Wins", "Ties", "Loses"})
        self.assertEqual({item["text"] for item in parsed["readable_data_points"]}, {"41.5", "34.5", "46.0"})

    def test_chart_layout_rejoins_legend_fragments_across_ocr_passes(self):
        items = [
            {"text": "—", "bbox": [183, 31, 28, 2], "region": "full", "line_id": "full-legend", "confidence": 0.8},
            {"text": "7B", "bbox": [222, 27, 15, 10], "region": "plot_crop", "line_id": "crop-legend", "confidence": 0.95},
            {"text": "w/o", "bbox": [243, 28, 22, 10], "region": "plot_crop", "line_id": "crop-legend", "confidence": 0.95},
            {"text": "image", "bbox": [271, 27, 39, 13], "region": "plot_crop", "line_id": "crop-legend", "confidence": 0.96},
            {"text": "generation", "bbox": [316, 27, 71, 13], "region": "full", "line_id": "full-legend", "confidence": 0.96},
            {"text": "Step", "bbox": [216, 288, 29, 13], "region": "full", "line_id": "x-label", "confidence": 0.96},
        ]

        parsed = parse_chart_layout(
            {"backend": "test", "status": "ok", "width": 418, "height": 306, "items": items}
        )

        self.assertEqual([item["text"] for item in parsed["legends"]], ["7B w/o image generation"])
        self.assertEqual(parsed["title"], "")
        self.assertEqual(parsed["readable_data_points"], [])

    def test_chart_layout_keeps_series_names_out_of_data_points(self):
        items = [
            {"text": "7B", "bbox": [320, 20, 20, 11], "region": "full", "line_id": "legend-1", "confidence": 0.95},
            {"text": "34B", "bbox": [320, 38, 25, 11], "region": "full", "line_id": "legend-2", "confidence": 0.95},
            {"text": "100k", "bbox": [100, 270, 32, 11], "region": "full", "line_id": "x-1", "confidence": 0.96},
            {"text": "Step", "bbox": [205, 288, 30, 13], "region": "full", "line_id": "x-label", "confidence": 0.96},
        ]

        parsed = parse_chart_layout(
            {"backend": "test", "status": "ok", "width": 410, "height": 306, "items": items}
        )

        self.assertEqual({item["text"] for item in parsed["legends"]}, {"7B", "34B"})
        self.assertEqual(parsed["readable_data_points"], [])
        self.assertNotIn("7B", parsed["numeric_tokens"])

    def test_chart_layout_recovers_horizontal_bar_categories_and_angled_ticks(self):
        items = [
            {"text": "Containing images", "bbox": [35, 36, 186, 20], "region": "full", "line_id": "row-1", "confidence": 0.96},
            {"text": "Accuracy", "bbox": [130, 503, 91, 20], "region": "full", "line_id": "row-2", "confidence": 0.96},
            {"text": "All", "bbox": [879, 447, 23, 16], "region": "full", "line_id": "legend-1", "confidence": 0.9},
            {"text": "Two", "bbox": [879, 477, 37, 15], "region": "full", "line_id": "legend-2", "confidence": 0.9},
            {"text": "None", "bbox": [881, 506, 50, 16], "region": "full", "line_id": "legend-3", "confidence": 0.9},
            {"text": "0", "bbox": None, "region": "x_axis_rotated", "line_id": "angled-0", "confidence": 0.9},
            {"text": "3000", "bbox": None, "region": "x_axis_rotated", "line_id": "angled-3000", "confidence": 0.9},
            {"text": "Count", "bbox": [559, 615, 70, 19], "region": "full", "line_id": "x-label", "confidence": 0.96},
        ]

        parsed = parse_chart_layout(
            {"backend": "test", "status": "ok", "width": 950, "height": 650, "items": items}
        )

        self.assertEqual(parsed["x_axis"]["label"], "Count")
        self.assertEqual({item["text"] for item in parsed["x_axis"]["tick_labels"]}, {"0", "3000"})
        self.assertEqual({item["text"] for item in parsed["y_axis"]["tick_labels"]}, {"Containing images", "Accuracy"})
        self.assertEqual({item["text"] for item in parsed["legends"]}, {"All", "Two", "None"})

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
                "table_structure": {
                    "header_meaning": [
                        {"meaning": "Score column", "source_cells": [[0, 1]]}
                    ],
                    "column_semantics": [
                        {"meaning": "The recorded scores are 10 and 20", "source_cells": [[1, 1], [2, 1]]},
                        {"meaning": "The score is 999", "source_cells": [[2, 1]]},
                    ],
                    "row_semantics": [],
                },
                "cell_grounding": [
                    {"claim": "A has score 10", "source_cells": [[1, 1]]},
                    {"claim": "B has score 999", "source_cells": [[2, 1]]},
                ],
                "semantic_confidence": 0.85,
                "confidence": 0.9,
            }

        structured, semantic, confidence = TableProcessor(fake_llm).process(table, {})

        self.assertEqual({cell["text"] for cell in structured["numeric_cells"]}, {"10", "20"})
        self.assertEqual(semantic["important_cells"], [{"row": 1, "col": 1, "value": "10", "reason": "reported score"}])
        self.assertEqual(len(semantic["comparisons"]), 1)
        self.assertNotIn("999", json.dumps(semantic))
        self.assertEqual(len(semantic["table_structure"]["header_meaning"]), 1)
        self.assertEqual(len(semantic["table_structure"]["column_semantics"]), 1)
        self.assertTrue(all(item["source_cells"] for item in semantic["cell_grounding"]))
        self.assertEqual(semantic["semantic_confidence"], 0.85)
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
                "evidence_source": {
                    "visual": ["A blue rectangular block is visible."],
                    "caption": ["Figure 2: System architecture"],
                    "nearby_text": ["As shown in Figure 2, the system contains two modules."],
                },
                "semantic_role": ["architecture"],
                "semantic_confidence": 0.75,
                "confidence": 0.8,
            }

        _, semantic, confidence = ImageProcessor(fake_vlm).process(media, context)

        self.assertEqual(context["direct_evidence"]["caption"], "Figure 2: System architecture")
        self.assertEqual(semantic["visual_facts"], ["A blue rectangular block is visible."])
        self.assertNotIn(context["direct_evidence"]["caption"], semantic["visual_facts"])
        self.assertEqual(semantic["evidence_source"]["visual"], ["A blue rectangular block is visible."])
        self.assertEqual(semantic["evidence_source"]["caption"], ["Figure 2: System architecture"])
        self.assertIn("system contains two modules", semantic["evidence_source"]["nearby_text"][0])
        self.assertEqual(semantic["semantic_role"], ["architecture"])
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
            reloaded = json.loads((working / "processed_media.json").read_text(encoding="utf-8"))
            self.assertEqual(reloaded[0]["structured_content"], reloaded[0]["content_understanding"])
            self.assertEqual(reloaded[0]["semantic_content"], reloaded[0]["semantic_understanding"])
            self.assertIn("media_context", reloaded[0])
            self.assertIn("grounding", reloaded[0])
            self.assertIn("semantic_role", reloaded[0])
            self.assertIn("semantic_confidence", reloaded[0])
            serialized = json.dumps(records)
            for forbidden in ("graph_text", "retrieval_text", "entity_info", "graph_knowledge"):
                self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
