import json
import tempfile
import unittest
from pathlib import Path

from multimodal.mm_query import query_mm_graph
from multimodal.retrieval.candidate_merge import merge_candidates
from multimodal.retrieval.source_resolver import complete_source_id, resolve_source_evidence


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class SourceResolverTests(unittest.TestCase):
    def test_mixed_entity_sources_resolve_text_media_and_table_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            _write_json(working / "mm_chunk.json", [{
                "chunk_id": "text_1", "hash_code": "chunk_1", "doc_id": "doc", "text": "text fact",
                "page_start": 2, "page_end": 2, "modality": "text",
            }])
            _write_json(working / "mm_media.json", [{
                "media_id": "table_1", "doc_id": "doc", "modality": "table", "mapped_type": "table",
                "page": 3, "path": "/assets/table.png", "bbox": [0, 0, 1, 1], "caption": "Scores",
                "summary": "table fact", "ocr_text": "A 90", "table_html": "<table><tr><td>90</td></tr></table>",
                "table_markdown": "|A|\n|-|\n|90|",
            }])
            result = resolve_source_evidence(
                working,
                [{"entity_name": "MODEL", "source_id": "chunk_1|table_1"}],
                max_text=1,
                max_media=1,
            )

            self.assertEqual(result["text_evidence"][0]["hash_code"], "chunk_1")
            media = result["media_evidence"][0]
            self.assertEqual(media["media_id"], "table_1")
            self.assertEqual(media["source_path"], "/assets/table.png")
            self.assertTrue(media["table_html"])
            self.assertTrue(media["table_markdown"])
            self.assertEqual({item["resolved_kind"] for item in result["trace"]}, {"text_chunk", "media"})
            self.assertTrue(all(item["expansion_origin"] == "entity_source_id" for item in result["trace"]))

    def test_modality_budgets_preserve_unique_media_source(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            _write_json(working / "mm_chunk.json", [
                {"chunk_id": f"t{i}", "hash_code": f"c{i}", "doc_id": "doc", "text": f"fact {i}"}
                for i in range(10)
            ])
            _write_json(working / "mm_media.json", [{
                "media_id": "chart_1", "doc_id": "doc", "modality": "chart", "mapped_type": "chart",
                "page": 1, "path": "/assets/chart.png", "summary": "chart fact",
            }])
            sources = "|".join([f"c{i}" for i in range(10)] + ["chart_1"])
            result = resolve_source_evidence(
                working, [{"entity_name": "MODEL", "source_id": sources}], max_text=2, max_media=1
            )
            self.assertEqual(len(result["text_evidence"]), 2)
            self.assertEqual([item["media_id"] for item in result["media_evidence"]], ["chart_1"])

    def test_complete_source_id_never_truncates(self):
        sources = [f"source_{index}" for index in range(12)]
        self.assertEqual(complete_source_id("|".join(sources)).split("|"), sources)
        database_source = (Path(__file__).parents[1] / "database_utils.py").read_text(encoding="utf-8")
        self.assertNotIn('split("|")[:5]', database_source)
        self.assertIn("source_id TEXT", database_source)

    def test_candidate_merge_keeps_entity_source_origin(self):
        merged = merge_candidates([
            {"node_id": "m1", "node_type": "media", "score": 1, "retriever": "media_retriever",
             "source": "direct_recall"},
            {"node_id": "m1", "node_type": "media", "score": 2, "retriever": "entity_source_resolver",
             "source": "entity_source_id"},
        ])
        self.assertIn("entity_source_id", merged[0]["source"])
        self.assertIn("direct_recall", merged[0]["source"])


class QuerySourceExpansionTests(unittest.TestCase):
    def test_entity_source_table_reaches_table_reasoner_with_structure(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            _write_json(working / "mm_chunk.json", [])
            _write_json(working / "mm_media.json", [{
                "media_id": "table_1", "doc_id": "doc", "modality": "table", "mapped_type": "table",
                "page": 4, "path": str(working / "table.png"), "caption": "Scores", "summary": "A scores 90",
                "ocr_text": "A 90", "table_html": "<table><tr><td>A</td><td>90</td></tr></table>",
                "table_markdown": "|Model|Score|\n|-|-|\n|A|90|", "bbox": [0, 0, 1, 1],
            }])
            entity_node = {
                "node_id": "doc::entity::score", "doc_id": "doc", "node_type": "entity", "page_id": 4,
                "text_for_embedding": "score table model", "raw_ref": {
                    "entity_name": "SCORE", "source_id": "table_1",
                }, "metadata": {"entity_type": "METRIC"},
            }
            (working / "mm_nodes.jsonl").write_text(json.dumps(entity_node) + "\n", encoding="utf-8")
            prompts = []

            def fake_llm(prompt, **kwargs):
                prompts.append(str(prompt) + str(kwargs.get("system_prompt") or ""))
                return "90"

            answer, trace = query_mm_graph({
                "working_dir": str(working), "retrieval": {"mode": "mm_hybrid", "topk_entity": 4},
                "graph_expansion": {"enabled": False},
                "evidence_budget": {"max_text_nodes": 1, "max_entity_nodes": 2, "max_media_nodes": 1,
                                    "max_table_nodes": 1, "max_page_nodes": 1, "max_vlm_images": 1},
                "generation": {"use_vlm": True, "use_table_reasoner": True, "max_prompt_chars": 8000},
                "use_llm_func": fake_llm, "text_topk": 1, "max_tables_per_query": 1,
            }, None, "What score is shown in the table?", doc_id="doc")

            self.assertEqual(answer, "90")
            table = trace["table_evidence"][0]
            self.assertTrue(table["table_html"])
            self.assertTrue(table["table_markdown"])
            self.assertEqual(trace["evidence_package"]["table_evidence"][0]["raw_ref"]["table_markdown"], table["table_markdown"])
            self.assertTrue(trace["table_reasoner_calls"])
            self.assertTrue(any("|Model|Score|" in prompt for prompt in prompts))

    def test_entity_source_chart_reaches_vlm_with_real_path_and_trace(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            image_path = working / "chart.png"
            image_path.write_bytes(b"not-a-real-image-but-a-real-path")
            _write_json(working / "mm_chunk.json", [{
                "chunk_id": "text_1", "hash_code": "chunk_1", "doc_id": "doc", "text": "model text evidence",
                "modality": "text", "page_start": 1, "page_end": 1, "order": 0,
            }])
            _write_json(working / "mm_media.json", [{
                "media_id": "chart_1", "doc_id": "doc", "modality": "chart", "mapped_type": "chart",
                "page": 1, "path": str(image_path), "caption": "Model chart", "summary": "model reaches 90",
                "ocr_text": "MODEL 90", "bbox": [0, 0, 1, 1],
            }])
            entity_node = {
                "node_id": "doc::entity::model", "doc_id": "doc", "node_type": "entity", "page_id": 1,
                "text_for_embedding": "MODEL chart performance", "raw_ref": {
                    "entity_name": "MODEL", "source_id": "chunk_1|chart_1",
                }, "metadata": {"entity_type": "MODEL"},
            }
            (working / "mm_nodes.jsonl").write_text(json.dumps(entity_node) + "\n", encoding="utf-8")
            captured = {}

            def fake_vlm(**kwargs):
                captured.update(kwargs)
                return "90"

            answer, trace = query_mm_graph({
                "working_dir": str(working),
                "retrieval": {"mode": "mm_hybrid", "topk_entity": 4},
                "graph_expansion": {"enabled": False},
                "evidence_budget": {"max_text_nodes": 1, "max_entity_nodes": 2, "max_media_nodes": 1,
                                    "max_table_nodes": 1, "max_page_nodes": 1, "max_vlm_images": 1},
                "generation": {"use_vlm": True, "answer_with_vlm_when_media": True,
                               "use_table_reasoner": True, "max_prompt_chars": 8000},
                "use_vlm_func": fake_vlm,
                "use_llm_func": lambda *args, **kwargs: "90",
                "text_topk": 1,
                "max_images_per_query": 1,
            }, None, "What value does the model chart show?", doc_id="doc")

            self.assertEqual(answer, "90")
            self.assertEqual(captured["image_paths"], [str(image_path)])
            self.assertEqual(trace["source_resolution"]["counts"]["resolved_media"], 1)
            self.assertEqual(trace["visual_evidence"][0]["source_resolution"], "entity_source_id")
            self.assertIn("entity_source_id", trace["evidence_package"]["visual_evidence"][0]["source"])
            self.assertEqual(trace["vlm_calls"][0]["image_paths"], [str(image_path)])


if __name__ == "__main__":
    unittest.main()
