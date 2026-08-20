import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multimodal.media_graph_pipeline import (
    _normalize_extraction_output,
    _record_hierarchy_success,
    merge_legacy_graphs,
    overwrite_grounded_summaries,
    run_media_graph_pipeline,
    semantic_units_to_graph_chunks,
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class MediaGraphPureLogicTests(unittest.TestCase):
    def test_summary_overwrite_preserves_other_fields_and_empty_preserves_summary(self):
        media = [
            {"media_id": "m1", "summary": "old", "path": "a.png", "bbox": [1, 2, 3, 4], "table_html": "<table/>"},
            {"media_id": "m2", "summary": "keep", "path": "b.png"},
        ]
        processed = [
            {"media_id": "m1", "semantic_content": {"grounded_summary": " grounded summary "}},
            {"media_id": "m2", "semantic_content": {"grounded_summary": "  "}},
        ]
        updated, _ = overwrite_grounded_summaries(media, processed)
        self.assertEqual(updated[0]["summary"], "grounded summary")
        self.assertEqual(updated[0]["path"], "a.png")
        self.assertEqual(updated[0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(updated[0]["table_html"], "<table/>")
        self.assertEqual(updated[1]["summary"], "keep")
        expected = dict(media[0])
        expected["summary"] = "grounded summary"
        self.assertEqual(updated[0], expected)

    def test_graph_chunks_skip_empty_graph_text(self):
        chunks, trace = semantic_units_to_graph_chunks([
            {"media_id": "chart_1", "graph_text": "A exceeds B"},
            {"media_id": "image_2", "graph_text": " "},
        ])
        self.assertEqual(chunks, [{"hash_code": "chart_1", "text": "A exceeds B"}])
        self.assertEqual(trace[0]["event"], "empty_graph_text_skipped")

    def test_graph_chunk_rejects_nonempty_text_without_media_id(self):
        with self.assertRaisesRegex(ValueError, "missing media_id"):
            semantic_units_to_graph_chunks([{"media_id": "", "graph_text": "fact"}])

    def test_entity_and_directed_relation_merge(self):
        text_entities = [{"entity_name": ' "LLM" ', "entity_type": "CONCEPT", "description": "desc one",
                          "source_id": "chunk_b|chunk_a", "degree": 0}]
        media_entities = [{"entity_name": "llm", "entity_type": "METHOD", "description": "desc one",
                           "source_id": "chart_1", "degree": 0}]
        relations = [
            {"src_tgt": "A", "tgt_src": "B", "description": "supports", "source_id": "chunk_a", "weight": 1},
            {"src_tgt": "B", "tgt_src": "A", "description": "uses", "source_id": "chart_1", "weight": 1},
            {"src_tgt": "A", "tgt_src": "B", "description": "supports", "source_id": "chart_1", "weight": 1},
            {"src_tgt": "A", "tgt_src": "B", "description": "layout", "source_id": "x", "weight": 1,
             "relation_type": "page_contains_node"},
        ]
        entities, merged_relations, warnings = merge_legacy_graphs(
            text_entities, [], media_entities, relations, token_limit=100
        )
        self.assertEqual(len(entities), 1)
        self.assertEqual(entities[0]["entity_name"], "LLM")
        self.assertEqual(entities[0]["description"], "desc one")
        self.assertEqual(entities[0]["source_id"], "chart_1|chunk_a|chunk_b")
        self.assertEqual({(row["src_tgt"], row["tgt_src"]) for row in merged_relations}, {("A", "B"), ("B", "A")})
        forward = next(row for row in merged_relations if row["src_tgt"] == "A")
        self.assertEqual(forward["weight"], 2)
        self.assertTrue(any(item["code"] == "entity_type_conflict" for item in warnings))
        self.assertTrue(any(item["code"] == "layout_relation_excluded" for item in warnings))

    def test_long_description_uses_summary_and_failure_falls_back(self):
        rows = [{"entity_name": "X", "entity_type": "CONCEPT", "description": "alpha beta gamma",
                 "source_id": "a", "degree": 0}]
        entities, _, _ = merge_legacy_graphs(rows, [], [], [], summarizer=lambda prompt: "short", token_limit=1)
        self.assertEqual(entities[0]["description"], "short")

        def fail(_):
            raise RuntimeError("offline")

        entities, _, warnings = merge_legacy_graphs(rows, [], [], [], summarizer=fail, token_limit=1)
        self.assertTrue(entities[0]["description"])
        self.assertTrue(any(item["code"] == "description_summary_failed" for item in warnings))

    def test_relation_description_summary_and_failure_fallback(self):
        rows = [
            {"src_tgt": "A", "tgt_src": "B", "description": "alpha beta", "source_id": "s1"},
            {"src_tgt": "A", "tgt_src": "B", "description": "gamma delta", "source_id": "s2"},
        ]
        _, relations, _ = merge_legacy_graphs([], rows, [], [], summarizer=lambda _: "short relation", token_limit=1)
        self.assertEqual(relations[0]["description"], "short relation")

        def fail(_):
            raise RuntimeError("offline")

        _, relations, warnings = merge_legacy_graphs([], rows, [], [], summarizer=fail, token_limit=1)
        self.assertTrue(relations[0]["description"])
        self.assertTrue(any(item["code"] == "description_summary_failed" for item in warnings))

    def test_extraction_normalization_drops_layout_relations(self):
        _, relations = _normalize_extraction_output([], [[
            {"src_tgt": "PAGE", "tgt_src": "IMAGE", "description": "contains", "source_id": "m1",
             "relation_type": "page_contains_node"},
            {"src_tgt": "MODEL", "tgt_src": "METRIC", "description": "achieves", "source_id": "m1"},
        ]])
        self.assertEqual(len(relations), 1)
        self.assertEqual((relations[0]["src_tgt"], relations[0]["tgt_src"]), ("MODEL", "METRIC"))


class MediaGraphPipelineIntegrationTests(unittest.TestCase):
    def _minimal_workspace(self, working: Path) -> None:
        _write_json(working / "mm_media.json", [{
            "media_id": "chart_1", "doc_id": "doc", "modality": "chart", "mapped_type": "chart",
            "page": 1, "path": "chart.png", "summary": "old", "caption": "Results chart",
            "ocr_text": "LLM 90", "table_html": "", "table_markdown": "", "bbox": [0, 0, 1, 1],
        }])
        _write_json(working / "processed_media.json", [{
            "media_id": "chart_1", "media_type": "chart",
            "semantic_content": {"grounded_summary": "LLM reaches 90."},
        }])
        _write_jsonl(working / "entity.jsonl", [])
        _write_jsonl(working / "relation.jsonl", [])

    def test_hierarchy_success_clears_stale_lightweight_state(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            _write_json(working / "graph_build_error.json", {"status": "failed"})
            _write_json(working / "manifest.json", {"graph_status": "lightweight", "doc_id": "doc"})

            _record_hierarchy_success(working)

            self.assertFalse((working / "graph_build_error.json").exists())
            manifest = json.loads((working / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["graph_status"], "built")
            self.assertEqual(manifest["graph_input"], "phase3/merged_graph")

    def test_pipeline_builds_merged_graph_idempotently_without_touching_text_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            media = [{
                "media_id": "chart_1", "doc_id": "doc", "modality": "chart", "mapped_type": "chart",
                "page": 1, "path": "chart.png", "summary": "old", "caption": "Results chart",
                "ocr_text": "LLM 90", "table_html": "", "table_markdown": "", "bbox": [0, 0, 1, 1],
            }]
            processed = [{
                "media_id": "chart_1", "media_type": "chart", "semantic_confidence": 0.9,
                "media_context": {"direct_evidence": {"caption": "Results chart"}},
                "structured_content": {"ocr_text": "LLM 90", "readable_data_points": [{"value": 90}]},
                "semantic_content": {
                    "title": "Results", "series": ["LLM"], "qualitative_trends": [],
                    "grounded_summary": "LLM reaches 90.",
                    "chart_grounding": {"ocr_evidence": ["LLM 90"]},
                },
                "confidence": {"overall": 0.9, "summary_validated": True},
            }]
            _write_json(working / "mm_media.json", media)
            _write_json(working / "processed_media.json", processed)
            _write_json(working / "mm_chunk.json", [{
                "chunk_id": "text_1", "hash_code": "chunk_1", "doc_id": "doc", "text": "LLM text fact",
                "modality": "text", "page_start": 1, "page_end": 1, "order": 0,
                "attached_media_ids": ["chart_1"],
            }])
            _write_jsonl(working / "entity.jsonl", [{
                "entity_name": "Llm", "entity_type": "CONCEPT", "description": "text fact",
                "source_id": "chunk_1", "degree": 0,
            }])
            _write_jsonl(working / "relation.jsonl", [])
            original_entities = (working / "entity.jsonl").read_bytes()
            original_relations = (working / "relation.jsonl").read_bytes()
            calls = []

            def fake_extract(chunks):
                calls.append(chunks)
                return ([{"entity_name": "llm", "entity_type": "CONCEPT", "description": "chart fact",
                          "source_id": "chart_1"}], [])

            manifest = run_media_graph_pipeline(
                working, config={}, llm_mode="none", skip_hierarchy=True, force=True,
                extraction_runner=fake_extract,
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(calls[0][0]["hash_code"], "chart_1")
            merged = [json.loads(line) for line in (working / "phase3/merged_graph/entity.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0]["source_id"], "chart_1|chunk_1")
            self.assertEqual((working / "entity.jsonl").read_bytes(), original_entities)
            self.assertEqual((working / "relation.jsonl").read_bytes(), original_relations)
            nodes = [json.loads(line) for line in (working / "mm_nodes.jsonl").read_text(encoding="utf-8").splitlines()]
            media_node = next(node for node in nodes if node["node_type"] == "media")
            entity_node = next(node for node in nodes if node["node_type"] == "entity")
            self.assertIn("Modality: chart", media_node["text_for_embedding"])
            self.assertEqual(entity_node["raw_ref"]["entity_name"], "LLM")
            node_trace = json.loads((working / "mm_node_build_trace.json").read_text(encoding="utf-8"))
            self.assertTrue(node_trace["validation"]["ok"])
            self.assertTrue((working / "mm_edges_seed.jsonl").exists())
            self.assertTrue((working / "mm_edges.jsonl").exists())

            reused = run_media_graph_pipeline(
                working, config={}, llm_mode="none", skip_hierarchy=True,
                extraction_runner=fake_extract,
            )
            self.assertTrue(reused["reused"])
            self.assertEqual(len(calls), 1)

    @patch("multimodal.media_graph_pipeline.make_embedding_func", return_value=lambda _: [])
    @patch("multimodal.media_graph_pipeline.make_chat_func", return_value=lambda _: "summary")
    def test_hierarchy_cannot_report_success_without_required_outputs(self, _chat, _embedding):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            self._minimal_workspace(working)

            with self.assertRaisesRegex(Exception, "required outputs"):
                run_media_graph_pipeline(
                    working,
                    config={"deepseek": {}, "glm": {}},
                    llm_mode="configured",
                    force=True,
                    rebuild_retrieval=False,
                    extraction_runner=lambda _: ([], []),
                    summarizer=lambda _: "summary",
                    hierarchy_runner=lambda *_: None,
                )

            manifest = json.loads((working / "phase3/media_graph_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["stages"]["merge"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
