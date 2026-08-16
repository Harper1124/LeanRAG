from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from multimodal.phase3.entity_extractor import extract_media_graph
from multimodal.phase3.input_adapter import (
    JoinedMedia,
    adapt_legacy_entities,
    adapt_legacy_relations,
    join_media_records,
)
from multimodal.phase3.pipeline import Phase3BuildError, run_phase3
from multimodal.phase3.schema import (
    CanonicalRelation,
    GenerationInfo,
    Grounding,
    MediaSemanticUnit,
    SourceReference,
)
from multimodal.phase3.semantic_unit_builder import MediaSemanticUnitBuilder
from multimodal.phase3.validators import SchemaValidationError, validate_media_extraction


def _legacy_entities(source_id: str = "hash-a|hash-b") -> list[dict]:
    return [
        {"entity_name": "Model A", "entity_type": "CONCEPT", "description": "A model.", "source_id": source_id},
        {"entity_name": "Dataset X", "entity_type": "PRODUCT", "description": "A dataset-like product.", "source_id": "hash-a"},
    ]


def _legacy_relations() -> list[dict]:
    return [{
        "src_tgt": "Model A", "tgt_src": "Dataset X", "description": "Model A is related to Dataset X.",
        "weight": 1.0, "source_id": "hash-a",
    }]


def _media(media_id: str, modality: str) -> dict:
    return {
        "media_id": media_id, "doc_id": "doc", "modality": modality, "mapped_type": modality,
        "type": modality, "original_type": modality, "page": 1, "path": f"{media_id}.png",
        "caption": "", "ocr_text": "", "summary": "", "table_html": "", "table_markdown": "",
        "bbox": [1, 2, 3, 4], "nearby_chunk_ids": [], "attached_entity_names": [], "attach_scores": {},
    }


def _processed_image(media_id: str = "img-1", fact: str = "Model A uses Method B.") -> dict:
    return {
        "media_id": media_id, "media_type": "image",
        "media_context": {
            "direct_evidence": {"caption": "Architecture of Model A", "ocr": "", "table_source": ""},
            "layout_context": {"page": 1, "section": "Architecture", "nearby_text": "not copied"},
            "reference_context": "Figure 1 describes Model A.", "uncertainty": "",
        },
        "structured_content": {"width": 100, "height": 100},
        "semantic_content": {
            "visual_facts": [fact], "visible_text": ["Model A", "Method B"],
            "objects": [], "spatial_relations": [], "grounded_summary": fact,
            "evidence_source": {"visual": [fact], "caption": [], "nearby_text": []},
            "uncertain_items": [], "confidence": 0.9,
        },
        "grounding": {"visual": [fact]}, "semantic_role": "architecture", "semantic_confidence": 0.9,
        "confidence": {"overall": 0.9}, "source": {"page_id": 1, "bbox": [1, 2, 3, 4], "raw_path": "img.png"},
    }


def _processed_chart(media_id: str = "chart-1", grounded: bool = True) -> dict:
    grounding = {"visual_evidence": ["A labeled trend is visible."], "ocr_evidence": ["Model A 10"], "context_evidence": []} if grounded else {}
    return {
        "media_id": media_id, "media_type": "chart",
        "media_context": {
            "direct_evidence": {"caption": "Model A performance", "ocr": "", "table_source": ""},
            "layout_context": {"page": 1, "section": "Results", "nearby_text": ""},
            "reference_context": "", "uncertainty": "",
        },
        "structured_content": {
            "ocr_text": "Model A 10", "readable_data_points": [{"text": "10", "bbox": [1, 2, 3, 4]}]
        },
        "semantic_content": {
            "title": "Model A performance", "series": ["Model A"],
            "qualitative_trends": [{"series": "Model A", "trend": "increases", "evidence": "reaches 10"}],
            "grounded_summary": "Model A ranks highest at 10.", "chart_grounding": grounding,
            "confidence": 0.9,
        },
        "grounding": grounding, "semantic_role": "experiment", "semantic_confidence": 0.9,
        "confidence": {"overall": 0.9}, "source": {"page_id": 1, "bbox": [], "raw_path": "chart.png"},
    }


def _processed_table(media_id: str = "table-1") -> dict:
    return {
        "media_id": media_id, "media_type": "table",
        "media_context": {
            "direct_evidence": {"caption": "Scores", "ocr": "", "table_source": "omitted"},
            "layout_context": {"page": 1, "section": "Results", "nearby_text": ""},
            "reference_context": "", "uncertainty": "",
        },
        "structured_content": {"table_parse_available": True, "cells": []},
        "semantic_content": {
            "title_and_purpose": "Model scores",
            "comparisons": [{"statement": "Model A has score 10.", "source_cells": [[1, 0], [1, 1]]}],
            "cell_grounding": [{
                "claim": "Model A has score 10.", "source_cells": [[1, 0], [1, 1]],
                "source_values": ["Model A", "10"],
            }],
            "grounded_summary": "Model A has score 10.", "confidence": 0.9,
        },
        "grounding": {"cell_grounding": []}, "semantic_role": "experiment", "semantic_confidence": 0.9,
        "confidence": {"overall": 0.9}, "source": {"page_id": 1, "bbox": [], "raw_path": "table.png"},
    }


def _joined(media_id: str, modality: str, processed: dict) -> JoinedMedia:
    return JoinedMedia(media_id=media_id, media_type=modality, media=_media(media_id, modality), processed=processed)


def _unit(graph_text: str = "Model A uses Method B.", media_id: str = "img-1") -> MediaSemanticUnit:
    return MediaSemanticUnit(
        chunk_id=f"media_chunk_{'a' * 24 if media_id == 'img-1' else 'b' * 24}",
        media_id=media_id,
        retrieval_text=graph_text,
        graph_text=graph_text,
        evidence_refs=[SourceReference(
            ref_type="media", ref_id=media_id, media_id=media_id,
            grounding=Grounding(kind="visual_fact", locator={"source_file": "processed_media.json", "json_path": "/fact"}),
            confidence=0.9,
        )],
        generation=GenerationInfo(
            schema_version="phase3.media_semantic_unit.v1", generator_version="test", confidence=0.9, warnings=[]
        ),
    )


def _valid_llm_response(confidence: float = 0.9, include_relation: bool = True) -> dict:
    relations = [{
        "source_key": "e1", "target_key": "e2", "relation_type": "uses",
        "description": "Model A uses Method B.", "confidence": confidence, "evidence_ref_indices": [0],
    }] if include_relation else []
    return {
        "entities": [
            {"key": "e1", "entity_name": "Model A", "entity_type": "MODEL", "description": "Model A uses Method B.", "confidence": confidence, "aliases": [], "evidence_ref_indices": [0]},
            {"key": "e2", "entity_name": "Method B", "entity_type": "METHOD", "description": "Model A uses Method B.", "confidence": confidence, "aliases": [], "evidence_ref_indices": [0]},
        ],
        "relations": relations,
    }


class Phase3ContractTest(unittest.TestCase):
    def test_legacy_entity_and_relation_conversion(self):
        entities = adapt_legacy_entities(_legacy_entities(), {"hash-a", "hash-b"})
        relations = adapt_legacy_relations(_legacy_relations(), entities, {"hash-a", "hash-b"})

        self.assertEqual({item.entity_type for item in entities}, {"CONCEPT", "OTHER"})
        self.assertEqual(relations[0].relation_type, "RELATED_TO")
        self.assertIn(relations[0].source_entity_id, {item.entity_id for item in entities})
        self.assertIn(relations[0].target_entity_id, {item.entity_id for item in entities})

    def test_multiple_pipe_delimited_sources_become_sorted_refs(self):
        entities = adapt_legacy_entities(_legacy_entities("hash-b|hash-a|hash-b"), {"hash-a", "hash-b"})
        refs = next(item for item in entities if item.entity_name == "Model A").source_refs
        self.assertEqual([ref.ref_id for ref in refs], ["hash-a", "hash-b"])
        self.assertTrue(all(ref.media_id is None for ref in refs))

    def test_legacy_relation_prefers_exact_case_before_casefold_fallback(self):
        rows = [
            {"entity_name": "Llama-2", "entity_type": "CONCEPT", "description": "First spelling.", "source_id": "hash-a"},
            {"entity_name": "LLaMa-2", "entity_type": "CONCEPT", "description": "Second spelling.", "source_id": "hash-b"},
            {"entity_name": "Model A", "entity_type": "MODEL", "description": "A model.", "source_id": "hash-a"},
        ]
        entities = adapt_legacy_entities(rows, {"hash-a", "hash-b"})
        relation = adapt_legacy_relations([{
            "src_tgt": "Model A", "tgt_src": "Llama-2", "description": "Model A uses Llama-2.",
            "weight": 1.0, "source_id": "hash-a",
        }], entities, {"hash-a", "hash-b"})[0]
        expected = next(item.entity_id for item in entities if item.entity_name == "Llama-2")
        self.assertEqual(relation.target_entity_id, expected)

    def test_media_join_and_generic_phase2_override(self):
        generic = _media("generic-1", "generic")
        result = join_media_records(
            [_processed_image("img-1"), _processed_image("generic-1")],
            [_media("img-1", "image"), generic, _media("noise-1", "noise")],
        )
        self.assertEqual([(item.media_id, item.media_type) for item in result.joined], [("generic-1", "image"), ("img-1", "image")])
        self.assertEqual(result.skipped, [{"media_id": "noise-1", "reason": "noise"}])
        self.assertEqual(result.errors, [])

    def test_missing_duplicate_ids_and_modality_conflicts(self):
        with self.assertRaisesRegex(SchemaValidationError, "missing media_id"):
            join_media_records([{"media_type": "image"}], [])
        with self.assertRaisesRegex(SchemaValidationError, "duplicate media_id"):
            join_media_records([], [_media("x", "image"), _media("x", "image")])
        conflict = join_media_records([_processed_chart("x")], [_media("x", "image")])
        self.assertEqual(conflict.errors[0]["code"], "modality_conflict")

    def test_image_chart_and_table_semantic_units(self):
        builder = MediaSemanticUnitBuilder(generator_version="test")
        units = [
            builder.build(_joined("img-1", "image", _processed_image())),
            builder.build(_joined("chart-1", "chart", _processed_chart())),
            builder.build(_joined("table-1", "table", _processed_table())),
        ]
        self.assertEqual(len({unit.media_id for unit in units}), 3)
        self.assertTrue(all(unit.retrieval_text and unit.evidence_refs for unit in units))
        self.assertIn("Model A has score 10.", units[2].graph_text)

    def test_noise_is_excluded_before_semantic_builder(self):
        result = join_media_records([], [_media("noise-1", "noise")])
        self.assertEqual(result.joined, [])
        self.assertEqual(result.skipped[0]["reason"], "noise")

    def test_table_numeric_facts_have_cell_grounding(self):
        unit = MediaSemanticUnitBuilder().build(_joined("table-1", "table", _processed_table()))
        numeric_refs = [ref for ref in unit.evidence_refs if ref.grounding.kind == "table_cells"]
        self.assertTrue(numeric_refs)
        self.assertTrue(all(ref.grounding.locator.get("cells") for ref in numeric_refs))

    def test_chart_ungrounded_numeric_ranking_is_rejected(self):
        unit = MediaSemanticUnitBuilder().build(_joined("chart-1", "chart", _processed_chart(grounded=False)))
        self.assertNotIn("highest", unit.graph_text.lower())
        self.assertIn("chart_summary_rejected_without_sufficient_grounding", unit.generation.warnings)


class Phase3ExtractionTest(unittest.TestCase):
    def test_visual_layout_objects_are_filtered(self):
        unit = _unit("Model A is shown beside an axis.")
        response = {
            "entities": [
                {"key": "e1", "entity_name": "Model A", "entity_type": "MODEL", "description": "Model A is shown beside an axis.", "confidence": 0.9, "aliases": [], "evidence_ref_indices": [0]},
                {"key": "e2", "entity_name": "axis", "entity_type": "OTHER", "description": "Model A is shown beside an axis.", "confidence": 0.99, "aliases": [], "evidence_ref_indices": [0]},
            ],
            "relations": [],
        }
        result = extract_media_graph([unit], lambda **kwargs: response, {"img-1": "image"}, 0.75, 0.75)
        self.assertEqual([item.entity_name for item in result.entities], ["Model A"])
        self.assertTrue(any(item.get("reason") == "pure_visual_layout_object" for item in result.trace))

    def test_relation_endpoints_must_be_retained_entities(self):
        response = _valid_llm_response()
        response["entities"] = response["entities"][:1]
        result = extract_media_graph([_unit()], lambda **kwargs: response, {"img-1": "image"}, 0.75, 0.75)
        self.assertEqual(result.relations, [])
        self.assertTrue(any(item.get("reason") == "missing_or_filtered_endpoint" for item in result.trace))

    def test_low_confidence_results_are_filtered_with_trace(self):
        result = extract_media_graph(
            [_unit()], lambda **kwargs: _valid_llm_response(0.5), {"img-1": "image"}, 0.75, 0.75
        )
        self.assertEqual(result.entities, [])
        self.assertEqual(result.relations, [])
        self.assertGreaterEqual(sum(item.get("reason") == "below_confidence_threshold" for item in result.trace), 2)

    def test_invalid_json_and_llm_exception_are_recorded_after_retry(self):
        invalid_calls = []

        def invalid(**kwargs):
            invalid_calls.append(1)
            return "not json"

        invalid_result = extract_media_graph([_unit()], invalid, {"img-1": "image"}, 0.75, 0.75)
        self.assertEqual(len(invalid_calls), 2)
        self.assertEqual(invalid_result.errors[0]["code"], "invalid_llm_response")

        exception_calls = []

        def failing(**kwargs):
            exception_calls.append(1)
            raise RuntimeError("service unavailable")

        exception_result = extract_media_graph([_unit()], failing, {"img-1": "image"}, 0.75, 0.75)
        self.assertEqual(len(exception_calls), 2)
        self.assertIn("service unavailable", exception_result.errors[0]["message"])

    def test_source_refs_resolve_and_relations_validate(self):
        unit = _unit()
        result = extract_media_graph(
            [unit], lambda **kwargs: _valid_llm_response(), {"img-1": "image"}, 0.75, 0.75
        )
        validate_media_extraction(result.entities, result.relations, [unit], {"img-1"})
        self.assertTrue(all(ref.ref_id == unit.chunk_id for entity in result.entities for ref in entity.source_refs))

    def test_canonical_outputs_have_no_legacy_alias_fields(self):
        result = extract_media_graph(
            [_unit()], lambda **kwargs: _valid_llm_response(), {"img-1": "image"}, 0.75, 0.75
        )
        forbidden = {"name", "type", "source_id", "source", "target", "relation", "src_tgt", "tgt_src"}
        for record in [asdict(item) for item in result.entities + result.relations]:
            self.assertTrue(forbidden.isdisjoint(record))


class Phase3PipelineTest(unittest.TestCase):
    def _workspace(self, media_count: int = 1) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        (root / "mm_chunk.json").write_text(json.dumps([
            {"hash_code": "hash-a", "text": "Model A"}, {"hash_code": "hash-b", "text": "Dataset X"}
        ]), encoding="utf-8")
        media = [_media("img-1", "image")]
        processed = [_processed_image("img-1")]
        if media_count > 1:
            media.append(_media("img-2", "image"))
            processed.append(_processed_image("img-2", "Model C exists."))
        (root / "mm_media.json").write_text(json.dumps(media), encoding="utf-8")
        (root / "processed_media.json").write_text(json.dumps(processed), encoding="utf-8")
        (root / "entity.jsonl").write_text("".join(json.dumps(item) + "\n" for item in _legacy_entities()), encoding="utf-8")
        (root / "relation.jsonl").write_text("".join(json.dumps(item) + "\n" for item in _legacy_relations()), encoding="utf-8")
        return temporary, root

    def test_phase1_phase2_inputs_are_not_modified_and_runs_are_identical(self):
        temporary, root = self._workspace()
        self.addCleanup(temporary.cleanup)
        protected = ["mm_chunk.json", "mm_media.json", "processed_media.json", "entity.jsonl", "relation.jsonl"]
        before = {name: (root / name).read_bytes() for name in protected}

        first = run_phase3(root, lambda **kwargs: _valid_llm_response())
        first_outputs = {name: (root / "phase3" / name).read_bytes() for name in (
            "media_semantic_units.jsonl", "media_entity.jsonl", "media_relation.jsonl", "build_manifest.json"
        )}
        second = run_phase3(root, lambda **kwargs: _valid_llm_response())
        second_outputs = {name: (root / "phase3" / name).read_bytes() for name in first_outputs}

        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        self.assertEqual(first_outputs, second_outputs)
        self.assertEqual(before, {name: (root / name).read_bytes() for name in protected})

    def test_single_media_failure_does_not_discard_other_media(self):
        temporary, root = self._workspace(media_count=2)
        self.addCleanup(temporary.cleanup)

        def selective(**kwargs):
            prompt = kwargs["prompt"]
            if "Model C exists" in prompt:
                raise RuntimeError("intentional media failure")
            return _valid_llm_response()

        manifest = run_phase3(root, selective)
        entities = [json.loads(line) for line in (root / "phase3" / "media_entity.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(manifest["status"], "partial_failed")
        self.assertEqual({ref["media_id"] for item in entities for ref in item["source_refs"]}, {"img-1"})

    def test_manifest_distinguishes_completed_partial_and_failed(self):
        complete_temp, complete_root = self._workspace()
        self.addCleanup(complete_temp.cleanup)
        self.assertEqual(run_phase3(complete_root, lambda **kwargs: _valid_llm_response())["status"], "completed")

        partial_temp, partial_root = self._workspace(media_count=2)
        self.addCleanup(partial_temp.cleanup)

        def partial(**kwargs):
            return _valid_llm_response() if "Model A uses Method B" in kwargs["prompt"] else "bad"

        self.assertEqual(run_phase3(partial_root, partial)["status"], "partial_failed")

        failed_temp, failed_root = self._workspace()
        self.addCleanup(failed_temp.cleanup)
        with self.assertRaises(Phase3BuildError) as caught:
            run_phase3(failed_root, lambda **kwargs: "bad")
        self.assertEqual(caught.exception.manifest["status"], "failed")
        self.assertTrue((failed_root / "phase3" / "build_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
