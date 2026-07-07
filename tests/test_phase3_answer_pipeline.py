from __future__ import annotations

import unittest
from pathlib import Path

from multimodal.generation.answer_planner import plan_answer
from multimodal.generation.evidence_package import build_evidence_package
from multimodal.generation.final_generator import generate_final_answer
from multimodal.generation.vlm_reasoner import run_vlm_reasoner
from multimodal.mm_query import query_mm_graph
from multimodal.retrieval.mm_retriever import mm_hybrid_retrieve


WORKSPACE = Path("exp/mm_leanrag_leanrag_pdf/LeanRAG")
NODES = WORKSPACE / "mm_nodes.jsonl"


def fake_llm(*args, **kwargs):
    text = " ".join(str(arg) for arg in args) + " " + str(kwargs.get("system_prompt", ""))
    if "clustersize" in text.lower():
        return "20"
    if "figure" in text.lower() or "overview" in text.lower():
        return "Overview of the LeanRAG framework"
    return "supported answer"


def fake_vlm(query=None, context="", image_paths=None, **kwargs):
    del kwargs
    return f"visual answer for {len(image_paths or [])} image(s): {query or context[:20]}"


@unittest.skipUnless(NODES.exists(), "Phase 1/2 mm_nodes.jsonl fixture is not available")
class Phase3AnswerPipelineTest(unittest.TestCase):
    def _package(self, query: str):
        merged, query_info, _ = mm_hybrid_retrieve(
            query,
            {"working_dir": str(WORKSPACE), "nodes_file": str(NODES)},
            doc_id="LeanRAG",
        )
        package = build_evidence_package(merged, query_info, {})
        return package, query_info

    def test_visual_evidence_triggers_vlm_plan(self):
        package, query_info = self._package("What does Figure 2 show?")
        plan = plan_answer("What does Figure 2 show?", package, query_info, {"generation": {"use_vlm": True}})
        self.assertTrue(package["visual_evidence"])
        self.assertTrue(plan["use_vlm"])

    def test_table_evidence_triggers_table_reasoner_plan(self):
        package, query_info = self._package("According to Table 6, what is the clustersize for Mix?")
        plan = plan_answer("According to Table 6, what is the clustersize for Mix?", package, query_info, {})
        self.assertTrue(package["table_evidence"])
        self.assertTrue(plan["use_table_reasoner"])

    def test_evidence_package_classifies_media_table_text_page(self):
        package, _ = self._package("According to the table, what is shown on page 6?")
        self.assertTrue(package["text_evidence"])
        self.assertTrue(package["page_evidence"])
        self.assertTrue(package["table_evidence"])

    def test_vlm_missing_path_does_not_crash(self):
        calls = run_vlm_reasoner("question", [{"node_id": "m1", "raw_ref": {}, "metadata": {"media_type": "image"}}], fake_vlm)
        self.assertEqual(calls[0]["error"], "no_image_path")

    def test_final_generator_returns_answer(self):
        package, query_info = self._package("According to Table 6, what is the clustersize for Mix?")
        plan = plan_answer("According to Table 6, what is the clustersize for Mix?", package, query_info, {})
        answer, trace = generate_final_answer(
            "According to Table 6, what is the clustersize for Mix?",
            package,
            plan,
            [],
            [{"table_answer": "20", "used_table_nodes": [], "error": None}],
            {"use_llm_func": fake_llm},
        )
        self.assertEqual(answer, "20")
        self.assertTrue(trace["used_llm"])

    def test_query_mm_graph_trace_contains_phase3_fields(self):
        answer, trace = query_mm_graph(
            {
                "working_dir": str(WORKSPACE),
                "chunks_file": str(WORKSPACE / "leanrag_chunk.json"),
                "use_llm_func": fake_llm,
                "use_vlm_func": fake_vlm,
                "retrieval": {"mode": "mm_hybrid"},
                "generation": {"use_vlm": True, "use_table_reasoner": True, "max_prompt_chars": 12000},
            },
            None,
            "What does Figure 2 show?",
            doc_id="LeanRAG",
        )
        self.assertIsInstance(answer, str)
        self.assertIn("evidence_package", trace)
        self.assertIn("answer_plan", trace)
        self.assertIn("vlm_calls", trace)
        self.assertIn("table_reasoner_calls", trace)
        self.assertIn("selected_evidence_nodes", trace)


if __name__ == "__main__":
    unittest.main()
