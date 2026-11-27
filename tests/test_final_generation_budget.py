import unittest

from multimodal.generation.final_generator import PROMPT_INSTRUCTIONS, _build_prompt, generate_final_answer


class FinalGenerationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.question = "What is the color of the line?"
        self.node_id = "doc::media::figure_5"
        self.visual_answer = "The line is red."
        self.evidence = {
            "text_evidence": [{"node_id": "doc::text::1", "text_for_embedding": "Background " * 500}],
            "visual_evidence": [{"node_id": self.node_id, "page_id": 8, "raw_ref": {"path": "figure.jpg"}}],
        }
        self.vlm_calls = [{"node_ids": [self.node_id], "vlm_answer": self.visual_answer, "error": None}]
        self.limit = 1400

    def test_reserves_visual_answer_and_instructions_with_same_budget(self):
        prompts = []
        diagnostics = []
        for reserve in (False, True):
            _, trace = generate_final_answer(
                self.question,
                self.evidence,
                {"answer_mode": "visual", "expected_answer_type": "str"},
                self.vlm_calls,
                [],
                {
                    "answer_format": "Str",
                    "generation": {"max_prompt_chars": self.limit, "reserve_vlm_evidence": reserve},
                    "use_llm_func": lambda prompt: prompts.append(prompt) or "red",
                },
            )
            diagnostics.append(trace["prompt_budget"])

        self.assertTrue(all(len(prompt) <= self.limit for prompt in prompts))
        self.assertNotIn(self.visual_answer, prompts[0])
        self.assertNotIn(PROMPT_INSTRUCTIONS, prompts[0])
        self.assertEqual(prompts[1].count(self.visual_answer), 1)
        self.assertTrue(prompts[1].endswith(PROMPT_INSTRUCTIONS))
        self.assertEqual([row["vlm_answers_retained"] for row in diagnostics], [0, 1])
        self.assertEqual([row["instructions_retained"] for row in diagnostics], [False, True])

    def test_failed_visual_call_keeps_baseline_prompt(self):
        calls = [{"node_ids": [self.node_id], "vlm_answer": "", "error": "model failed"}]
        args = (self.question, self.evidence, calls, [], self.limit, "str")
        self.assertEqual(_build_prompt(*args), _build_prompt(*args, reserve_vlm_evidence=True))

    def test_tiny_budget_does_not_exceed_limit(self):
        prompt = _build_prompt(
            self.question, self.evidence, self.vlm_calls, [], 50, "str", reserve_vlm_evidence=True,
        )
        self.assertLessEqual(len(prompt), 50)
