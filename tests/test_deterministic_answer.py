import json
import tempfile
import unittest
from pathlib import Path

from multimodal.generation.deterministic_answer import try_deterministic_answer


class DeterministicAnswerTests(unittest.TestCase):
    def test_excluding_appendix_counts_only_main_document(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            chunks = [
                {"page_start": 1, "text": "Introduction"},
                {"page_start": 5, "section_title": "Appendix A", "text": "Appendix details"},
            ]
            media = [
                {"media_id": "main_1", "modality": "image", "page": 2},
                {"media_id": "main_2", "modality": "image", "page": 4},
                {"media_id": "appendix_1", "modality": "image", "page": 5},
            ]
            (working / "mm_chunk.json").write_text(json.dumps(chunks), encoding="utf-8")
            (working / "mm_media.json").write_text(json.dumps(media), encoding="utf-8")

            answer, trace = try_deterministic_answer(
                "How many pictures are used, excluding the Appendix?",
                {"working_dir": str(working)},
            )

            self.assertEqual(answer, "2")
            self.assertEqual(trace["scope"]["label"], "main_document_excluding_appendix")
            self.assertEqual(trace["used_media_ids"], ["main_1", "main_2"])


if __name__ == "__main__":
    unittest.main()
