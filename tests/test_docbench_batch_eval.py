import json
import tempfile
import unittest
from pathlib import Path

from multimodal.docbench_loader import load_docbench
from multimodal.evaluate_docbench import _resolve_working_dir, _workspace_doc_id, compact_row, run_docbench_eval
from multimodal.score_mmlongbench_doc import _extract_pages, _parse_list


class DocBenchBatchEvalCompatibilityTests(unittest.TestCase):
    def test_pdf_filename_doc_id_is_preserved_for_benchmark_matching(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pdfs").mkdir()
            (root / "pdfs" / "2405.09818v1.pdf").write_bytes(b"%PDF-1.4\n")
            row = {
                "question_id": "2405.09818v1_00769",
                "doc_id": "2405.09818v1.pdf",
                "pdf_path": "pdfs/2405.09818v1.pdf",
                "question": "Which figures include line plots in the paper?",
                "answer": "['Figure 5', 'Figure 6']",
                "evidence_pages": "[6, 7]",
                "evidence_sources": "['Chart']",
                "answer_format": "List",
            }
            (root / "qa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            samples = load_docbench(str(root))

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["doc_id"], "2405.09818v1.pdf")
            self.assertEqual(samples[0]["metadata"]["answer_format"], "List")
            self.assertEqual(samples[0]["metadata"]["evidence_pages"], "[6, 7]")

    def test_stringified_benchmark_lists_are_parsed_for_scoring(self):
        self.assertEqual(_parse_list("['Figure 5', 'Figure 6']"), ["Figure 5", "Figure 6"])
        self.assertEqual(_parse_list("[6, 7]"), [6, 7])
        self.assertEqual(_parse_list("['Chart']"), ["Chart"])

    def test_query_samples_survive_when_source_pdf_is_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = {
                "question_id": "q1",
                "doc_id": "paper.pdf",
                "pdf_path": "pdfs/paper.pdf",
                "question": "What is shown?",
                "answer": "A chart",
            }
            (root / "qa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            # The query workspace exists, but the original PDF was archived.
            (root / "paper").mkdir()

            samples = load_docbench(str(root))

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["doc_id"], "paper.pdf")
            self.assertEqual(Path(samples[0]["pdf_path"]), root / "pdfs" / "paper.pdf")

    def test_workspace_resolution_accepts_pdf_suffix_and_stem_layouts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = {"doc_id": "paper.pdf", "pdf_path": str(root / "paper.pdf")}
            suffixed = root / "paper.pdf"
            suffixed.mkdir()
            (suffixed / "manifest.json").write_text('{"doc_id":"paper.pdf"}', encoding="utf-8")

            resolved, _ = _resolve_working_dir(root, sample)
            self.assertEqual(resolved, suffixed)
            self.assertEqual(_workspace_doc_id(resolved, sample["doc_id"]), "paper.pdf")

            suffixed.rename(root / "paper")
            resolved, _ = _resolve_working_dir(root, sample)
            self.assertEqual(resolved, root / "paper")

    def test_pdf_suffixed_workspace_is_not_mistaken_for_source_pdf(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "paper.pdf").mkdir()
            row = {
                "doc_id": "paper.pdf",
                "pdf_path": "pdfs/paper.pdf",
                "question": "Question",
                "answer": "Answer",
            }
            (root / "qa.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            sample = load_docbench(str(root))[0]

            self.assertEqual(Path(sample["pdf_path"]), root / "pdfs" / "paper.pdf")
            self.assertFalse(Path(sample["pdf_path"]).exists())

    def test_evaluation_rejects_a_zero_sample_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "No QA samples"):
                run_docbench_eval(temp, temp, str(Path(temp) / "predictions.jsonl"))

    def test_compact_predictions_preserve_full_trace_page_coverage(self):
        trace = {
            "text_evidence": [{"page_start": 3, "page_end": 5, "text": "text"}],
            "visual_evidence": [{"page": 10}, {"page": 11}],
            "table_evidence": [],
        }
        compact = compact_row({"question_id": "q", "prediction": "x", "trace": trace}, max_visual_evidence=1)
        self.assertEqual(compact["retrieved_pages"], [3, 4, 5, 10, 11])
        self.assertEqual(_extract_pages(compact), {3, 4, 5, 10, 11})


if __name__ == "__main__":
    unittest.main()
