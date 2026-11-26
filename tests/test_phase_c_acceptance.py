import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from multimodal.check_phase_c_acceptance import REQUIRED_ARTIFACTS, check_phase_c_acceptance


class PhaseCAcceptanceTests(unittest.TestCase):
    def test_missing_artifacts_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as temp:
            result = check_phase_c_acceptance(temp)
            self.assertEqual(result["status"], "failed")
            self.assertIn("mm_media.json", result["checks"]["required_artifacts"]["missing"])

    def test_minimal_consistent_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            working = Path(temp)
            for name in REQUIRED_ARTIFACTS:
                path = working / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")
            image = working / "chart.png"
            image.write_bytes(b"image")
            media = [
                {"media_id": "chart_1", "mapped_type": "chart", "path": "chart.png"},
                {"media_id": "table_1", "mapped_type": "table", "path": "", "table_html": "", "table_markdown": "|x|"},
            ]
            chunks = [{"hash_code": "chunk_1", "text": "text"}]
            entities = [{"entity_name": "MODEL", "source_id": "chunk_1|chart_1"}]
            (working / "mm_media.json").write_text(json.dumps(media), encoding="utf-8")
            (working / "mm_chunk.json").write_text(json.dumps(chunks), encoding="utf-8")
            (working / "phase3/media_graph_chunks.json").write_text(
                json.dumps([{"hash_code": "chart_1", "text": "chart fact"}]), encoding="utf-8"
            )
            (working / "phase3/merged_graph/entity.jsonl").write_text(json.dumps(entities[0]) + "\n", encoding="utf-8")
            (working / "phase3/merged_graph/relation.jsonl").write_text("", encoding="utf-8")
            for name in ("entity.jsonl", "relation.jsonl"):
                (working / name).write_text("", encoding="utf-8")
            empty_hash = hashlib.sha256(b"").hexdigest()
            manifest = {
                "status": "completed",
                "stages": {"hierarchy": {"status": "completed", "input": "phase3/merged_graph"}},
                "inputs": {"entity.jsonl": {"sha256": empty_hash}, "relation.jsonl": {"sha256": empty_hash}},
            }
            (working / "phase3/media_graph_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = check_phase_c_acceptance(working)

            self.assertEqual(result["status"], "passed", result)
            self.assertEqual(result["counts"]["mixed_source_entities"], 1)


if __name__ == "__main__":
    unittest.main()
