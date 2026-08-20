import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


build_module = importlib.import_module("multimodal.build_docbench")


class BuildDocbenchSkipHierarchyTests(unittest.TestCase):
    def test_skip_hierarchy_prepares_graph_inputs_without_building_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            working_root = root / "work"
            dataset.mkdir()

            def save_artifacts(_chunks, _media, output_dir):
                output = Path(output_dir)
                leanrag_chunk = output / "leanrag_chunk.json"
                leanrag_chunk.write_text(
                    json.dumps([{"hash_code": "chunk-1", "text": "LeanRAG connects text and media evidence."}]),
                    encoding="utf-8",
                )
                paths = {}
                for key, name in {
                    "mm_chunk_file": "mm_chunk.json",
                    "mm_media_file": "mm_media.json",
                    "mm_nodes_file": "mm_nodes.jsonl",
                    "mm_edges_seed_file": "mm_edges_seed.jsonl",
                    "media_type_stats_file": "media_type_stats.json",
                }.items():
                    path = output / name
                    path.write_text("[]", encoding="utf-8")
                    paths[key] = str(path)
                paths["leanrag_chunk_file"] = str(leanrag_chunk)
                return paths

            with (
                patch.object(
                    build_module,
                    "load_docbench",
                    return_value=[{"doc_id": "paper.pdf", "pdf_path": "paper.pdf", "question": "q"}],
                ),
                patch.object(
                    build_module,
                    "parse_pdf_with_mineru",
                    return_value={"mineru_output_dir": "mineru"},
                ),
                patch.object(build_module, "build_mm_chunks_from_mineru", return_value=([], [])),
                patch.object(build_module, "media_type_statistics", return_value={}),
                patch.object(build_module, "link_media_to_chunks", return_value=([], [])),
                patch.object(build_module, "save_mm_artifacts", side_effect=save_artifacts),
                patch.object(build_module, "link_media_to_entities", return_value=[]),
                patch.object(build_module, "save_dataclasses"),
                patch.object(build_module, "build_phase1_mm_graph") as phase1,
                patch.object(build_module, "_try_build_leanrag_graph") as hierarchy,
            ):
                manifests = build_module.build_docbench(
                    str(dataset),
                    str(working_root),
                    force=True,
                    build_graph=True,
                    build_hierarchy=False,
                )

            workspace = working_root / "paper.pdf"
            self.assertTrue((workspace / "entity.jsonl").exists())
            self.assertTrue((workspace / "relation.jsonl").exists())
            phase1.assert_called_once()
            hierarchy.assert_not_called()
            self.assertEqual(manifests[0]["graph_status"], "prepared")
            self.assertEqual(manifests[0]["hierarchy_status"], "skipped")

    def test_prepared_manifest_is_reusable_only_when_hierarchy_is_deferred(self):
        manifest = {"graph_status": "prepared"}
        self.assertTrue(
            build_module._can_reuse_manifest(
                manifest, force=False, build_graph=True, build_hierarchy=False
            )
        )
        self.assertFalse(
            build_module._can_reuse_manifest(
                manifest, force=False, build_graph=True, build_hierarchy=True
            )
        )


if __name__ == "__main__":
    unittest.main()
