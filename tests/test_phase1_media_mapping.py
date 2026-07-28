from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multimodal.chunk_builder import (
    build_mm_chunks_from_mineru,
    media_type_statistics,
    mm_chunks_to_leanrag_records,
    mm_media_as_dicts,
)
from multimodal.evidence_store import media_records_for_index
from multimodal.media_linker import link_media_to_chunks
from multimodal.mm_node_builder import build_mm_nodes


class Phase1MediaMappingTest(unittest.TestCase):
    def _parse(self, items: list[dict], doc_id: str = "doc"):
        temp_dir = tempfile.TemporaryDirectory()
        root = Path(temp_dir.name)
        (root / "sample_content_list.json").write_text(json.dumps(items), encoding="utf-8")
        chunks, media = build_mm_chunks_from_mineru(str(root), doc_id, "source.pdf")
        self.addCleanup(temp_dir.cleanup)
        return chunks, media

    def test_chart_types_remain_chart_and_unknown_media_is_retained(self):
        chart_types = ["chart", "bar_chart", "line_chart", "pie_chart", "scatter_plot"]
        items = [
            {"type": raw_type, "page_idx": index, "img_path": f"images/{index}.png", "caption": raw_type}
            for index, raw_type in enumerate(chart_types)
        ]
        items.extend(
            [
                {"type": "figure", "page_idx": 0, "img_path": "images/figure.png"},
                {"type": "new_mineru_media", "page_idx": 0, "asset_path": "assets/new.bin"},
            ]
        )

        _, media = self._parse(items)
        charts = [item for item in media if item.original_type in chart_types]

        self.assertEqual(len(charts), len(chart_types))
        self.assertTrue(all(item.type == item.mapped_type == item.modality == "chart" for item in charts))
        self.assertEqual(next(item for item in media if item.original_type == "figure").mapped_type, "image")
        self.assertEqual(next(item for item in media if item.original_type == "new_mineru_media").mapped_type, "generic")
        required = {"media_id", "original_type", "mapped_type", "type", "page", "bbox", "caption", "path"}
        self.assertTrue(all(required.issubset(record) for record in mm_media_as_dicts(media)))

    def test_noise_is_retained_as_metadata_but_excluded_from_index_and_graph(self):
        chunks, media = self._parse(
            [
                {"type": "text", "page_idx": 0, "text": "LeanRAG text retrieval remains available."},
                {"type": "page_number", "page_idx": 0, "text": "1", "bbox": [1, 2, 3, 4]},
                {"type": "logo", "page_idx": 0, "img_path": "images/logo.png", "caption": "Publisher logo"},
                {"type": "chart", "page_idx": 0, "img_path": "images/chart.png", "caption": "Revenue chart"},
            ]
        )
        chunks, media = link_media_to_chunks(chunks, media, embedding_func=None)
        noise = [item for item in media if item.mapped_type == "noise"]
        noise_ids = {item.media_id for item in noise}

        self.assertEqual(len(noise), 2)
        self.assertTrue(all(item.nearby_chunk_ids == [] and item.attach_scores == {} for item in noise))
        self.assertTrue(all(noise_ids.isdisjoint(chunk.attached_media_ids) for chunk in chunks))
        indexed_ids = {record["media_id"] for record in media_records_for_index(media)}
        self.assertTrue(noise_ids.isdisjoint(indexed_ids))
        nodes, indexes = build_mm_nodes(chunks, media)
        graph_media_ids = {
            node.raw_ref.get("media_id") for node in nodes if node.node_type == "media"
        }
        self.assertTrue(noise_ids.isdisjoint(graph_media_ids))
        self.assertNotIn(next(iter(noise_ids)), indexes["media"])

    def test_text_export_is_unchanged_by_media_and_noise_blocks(self):
        text_items = [
            {"type": "text", "page_idx": 0, "text": "First retrieval paragraph."},
            {"type": "text", "page_idx": 1, "text": "Second retrieval paragraph."},
        ]
        baseline_chunks, _ = self._parse(text_items, doc_id="same-doc")
        mixed_chunks, media = self._parse(
            [
                text_items[0],
                {"type": "chart", "page_idx": 0, "img_path": "chart.png"},
                {"type": "page_number", "page_idx": 0, "text": "1"},
                text_items[1],
            ],
            doc_id="same-doc",
        )

        self.assertEqual(mm_chunks_to_leanrag_records(mixed_chunks), mm_chunks_to_leanrag_records(baseline_chunks))
        self.assertEqual(media_type_statistics(media), {"image": 0, "chart": 1, "table": 0, "noise": 1, "generic": 0})

    def test_nested_mineru_v2_pages_preserve_chart_and_noise(self):
        nested_pages = [
            [
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "Nested text."}]}},
                {
                    "type": "chart",
                    "content": {
                        "image_source": {"path": "images/chart.jpg"},
                        "chart_caption": [{"type": "text", "content": "A line chart."}],
                    },
                },
                {"type": "page_number", "content": {"page_number_content": [{"type": "text", "content": "1"}]}},
            ]
        ]
        chunks, media = self._parse(nested_pages)

        self.assertEqual([chunk.text for chunk in chunks], ["Nested text."])
        chart = next(item for item in media if item.mapped_type == "chart")
        self.assertEqual(chart.page, 1)
        self.assertEqual(Path(chart.path).name, "chart.jpg")
        self.assertEqual(chart.caption, "A line chart.")
        self.assertEqual(media_type_statistics(media)["noise"], 1)


if __name__ == "__main__":
    unittest.main()
