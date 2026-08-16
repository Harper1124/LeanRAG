from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..io_utils import load_dataclasses, read_jsonl, write_json
from ..schema import MMChunk, MMMedia, is_indexable_media
from .context_builder import MultimodalContextBuilder
from .processors import ChartProcessor, ImageProcessor, TableProcessor


logger = logging.getLogger(__name__)
SUPPORTED_MEDIA_TYPES = {"image", "chart", "table"}
FORBIDDEN_OUTPUT_KEYS = {"graph_text", "retrieval_text", "entity", "entity_info", "graph_knowledge"}


class EvidenceAwareMultimodalProcessor:
    """Route Phase 1 media to type-specific processors without mutating graph artifacts."""

    def __init__(
        self,
        context_builder: MultimodalContextBuilder,
        image_processor: ImageProcessor | None = None,
        chart_processor: ChartProcessor | None = None,
        table_processor: TableProcessor | None = None,
    ) -> None:
        self.context_builder = context_builder
        self.processors = {
            "image": image_processor or ImageProcessor(),
            "chart": chart_processor or ChartProcessor(),
            "table": table_processor or TableProcessor(),
        }

    def process(self, media_items: list[MMMedia]) -> list[dict[str, Any]]:
        records = []
        for media in media_items:
            media_type = str(media.mapped_type or media.type or media.modality).lower()
            if not is_indexable_media(media) or media_type not in SUPPORTED_MEDIA_TYPES:
                continue
            context = self.context_builder.build(media)
            structured, semantic, confidence = self.processors[media_type].process(media, context)
            record = {
                "media_id": media.media_id,
                "media_type": media_type,
                "media_context": context,
                "structured_content": structured,
                "semantic_content": semantic,
                "confidence": confidence,
                "source": {
                    "page_id": context["layout_context"]["page"],
                    "bbox": media.bbox,
                    "raw_path": _portable_path(media.path),
                },
            }
            _reject_forbidden_output(record)
            records.append(record)
        return records


def process_workspace(
    working_dir: str | Path,
    output_file: str | Path = "processed_media.json",
    vlm_func: Callable | None = None,
    llm_func: Callable | None = None,
    chart_ocr_config: dict[str, Any] | None = None,
    require_chart_ocr: bool = False,
) -> list[dict[str, Any]]:
    working = Path(working_dir)
    media_path = working / "mm_media.json"
    chunk_path = working / "mm_chunk.json"
    if not media_path.exists() or not chunk_path.exists():
        raise FileNotFoundError(f"Phase 1 artifacts mm_media.json/mm_chunk.json are required under {working}")
    media_items = load_dataclasses(media_path, MMMedia)
    chunks = load_dataclasses(chunk_path, MMChunk)
    nodes = read_jsonl(working / "mm_nodes.jsonl") if (working / "mm_nodes.jsonl").exists() else []
    # Phase 2 deliberately reads only Phase 1 seed relations and never writes them.
    edges = read_jsonl(working / "mm_edges_seed.jsonl") if (working / "mm_edges_seed.jsonl").exists() else []
    context_builder = MultimodalContextBuilder(chunks, nodes, edges)
    processor = EvidenceAwareMultimodalProcessor(
        context_builder,
        image_processor=ImageProcessor(vlm_func),
        chart_processor=ChartProcessor(
            vlm_func,
            ocr_config=chart_ocr_config,
            require_ocr_backend=require_chart_ocr,
        ),
        table_processor=TableProcessor(llm_func),
    )
    records = processor.process(media_items)
    target = Path(output_file)
    if not target.is_absolute():
        target = working / target
    write_json(records, target)
    counts = {kind: sum(record["media_type"] == kind for record in records) for kind in sorted(SUPPORTED_MEDIA_TYPES)}
    logger.info("Processed media written to %s: %s", target, " ".join(f"{key}={value}" for key, value in counts.items()))
    return records


def _reject_forbidden_output(value: Any, path: str = "processed_media") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_OUTPUT_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"Forbidden Phase 2 fields at {path}: {sorted(forbidden)}")
        for key, item in value.items():
            _reject_forbidden_output(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_output(item, f"{path}[{index}]")


def _portable_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")
