from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from ..retrieval.media_ref import extract_media_refs
from ..schema import MMChunk, MMMedia


class MultimodalContextBuilder:
    """Build media context only from Phase 1 nodes, near-text edges, and direct evidence."""

    def __init__(
        self,
        chunks: list[MMChunk],
        nodes: list[dict[str, Any]] | None = None,
        edges: list[dict[str, Any]] | None = None,
        max_nearby_chars: int = 4000,
    ) -> None:
        self.chunks = chunks
        self.nodes = list(nodes or [])
        self.edges = list(edges or [])
        self.max_nearby_chars = max_nearby_chars
        self.node_by_id = {str(node.get("node_id")): node for node in self.nodes if node.get("node_id")}
        self.chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.chunk_by_hash = {chunk.hash_code: chunk for chunk in chunks}
        self.text_node_to_chunk = self._text_node_chunk_index()
        self.media_node_by_media_id = {
            str((node.get("raw_ref") or {}).get("media_id")): node
            for node in self.nodes
            if node.get("node_type") == "media" and (node.get("raw_ref") or {}).get("media_id")
        }
        self.neighbors = self._near_text_neighbors()

    def build(self, media: MMMedia) -> dict[str, Any]:
        media_node = self.media_node_by_media_id.get(media.media_id) or {}
        nearby_chunks = self._nearby_chunks(media, media_node)
        nearby_text = "\n\n".join(chunk.text.strip() for chunk in nearby_chunks if chunk.text.strip())
        nearby_text = nearby_text[: self.max_nearby_chars]
        sections = list(dict.fromkeys(chunk.section_title for chunk in nearby_chunks if chunk.section_title))
        reference_context = self._reference_context(media, nearby_chunks)
        missing = []
        if not media.caption:
            missing.append("caption unavailable")
        if not media.ocr_text:
            missing.append("OCR unavailable")
        if not nearby_text:
            missing.append("nearby text unavailable")
        if media.bbox is None:
            missing.append("bbox unavailable")

        page = self._page_from_page_node(media_node)
        if page is None:
            page = media_node.get("page_id") if media_node.get("page_id") is not None else media.page
        return {
            "direct_evidence": {
                "caption": media.caption or "",
                "ocr": media.ocr_text or "",
                "table_source": media.table_markdown or media.table_html or "",
            },
            "layout_context": {
                "page": page,
                "section": " > ".join(str(section) for section in sections),
                "nearby_text": nearby_text,
            },
            "reference_context": reference_context,
            "uncertainty": "; ".join(missing),
        }

    def _text_node_chunk_index(self) -> dict[str, MMChunk]:
        result = {}
        for node in self.nodes:
            if node.get("node_type") != "text" or not node.get("node_id"):
                continue
            raw_ref = node.get("raw_ref") or {}
            chunk = self.chunk_by_id.get(str(raw_ref.get("chunk_id") or ""))
            chunk = chunk or self.chunk_by_hash.get(str(raw_ref.get("hash_code") or ""))
            if chunk:
                result[str(node["node_id"])] = chunk
        return result

    def _near_text_neighbors(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = defaultdict(list)
        for edge in self.edges:
            if edge.get("edge_type") != "media_near_text":
                continue
            src, dst = str(edge.get("src") or ""), str(edge.get("dst") or "")
            src_node, dst_node = self.node_by_id.get(src, {}), self.node_by_id.get(dst, {})
            if src_node.get("node_type") == "media" and dst_node.get("node_type") == "text":
                result[src].append(dst)
            elif dst_node.get("node_type") == "media" and src_node.get("node_type") == "text":
                result[dst].append(src)
        return {key: list(dict.fromkeys(value)) for key, value in result.items()}

    def _nearby_chunks(self, media: MMMedia, media_node: dict[str, Any]) -> list[MMChunk]:
        chunks = []
        node_id = str(media_node.get("node_id") or "")
        for text_node_id in self.neighbors.get(node_id, []):
            chunk = self.text_node_to_chunk.get(text_node_id)
            if chunk and chunk not in chunks:
                chunks.append(chunk)
        # Legacy Phase 1 artifacts may have nearby_chunk_ids but no seed edge file.
        if not chunks:
            for chunk_id in media.nearby_chunk_ids:
                chunk = self.chunk_by_id.get(chunk_id)
                if chunk and chunk not in chunks:
                    chunks.append(chunk)
        return sorted(chunks, key=lambda item: (item.page_start or 0, item.order, item.chunk_id))

    def _page_from_page_node(self, media_node: dict[str, Any]) -> int | None:
        media_node_id = str(media_node.get("node_id") or "")
        if not media_node_id:
            return None
        for edge in self.edges:
            if edge.get("edge_type") != "page_contains_node":
                continue
            src, dst = str(edge.get("src") or ""), str(edge.get("dst") or "")
            other_id = src if dst == media_node_id else dst if src == media_node_id else ""
            other = self.node_by_id.get(other_id, {})
            if other.get("node_type") == "page" and other.get("page_id") is not None:
                try:
                    return int(other["page_id"])
                except (TypeError, ValueError):
                    return None
        return None

    def _reference_context(self, media: MMMedia, chunks: list[MMChunk]) -> str:
        caption_refs = extract_media_refs(media.caption or "")
        expected = {(ref["kind"], ref["number"]) for ref in caption_refs}
        sentences = []
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?。！？])\s+|\n+", chunk.text or ""):
                refs = extract_media_refs(sentence)
                if not refs:
                    continue
                if expected and not any((ref["kind"], ref["number"]) in expected for ref in refs):
                    continue
                clean = " ".join(sentence.split())
                if clean and clean not in sentences:
                    sentences.append(clean)
        return "\n".join(sentences)[:2000]
