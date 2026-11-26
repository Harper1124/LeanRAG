from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import load_dataclasses, read_jsonl, write_json, write_jsonl
from .generation.table_utils import parse_table
from .schema import MMChunk, MMEdge, MMMedia, MMNode, dataclass_to_dict, is_indexable_media


def build_phase1_mm_graph(
    working_dir: str,
    node_file: str = "mm_nodes.jsonl",
    edge_seed_file: str = "mm_edges_seed.jsonl",
    trace_file: str = "mm_node_build_trace.json",
    validate: bool = True,
    entity_file: str = "entity.jsonl",
    media_retrieval_text: dict[str, str] | None = None,
) -> dict[str, Any]:
    working = Path(working_dir)
    chunks = load_dataclasses(working / "mm_chunk.json", MMChunk)
    media_items = load_dataclasses(working / "mm_media.json", MMMedia) if (working / "mm_media.json").exists() else []
    entity_path = working / entity_file
    entities = read_jsonl(entity_path) if entity_path.exists() else []
    extra_pages = _collect_mineru_pages(working)

    nodes, indexes = build_mm_nodes(chunks, media_items, entities, extra_pages=extra_pages)
    for node in nodes:
        media_id = (node.raw_ref or {}).get("media_id") if node.node_type == "media" else None
        if media_id and media_retrieval_text and media_retrieval_text.get(media_id):
            node.text_for_embedding = media_retrieval_text[media_id]
    edges = build_mm_edges(chunks, media_items, nodes, indexes)

    node_path = working / node_file
    edge_path = working / edge_seed_file
    write_jsonl([dataclass_to_dict(node) for node in nodes], node_path)
    write_jsonl([dataclass_to_dict(edge) for edge in edges], edge_path)

    report = validate_phase1_outputs(
        working, node_path, edge_path, entity_file=entity_file
    ) if validate else {}
    trace = {
        "status": "built",
        "node_file": str(node_path),
        "edge_seed_file": str(edge_path),
        "counts": _count_nodes_edges(nodes, edges),
        "media_nodes_built": [
            {
                "node_id": node.node_id,
                "page_id": node.page_id,
                "raw_ref": node.raw_ref,
                "text_for_embedding": node.text_for_embedding[:200],
            }
            for node in nodes
            if node.node_type == "media"
        ],
        "validation": report,
    }
    write_json(trace, working / trace_file)
    return trace


def build_mm_nodes(
    chunks: list[MMChunk],
    media_items: list[MMMedia],
    entities: list[dict[str, Any]] | None = None,
    extra_pages: list[int] | None = None,
) -> tuple[list[MMNode], dict[str, dict[Any, str]]]:
    entities = entities or []
    doc_id = _infer_doc_id(chunks, media_items, entities)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunk_by_hash = {chunk.hash_code: chunk for chunk in chunks}

    nodes: list[MMNode] = []
    indexes: dict[str, dict[Any, str]] = {
        "document": {},
        "page": {},
        "text_by_chunk_id": {},
        "text_by_hash": {},
        "media": {},
        "entity": {},
    }

    document_node = _build_document_node(doc_id, chunks, media_items)
    nodes.append(document_node)
    indexes["document"][doc_id] = document_node.node_id

    for page in _collect_pages(chunks, media_items, extra_pages=extra_pages):
        page_node = _build_page_node(doc_id, page, chunks, media_items)
        nodes.append(page_node)
        indexes["page"][page] = page_node.node_id

    for chunk in sorted(chunks, key=lambda item: (item.page_start or 0, item.order, item.chunk_id)):
        if chunk.modality != "text":
            continue
        node = _build_text_node(chunk)
        nodes.append(node)
        indexes["text_by_chunk_id"][chunk.chunk_id] = node.node_id
        indexes["text_by_hash"][chunk.hash_code] = node.node_id

    for media in sorted((item for item in media_items if is_indexable_media(item)), key=lambda item: (item.page or 0, item.media_id)):
        node = _build_media_node(media, chunk_by_id)
        nodes.append(node)
        indexes["media"][media.media_id] = node.node_id

    for entity in entities:
        node = _build_entity_node(entity, doc_id, chunk_by_hash)
        if not node:
            continue
        nodes.append(node)
        indexes["entity"][entity.get("entity_name")] = node.node_id

    return nodes, indexes


def build_mm_edges(
    chunks: list[MMChunk],
    media_items: list[MMMedia],
    nodes: list[MMNode],
    indexes: dict[str, dict[Any, str]],
) -> list[MMEdge]:
    edges: list[MMEdge] = []
    seen: set[tuple[str, str, str]] = set()
    node_by_id = {node.node_id: node for node in nodes}
    doc_id = _infer_doc_id(chunks, media_items, [])
    document_id = indexes.get("document", {}).get(doc_id, f"{doc_id}::document")

    for page, page_node_id in sorted(indexes["page"].items()):
        _add_edge(
            edges,
            seen,
            document_id,
            page_node_id,
            "document",
            "page",
            "document_contains_page",
            evidence={"source": "rule", "reason": "page discovered from mm_chunk/mm_media page fields"},
            metadata={"page": page},
        )

    pages = sorted(indexes["page"])
    for current, nxt in zip(pages, pages[1:]):
        _add_edge(
            edges,
            seen,
            indexes["page"][current],
            indexes["page"][nxt],
            "page",
            "page",
            "page_next_page",
            evidence={"source": "rule", "reason": "consecutive page order"},
        )
        _add_edge(
            edges,
            seen,
            indexes["page"][nxt],
            indexes["page"][current],
            "page",
            "page",
            "page_prev_page",
            evidence={"source": "rule", "reason": "consecutive page order"},
        )

    for node in nodes:
        if node.node_type not in {"text", "media"} or node.page_id is None:
            continue
        page_node_id = indexes["page"].get(node.page_id)
        if not page_node_id:
            continue
        _add_edge(
            edges,
            seen,
            page_node_id,
            node.node_id,
            "page",
            node.node_type,
            "page_contains_node",
            evidence={"source": "rule", "reason": "node page_id matches page node"},
            metadata={"page": node.page_id},
        )

    for media in (item for item in media_items if is_indexable_media(item)):
        media_node_id = indexes["media"].get(media.media_id)
        if not media_node_id:
            continue
        nearby = list(media.nearby_chunk_ids or [])
        if not nearby:
            nearby = _attached_text_chunks(media.media_id, chunks)
        for chunk_id in nearby:
            text_node_id = indexes["text_by_chunk_id"].get(chunk_id)
            if not text_node_id:
                continue
            weight = float((media.attach_scores or {}).get(chunk_id, 1.0))
            _add_edge(
                edges,
                seen,
                media_node_id,
                text_node_id,
                "media",
                "text",
                "media_near_text",
                weight=weight,
                evidence={"source": "mineru_layout", "reason": "media nearby_chunk_ids or attached_media_ids"},
                metadata={
                    "media_id": media.media_id,
                    "chunk_id": chunk_id,
                    "media_page": media.page,
                    "text_page": node_by_id[text_node_id].page_id,
                },
            )
            _add_edge(
                edges,
                seen,
                text_node_id,
                media_node_id,
                "text",
                "media",
                "media_near_text",
                weight=weight,
                evidence={"source": "mineru_layout", "reason": "reverse edge for bidirectional near relation"},
                metadata={"media_id": media.media_id, "chunk_id": chunk_id},
            )

    return edges


def validate_phase1_outputs(
    working_dir: str | Path,
    node_path: str | Path | None = None,
    edge_path: str | Path | None = None,
    entity_file: str | Path = "entity.jsonl",
) -> dict[str, Any]:
    working = Path(working_dir)
    node_path = Path(node_path or working / "mm_nodes.jsonl")
    edge_path = Path(edge_path or working / "mm_edges_seed.jsonl")
    chunks = load_dataclasses(working / "mm_chunk.json", MMChunk)
    media_items = load_dataclasses(working / "mm_media.json", MMMedia) if (working / "mm_media.json").exists() else []
    entity_path = Path(entity_file)
    if not entity_path.is_absolute():
        entity_path = working / entity_path
    entities = read_jsonl(entity_path) if entity_path.exists() else []
    nodes = read_jsonl(node_path)
    edges = read_jsonl(edge_path)

    by_type = defaultdict(list)
    for node in nodes:
        by_type[node.get("node_type")].append(node)
    media_nodes_by_id = {node.get("raw_ref", {}).get("media_id"): node for node in by_type["media"]}
    text_nodes_by_chunk_id = {node.get("raw_ref", {}).get("chunk_id"): node for node in by_type["text"]}
    entity_nodes_by_name = {node.get("raw_ref", {}).get("entity_name"): node for node in by_type["entity"]}
    pages = _collect_pages(chunks, media_items, extra_pages=_collect_mineru_pages(working))
    edge_types = {edge.get("edge_type") for edge in edges}

    errors = []
    warnings = []
    for media in (item for item in media_items if is_indexable_media(item)):
        node = media_nodes_by_id.get(media.media_id)
        if not node:
            errors.append(f"missing media node for {media.media_id}")
            continue
        if not node.get("node_id"):
            errors.append(f"media node has empty node_id for {media.media_id}")
        if not str(node.get("text_for_embedding") or "").strip():
            errors.append(f"media node has empty text_for_embedding for {media.media_id}")
        raw_ref = node.get("raw_ref") or {}
        if raw_ref.get("media_id") != media.media_id:
            errors.append(f"media node missing raw_ref.media_id for {media.media_id}")
        if media.modality == "image" and "path" not in raw_ref:
            errors.append(f"image media node missing raw_ref.path for {media.media_id}")
        if media.modality == "table":
            missing_table_fields = [
                field for field in ("table_markdown", "table_html") if field not in raw_ref
            ]
            if missing_table_fields:
                errors.append(
                    f"table media node missing preserved fields {missing_table_fields} for {media.media_id}"
                )
            elif not (raw_ref.get("table_markdown") or raw_ref.get("table_html")):
                fallback_evidence = (
                    raw_ref.get("path")
                    or node.get("ocr_text")
                    or node.get("caption")
                    or node.get("summary")
                )
                if fallback_evidence:
                    warnings.append(
                        f"visual-only table has no markdown/html; using image/text evidence for {media.media_id}"
                    )
                else:
                    errors.append(
                        f"table media node has neither structured nor visual/text evidence for {media.media_id}"
                    )
    for page in pages:
        if not any(node.get("node_type") == "page" and node.get("page_id") == page for node in nodes):
            errors.append(f"missing page node for page {page}")
    for chunk in chunks:
        if chunk.modality == "text" and chunk.chunk_id not in text_nodes_by_chunk_id:
            errors.append(f"missing text node for {chunk.chunk_id}")
    for entity in entities:
        name = entity.get("entity_name")
        if name and name not in entity_nodes_by_name:
            errors.append(f"missing entity node for {name}")
    if "page_contains_node" not in edge_types:
        errors.append("missing edge_type page_contains_node")

    # A text-only document has no media node and therefore cannot produce a
    # media_near_text edge.  Require the edge only when at least one indexable
    # media item actually resolves to a text chunk that the builder can link.
    expects_media_near_text = False
    for media in (item for item in media_items if is_indexable_media(item)):
        nearby = list(media.nearby_chunk_ids or [])
        if not nearby:
            nearby = _attached_text_chunks(media.media_id, chunks)
        if any(chunk_id in text_nodes_by_chunk_id for chunk_id in nearby):
            expects_media_near_text = True
            break
    if expects_media_near_text and "media_near_text" not in edge_types:
        errors.append("missing edge_type media_near_text")

    report = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "text_nodes": len(by_type["text"]),
            "media_nodes": len(by_type["media"]),
            "page_nodes": len(by_type["page"]),
            "entity_nodes": len(by_type["entity"]),
        },
    }
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _build_document_node(doc_id: str, chunks: list[MMChunk], media_items: list[MMMedia]) -> MMNode:
    return MMNode(
        node_id=f"{doc_id}::document",
        doc_id=doc_id,
        node_type="document",
        page_id=None,
        text_for_embedding=doc_id,
        raw_ref={"doc_id": doc_id},
        metadata={"text_chunk_count": len(chunks), "media_count": len([item for item in media_items if is_indexable_media(item)])},
        source="mineru",
    )


def _build_page_node(doc_id: str, page: int, chunks: list[MMChunk], media_items: list[MMMedia]) -> MMNode:
    page_chunks = [
        chunk
        for chunk in chunks
        if chunk.page_start is not None and chunk.page_start <= page <= (chunk.page_end or chunk.page_start)
    ]
    page_chunks.sort(key=lambda item: item.order)
    page_text = " ".join(chunk.text for chunk in page_chunks[:8] if chunk.text).strip()
    media_count = len([item for item in media_items if item.page == page and is_indexable_media(item)])
    fallback = f"Page {page} of document {doc_id}. Contains {len(page_chunks)} text chunks and {media_count} media items."
    return MMNode(
        node_id=_page_node_id(doc_id, page),
        doc_id=doc_id,
        node_type="page",
        page_id=page,
        text_for_embedding=page_text or fallback,
        raw_ref={"page": page},
        metadata={"text_chunk_count": len(page_chunks), "media_count": media_count},
        source="mineru",
    )


def _build_text_node(chunk: MMChunk) -> MMNode:
    local_id = chunk.chunk_id or chunk.hash_code
    return MMNode(
        node_id=f"{chunk.doc_id}::text::{_safe_id(local_id)}",
        doc_id=chunk.doc_id,
        node_type="text",
        page_id=chunk.page_start,
        text_for_embedding=chunk.text or f"text chunk {local_id}",
        raw_ref={"chunk_id": chunk.chunk_id, "hash_code": chunk.hash_code},
        metadata={
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_title": chunk.section_title,
            "source_path": chunk.source_path,
            "order": chunk.order,
            "attached_media_ids": chunk.attached_media_ids,
        },
        bbox=chunk.bbox,
        source="mineru",
    )


def _build_media_node(media: MMMedia, chunk_by_id: dict[str, MMChunk]) -> MMNode:
    text_parts = [media.caption, media.footnote, media.ocr_text, media.summary, media.table_markdown, media.table_html]
    text_for_embedding = "\n".join(part.strip() for part in text_parts if str(part or "").strip())
    if not text_for_embedding:
        nearby_text = " ".join(chunk_by_id[chunk_id].text for chunk_id in media.nearby_chunk_ids if chunk_id in chunk_by_id)
        text_for_embedding = nearby_text.strip()
    if not text_for_embedding:
        text_for_embedding = f"media on page {media.page}, type={media.modality}, id={media.media_id}"

    raw_ref = {
        "media_id": media.media_id,
        "path": _portable_path(media.path),
        "original_type": media.original_type,
        "mapped_type": media.mapped_type,
        "type": media.type,
        "footnote": media.footnote,
        "table_html": media.table_html,
        "table_markdown": media.table_markdown,
    }
    return MMNode(
        node_id=f"{media.doc_id}::media::{_safe_id(media.media_id)}",
        doc_id=media.doc_id,
        node_type="media",
        page_id=media.page,
        text_for_embedding=text_for_embedding,
        raw_ref=raw_ref,
        metadata={
            "media_type": media.mapped_type,
            "modality": media.modality,
            "nearby_chunk_ids": media.nearby_chunk_ids,
            "attached_entity_names": media.attached_entity_names,
            "attach_scores": media.attach_scores,
            "table_info": _table_info(media),
        },
        bbox=media.bbox,
        caption=media.caption or "",
        ocr_text=media.ocr_text or "",
        summary=media.summary or "",
        source="mineru",
    )


def _build_entity_node(entity: dict[str, Any], doc_id: str, chunk_by_hash: dict[str, MMChunk]) -> MMNode | None:
    entity_name = str(entity.get("entity_name") or "").strip()
    if not entity_name:
        return None
    source_ids = [item for item in str(entity.get("source_id") or "").split("|") if item]
    pages = sorted(
        {
            page
            for source_id in source_ids
            for page in [chunk_by_hash[source_id].page_start if source_id in chunk_by_hash else None]
            if page is not None
        }
    )
    description = str(entity.get("description") or "").strip()
    text_for_embedding = f"{entity_name}\n{description}".strip() or entity_name
    return MMNode(
        node_id=f"{doc_id}::entity::{_normalize_entity_id(entity_name)}",
        doc_id=doc_id,
        node_type="entity",
        page_id=pages[0] if len(pages) == 1 else None,
        text_for_embedding=text_for_embedding,
        raw_ref={
            "entity_name": entity_name,
            "source_id": entity.get("source_id", ""),
        },
        metadata={
            "entity_type": entity.get("entity_type"),
            "degree": entity.get("degree"),
            "pages": pages,
        },
        source="leanrag",
    )


def _collect_pages(
    chunks: list[MMChunk],
    media_items: list[MMMedia],
    extra_pages: list[int] | None = None,
) -> list[int]:
    pages = set()
    for chunk in chunks:
        if chunk.page_start is None:
            continue
        end = chunk.page_end or chunk.page_start
        pages.update(range(chunk.page_start, end + 1))
    pages.update(item.page for item in media_items if item.page is not None)
    pages.update(extra_pages or [])
    return sorted(page for page in pages if isinstance(page, int) and page > 0)


def _collect_mineru_pages(working_dir: Path) -> list[int]:
    pages = set()
    for path in sorted((working_dir / "mineru_output").rglob("*content_list*.json")):
        try:
            with path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            continue
        items = data if isinstance(data, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("page_idx") not in (None, ""):
                try:
                    pages.add(int(item["page_idx"]) + 1)
                except (TypeError, ValueError):
                    pass
            else:
                value = item.get("page") if item.get("page") not in (None, "") else item.get("page_no")
                try:
                    if value not in (None, ""):
                        pages.add(int(value))
                except (TypeError, ValueError):
                    pass
    return sorted(page for page in pages if page > 0)


def _attached_text_chunks(media_id: str, chunks: list[MMChunk]) -> list[str]:
    return [chunk.chunk_id for chunk in chunks if media_id in (chunk.attached_media_ids or [])]


def _add_edge(
    edges: list[MMEdge],
    seen: set[tuple[str, str, str]],
    src: str,
    dst: str,
    src_type: str,
    dst_type: str,
    edge_type: str,
    weight: float = 1.0,
    evidence: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    key = (src, dst, edge_type)
    if key in seen:
        return
    seen.add(key)
    edge_hash = hashlib.md5("|".join(key).encode("utf-8")).hexdigest()[:16]
    edges.append(
        MMEdge(
            edge_id=f"edge_{edge_hash}",
            src=src,
            dst=dst,
            src_type=src_type,
            dst_type=dst_type,
            edge_type=edge_type,  # type: ignore[arg-type]
            weight=weight,
            direction="directed",
            evidence=evidence or {},
            metadata=metadata or {},
        )
    )


def _infer_doc_id(chunks: list[MMChunk], media_items: list[MMMedia], entities: list[dict[str, Any]]) -> str:
    if chunks:
        return chunks[0].doc_id
    if media_items:
        return media_items[0].doc_id
    for entity in entities:
        if entity.get("doc_id"):
            return str(entity["doc_id"])
    return "document"


def _page_node_id(doc_id: str, page: int) -> str:
    return f"{doc_id}::page::{page}"


def _safe_id(value: Any) -> str:
    text = str(value or "").strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _portable_path(value: Any) -> str:
    # Store JSON paths with forward slashes so artifacts survive Windows/Linux moves.
    text = str(value or "").strip()
    return text.replace("\\", "/")


def _normalize_entity_id(name: str) -> str:
    text = name.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-\u4e00-\u9fff]+", "", text)
    return text or hashlib.md5(name.encode("utf-8")).hexdigest()[:16]


def _infer_media_type(media: MMMedia) -> str:
    return str(media.mapped_type or media.type or media.modality or "generic")


def _table_info(media: MMMedia) -> dict[str, Any]:
    if media.modality != "table":
        return {}
    return parse_table(media.table_html, media.table_markdown)


def _count_nodes_edges(nodes: list[MMNode], edges: list[MMEdge]) -> dict[str, Any]:
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    for node in nodes:
        node_counts[node.node_type] += 1
    for edge in edges:
        edge_counts[edge.edge_type] += 1
    return {"nodes": dict(node_counts), "edges": dict(edge_counts), "total_nodes": len(nodes), "total_edges": len(edges)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 1 multimodal nodes and seed edges for a LeanRAG workspace.")
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--node_file", default="mm_nodes.jsonl")
    parser.add_argument("--edge_seed_file", default="mm_edges_seed.jsonl")
    parser.add_argument("--trace_file", default="mm_node_build_trace.json")
    parser.add_argument("--no_validate", action="store_true")
    args = parser.parse_args()
    trace = build_phase1_mm_graph(
        working_dir=args.working_dir,
        node_file=args.node_file,
        edge_seed_file=args.edge_seed_file,
        trace_file=args.trace_file,
        validate=not args.no_validate,
    )
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
