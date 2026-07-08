from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from multimodal.io_utils import read_json, read_jsonl, write_jsonl


CAPTION_PAT = re.compile(r"\b(Figure|Fig\.|Table|Chart|Exhibit)\b|[图表]")
REF_PAT = re.compile(
    r"as shown in|shown in Figure|shown in Fig\.|shown in Table|see Figure|see Table|"
    r"the figure below|the table below|in the chart|according to the chart|according to the table|"
    r"如图|如下图|如表|下表|上图|图中|表中",
    re.IGNORECASE,
)


def build_phase4_edges(
    working_dir: str | Path,
    node_file: str = "mm_nodes.jsonl",
    edge_seed_file: str = "mm_edges_seed.jsonl",
    edge_file: str = "mm_edges.jsonl",
) -> list[dict[str, Any]]:
    working = Path(working_dir)
    nodes = read_jsonl(working / node_file)
    edges = read_jsonl(working / edge_seed_file) if (working / edge_seed_file).exists() else []
    enhanced = enhance_edges(nodes, edges, working)
    write_jsonl(enhanced, working / edge_file)
    return enhanced


def enhance_edges(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], working_dir: str | Path | None = None) -> list[dict[str, Any]]:
    edge_list = [edge for edge in edges if edge.get("edge_type") != "node_semantic_similar_node"]
    seen = {(edge.get("src"), edge.get("dst"), edge.get("edge_type")) for edge in edge_list}
    by_type = defaultdict(list)
    for node in nodes:
        by_type[node.get("node_type")].append(node)
    chunks_by_id = _chunks_by_id(working_dir)
    entity_media = _entity_media(working_dir)

    for text in by_type["text"]:
        text_body = _node_text(text)
        for media in _same_page_media(text, by_type["media"]):
            if _caption_match(text, media, text_body):
                _add_edge(edge_list, seen, text, media, "text_caption_of_media", 1.0, "caption rule or overlap")
            if REF_PAT.search(text_body):
                _add_edge(edge_list, seen, text, media, "text_refers_to_media", 0.9, "explicit text reference")

    media_by_id = {(media.get("raw_ref") or {}).get("media_id"): media for media in by_type["media"]}
    for entity in by_type["entity"]:
        entity_name = str((entity.get("raw_ref") or {}).get("entity_name") or entity.get("text_for_embedding") or "").split("\n")[0].strip()
        if not entity_name:
            continue
        for media_id in entity_media.get(entity_name, []):
            media = media_by_id.get(media_id)
            if media:
                _add_edge(edge_list, seen, entity, media, "entity_link_media", 1.0, "entity_media.json")
        for media in by_type["media"]:
            haystack = _media_text(media)
            if _contains_name(haystack, entity_name):
                _add_edge(edge_list, seen, entity, media, "entity_link_media", 0.9, "entity name in media text")
            elif _entity_in_nearby_text(entity_name, media, chunks_by_id):
                _add_edge(edge_list, seen, entity, media, "entity_link_media", 0.6, "entity name in nearby text")

    media_by_page = defaultdict(list)
    for media in by_type["media"]:
        media_by_page[media.get("page_id")].append(media)
    for medias in media_by_page.values():
        medias.sort(key=lambda item: _order_key(item))
        for idx, media in enumerate(medias):
            for other in medias[max(0, idx - 3):idx] + medias[idx + 1:idx + 4]:
                _add_edge(edge_list, seen, media, other, "media_same_page_media", 0.4, "nearby media on same page", "undirected")

    return edge_list


def _caption_match(text: dict[str, Any], media: dict[str, Any], text_body: str) -> bool:
    media_caption = str(media.get("caption") or "").strip()
    if media_caption and _overlap(media_caption, text_body) >= 0.5:
        return True
    if not CAPTION_PAT.search(text_body):
        return False
    return _near_order(text, media)


def _same_page_media(text: dict[str, Any], media_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [media for media in media_nodes if media.get("page_id") == text.get("page_id")]


def _near_order(text: dict[str, Any], media: dict[str, Any]) -> bool:
    text_order = (text.get("metadata") or {}).get("order")
    media_order = (media.get("metadata") or {}).get("order")
    if text_order is not None and media_order is not None:
        return abs(int(text_order) - int(media_order)) <= 3
    nearby = set((media.get("metadata") or {}).get("nearby_chunk_ids") or [])
    chunk_id = (text.get("raw_ref") or {}).get("chunk_id")
    return bool(chunk_id and chunk_id in nearby) or True


def _add_edge(
    edges: list[dict[str, Any]],
    seen: set[tuple[Any, Any, Any]],
    src: dict[str, Any],
    dst: dict[str, Any],
    edge_type: str,
    weight: float,
    reason: str,
    direction: str = "directed",
) -> None:
    key = (src.get("node_id"), dst.get("node_id"), edge_type)
    if key in seen:
        return
    seen.add(key)
    edge_hash = hashlib.md5("|".join(str(item) for item in key).encode("utf-8")).hexdigest()[:16]
    edges.append(
        {
            "edge_id": f"edge_{edge_hash}",
            "src": key[0],
            "dst": key[1],
            "src_type": src.get("node_type"),
            "dst_type": dst.get("node_type"),
            "edge_type": edge_type,
            "weight": float(weight),
            "direction": direction,
            "evidence": {"source": "rule", "reason": reason},
            "metadata": {},
        }
    )


def _node_text(node: dict[str, Any]) -> str:
    return str(node.get("text_for_embedding") or node.get("caption") or node.get("summary") or "")


def _media_text(media: dict[str, Any]) -> str:
    raw_ref = media.get("raw_ref") or {}
    return "\n".join(
        str(part or "")
        for part in [
            media.get("caption"),
            media.get("ocr_text"),
            media.get("summary"),
            media.get("text_for_embedding"),
            raw_ref.get("table_markdown"),
        ]
    )


def _overlap(a: str, b: str) -> float:
    left = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", a.lower()))
    right = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", b.lower()))
    return len(left & right) / max(1, len(left))


def _contains_name(text: str, name: str) -> bool:
    return name.lower() in text.lower()


def _entity_in_nearby_text(name: str, media: dict[str, Any], chunks_by_id: dict[str, str]) -> bool:
    nearby = (media.get("metadata") or {}).get("nearby_chunk_ids") or []
    return any(_contains_name(chunks_by_id.get(chunk_id, ""), name) for chunk_id in nearby)


def _chunks_by_id(working_dir: str | Path | None) -> dict[str, str]:
    if not working_dir:
        return {}
    path = Path(working_dir) / "mm_chunk.json"
    if not path.exists():
        return {}
    try:
        return {item.get("chunk_id"): item.get("text", "") for item in read_json(path) if item.get("chunk_id")}
    except Exception:
        return {}


def _entity_media(working_dir: str | Path | None) -> dict[str, list[str]]:
    if not working_dir:
        return {}
    path = Path(working_dir) / "entity_media.json"
    if not path.exists():
        return {}
    try:
        data = read_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _order_key(node: dict[str, Any]) -> tuple[int, str]:
    order = (node.get("metadata") or {}).get("order")
    return (int(order) if order is not None else 0, str(node.get("node_id") or ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 4 typed multimodal edges.")
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--node_file", default="mm_nodes.jsonl")
    parser.add_argument("--edge_seed_file", default="mm_edges_seed.jsonl")
    parser.add_argument("--edge_file", default="mm_edges.jsonl")
    args = parser.parse_args()
    edges = build_phase4_edges(args.working_dir, args.node_file, args.edge_seed_file, args.edge_file)
    counts = defaultdict(int)
    for edge in edges:
        counts[edge.get("edge_type")] += 1
    print(json.dumps({"edge_file": str(Path(args.working_dir) / args.edge_file), "counts": dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
