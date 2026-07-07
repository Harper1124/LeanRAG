from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def run_vlm_reasoner(
    question: str,
    visual_nodes: list[dict[str, Any]],
    vlm_func: Callable | None,
    max_images: int = 4,
) -> list[dict[str, Any]]:
    selected = visual_nodes[: max(0, int(max_images))]
    if not selected:
        return []
    node_ids = [node.get("node_id") for node in selected]
    image_paths = [str((node.get("raw_ref") or {}).get("path") or "") for node in selected]
    valid_paths = [path for path in image_paths if path]
    context = _visual_context(selected)
    call = {
        "node_ids": node_ids,
        "image_paths": valid_paths,
        "context_preview": context[:1000],
        "vlm_answer": "",
        "error": None,
    }
    if not valid_paths:
        call["error"] = "no_image_path"
        return [call]
    if not vlm_func:
        call["error"] = "vlm_func_missing"
        return [call]
    try:
        call["vlm_answer"] = str(vlm_func(query=question, context=context, image_paths=valid_paths))
    except TypeError:
        try:
            call["vlm_answer"] = str(vlm_func(prompt=f"{question}\n\n{context}", image_paths=valid_paths))
        except Exception as exc:
            call["error"] = str(exc)
    except Exception as exc:
        call["error"] = str(exc)
    return [call]


def _visual_context(nodes: list[dict[str, Any]]) -> str:
    parts = []
    for node in nodes:
        raw_ref = node.get("raw_ref") or {}
        metadata = node.get("metadata") or {}
        path = raw_ref.get("path") or ""
        parts.append(
            "\n".join(
                [
                    f"node_id: {node.get('node_id')}",
                    f"page_id: {node.get('page_id')}",
                    f"media_id: {raw_ref.get('media_id', '')}",
                    f"path: {Path(path).name if path else ''}",
                    f"media_type: {metadata.get('media_type', '')}",
                    f"caption: {node.get('caption') or metadata.get('caption', '')}",
                    f"ocr_text: {node.get('ocr_text') or metadata.get('ocr_text', '')}",
                    f"summary: {node.get('summary') or metadata.get('summary', '')}",
                    f"nearby_chunk_ids: {metadata.get('nearby_chunk_ids', [])}",
                ]
            )
        )
    return "\n\n".join(parts)
