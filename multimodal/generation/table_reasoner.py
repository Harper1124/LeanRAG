from __future__ import annotations

from typing import Any, Callable

from .table_utils import has_structured_table


TABLE_REASONER_PROMPT = """You answer using only the provided table evidence.
Return the shortest supported answer. If the evidence is insufficient, return exactly: Not answerable.

Question:
{question}

Table Evidence:
{context}
"""


def run_table_reasoner(
    question: str,
    table_nodes: list[dict[str, Any]],
    llm_func: Callable | None = None,
    max_context_chars: int = 12000,
) -> list[dict[str, Any]]:
    if not table_nodes:
        return []
    context = _table_context(table_nodes, max_context_chars)
    call = {
        "table_answer": "",
        "used_table_nodes": [node.get("node_id") for node in table_nodes],
        "table_context_preview": context[:1000],
        "error": None,
    }
    if not context.strip():
        call["error"] = "table_context_empty"
        return [call]
    if not llm_func:
        call["table_answer"] = context[:2000]
        call["error"] = "llm_func_missing"
        return [call]
    prompt = TABLE_REASONER_PROMPT.format(question=question, context=context)
    try:
        call["table_answer"] = str(llm_func(prompt))
    except TypeError:
        try:
            call["table_answer"] = str(llm_func(question, system_prompt=prompt))
        except Exception as exc:
            call["error"] = str(exc)
    except Exception as exc:
        call["error"] = str(exc)
    return [call]


def _table_context(table_nodes: list[dict[str, Any]], max_chars: int) -> str:
    parts = []
    for node in table_nodes:
        if not has_structured_table(node):
            continue
        raw_ref = node.get("raw_ref") or {}
        metadata = node.get("metadata") or {}
        table_info = metadata.get("table_info") or {}
        text = (
            raw_ref.get("table_markdown")
            or raw_ref.get("table_html")
            or _cells_text(table_info.get("cells") or [])
            or ""
        )
        if not text.strip():
            continue
        parts.append(
            "\n".join(
                [
                    f"node_id: {node.get('node_id')}",
                    f"page_id: {node.get('page_id')}",
                    f"media_id: {raw_ref.get('media_id', '')}",
                    f"media_type: {metadata.get('media_type', '')}",
                    f"table: {text}",
                ]
            )
        )
    return "\n\n".join(parts)[:max_chars]


def _cells_text(cells: list[dict[str, Any]]) -> str:
    if not cells:
        return ""
    rows = []
    for cell in cells[:200]:
        rows.append(f"r{cell.get('row')} c{cell.get('col')}: {cell.get('text', '')}")
    return "\n".join(rows)
