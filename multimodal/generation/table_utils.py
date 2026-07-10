from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


def hydrate_table_candidate(candidate: dict[str, Any], media_by_id: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    item = dict(candidate)
    raw_ref = dict(item.get("raw_ref") or {})
    metadata = dict(item.get("metadata") or {})
    table_info = dict(metadata.get("table_info") or {})
    media_id = raw_ref.get("media_id") or _media_id_from_node_id(str(item.get("node_id") or ""))
    source = (media_by_id or {}).get(media_id or "")
    if source:
        raw_ref.setdefault("media_id", source.get("media_id"))
        raw_ref.setdefault("path", source.get("path"))
        raw_ref["table_html"] = raw_ref.get("table_html") or source.get("table_html") or ""
        raw_ref["table_markdown"] = raw_ref.get("table_markdown") or source.get("table_markdown") or ""
        for field in ("caption", "ocr_text", "summary"):
            if not item.get(field) and source.get(field):
                item[field] = source.get(field)
    parsed = parse_table(raw_ref.get("table_html") or "", raw_ref.get("table_markdown") or "")
    if parsed["table_parse_available"]:
        table_info.update(parsed)
    else:
        table_info.setdefault("format", "none")
        table_info.setdefault("cells", [])
        table_info.setdefault("n_rows", 0)
        table_info.setdefault("n_cols", 0)
        table_info.setdefault("parse_confidence", 0.0)
        table_info.setdefault("table_parse_available", False)
    metadata["table_info"] = table_info
    item["raw_ref"] = raw_ref
    item["metadata"] = metadata
    return item


def parse_table(table_html: str = "", table_markdown: str = "") -> dict[str, Any]:
    html = str(table_html or "").strip()
    markdown = str(table_markdown or "").strip()
    if html:
        cells, n_rows, n_cols = _parse_html_table(html)
        if cells:
            return {
                "format": "html",
                "cells": cells,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "parse_confidence": 1.0,
                "table_parse_available": True,
            }
    if markdown:
        cells, n_rows, n_cols = _parse_markdown_table(markdown)
        if cells:
            return {
                "format": "markdown",
                "cells": cells,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "parse_confidence": 0.8,
                "table_parse_available": True,
            }
    return {
        "format": "none",
        "cells": [],
        "n_rows": 0,
        "n_cols": 0,
        "parse_confidence": 0.0,
        "table_parse_available": False,
    }


def has_structured_table(candidate: dict[str, Any]) -> bool:
    raw_ref = candidate.get("raw_ref") or {}
    metadata = candidate.get("metadata") or {}
    table_info = metadata.get("table_info") or {}
    if table_info.get("table_parse_available") and (table_info.get("cells") or table_info.get("n_rows")):
        return True
    parsed = parse_table(raw_ref.get("table_html") or "", raw_ref.get("table_markdown") or "")
    return bool(parsed.get("table_parse_available") and (parsed.get("cells") or parsed.get("n_rows")))


def load_mm_media_by_id(working_dir: str | None) -> dict[str, dict[str, Any]]:
    if not working_dir:
        return {}
    path = Path(working_dir) / "mm_media.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, list):
        return {}
    return {str(item.get("media_id")): item for item in data if isinstance(item, dict) and item.get("media_id")}


def _media_id_from_node_id(node_id: str) -> str:
    marker = "::media::"
    return node_id.split(marker, 1)[1] if marker in node_id else ""


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell.strip() for cell in self._row):
                self.rows.append(self._row)
            self._row = None


def _parse_html_table(html: str) -> tuple[list[dict[str, Any]], int, int]:
    parser = _HTMLTableParser()
    try:
        parser.feed(html)
    except Exception:
        return [], 0, 0
    return _rows_to_cells(parser.rows)


def _parse_markdown_table(markdown: str) -> tuple[list[dict[str, Any]], int, int]:
    rows = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        rows.append(cells)
    return _rows_to_cells(rows)


def _rows_to_cells(rows: list[list[str]]) -> tuple[list[dict[str, Any]], int, int]:
    cells = []
    n_cols = max((len(row) for row in rows), default=0)
    for row_idx, row in enumerate(rows):
        for col_idx, text in enumerate(row):
            if str(text or "").strip():
                cells.append({"row": row_idx, "col": col_idx, "text": str(text).strip()})
    return cells, len(rows), n_cols
