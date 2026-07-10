from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def try_deterministic_answer(question: str, global_config: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    text = str(question or "").lower()
    if not re.search(r"\bhow many\b", text):
        return None, {"enabled": True, "matched": False, "reason": "not_counting_question"}
    target = _count_target(text)
    if not target:
        return None, {"enabled": True, "matched": False, "reason": "unsupported_count_target"}
    scope = _count_scope(text, global_config)
    if not scope:
        return None, {"enabled": True, "matched": False, "reason": "unsupported_count_scope", "target": target}
    media = _load_media(global_config.get("working_dir"))
    if not media:
        return None, {"enabled": True, "matched": False, "reason": "media_missing", "target": target, "scope": scope}
    count, used = _count_media(media, target, scope)
    return str(count), {
        "enabled": True,
        "matched": True,
        "rule": "media_count",
        "target": target,
        "scope": scope,
        "used_media_ids": [item.get("media_id") for item in used],
        "used_pages": sorted({item.get("page") for item in used if item.get("page") is not None}),
    }


def _count_target(text: str) -> str | None:
    if re.search(r"\btables?\b", text):
        return "table"
    if re.search(r"\bbar plots?\b", text):
        return "bar_plot"
    if re.search(r"\bline plots?\b", text):
        return "line_plot"
    if re.search(r"\bplots?\b", text):
        return "plot"
    if re.search(r"\b(figures?|images?|pictures?)\b", text):
        return "figure"
    if re.search(r"\bcharts?\b", text):
        return "chart"
    return None


def _count_scope(text: str, global_config: dict[str, Any]) -> dict[str, Any] | None:
    page_range = re.search(r"\bpages?\s*(\d+)\s*[-–]\s*(\d+)\b", text)
    if page_range:
        start, end = int(page_range.group(1)), int(page_range.group(2))
        if end < start:
            start, end = end, start
        return {"type": "page_range", "start": start, "end": end}
    single_page = re.search(r"\bpages?\s*(\d+)\b", text)
    if single_page:
        page = int(single_page.group(1))
        return {"type": "page_range", "start": page, "end": page}
    if "appendix" in text:
        start = _appendix_start_page(global_config.get("working_dir"))
        if start:
            return {"type": "page_range", "start": start, "end": 10**9, "label": "appendix"}
        return None
    if re.search(r"\b(in|within|throughout)\s+(this\s+)?(paper|document)\b", text):
        return {"type": "document"}
    return None


def _count_media(media: list[dict[str, Any]], target: str, scope: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    scoped = [item for item in media if _in_scope(item, scope)]
    used = [item for item in scoped if _matches_target(item, target)]
    return len(used), used


def _in_scope(item: dict[str, Any], scope: dict[str, Any]) -> bool:
    if scope.get("type") == "document":
        return True
    if scope.get("type") == "page_range":
        page = item.get("page")
        try:
            page_num = int(page)
        except (TypeError, ValueError):
            return False
        return int(scope["start"]) <= page_num <= int(scope["end"])
    return False


def _matches_target(item: dict[str, Any], target: str) -> bool:
    modality = str(item.get("modality") or "").lower()
    text = " ".join(
        str(item.get(field) or "") for field in ("caption", "ocr_text", "summary", "table_markdown", "table_html", "media_id")
    ).lower()
    if target == "table":
        return modality == "table"
    if modality != "image":
        return False
    if target == "figure":
        return True
    if target == "chart":
        return bool(re.search(r"\b(chart|plot|axis|bar|line graph|curve)\b", text))
    if target == "plot":
        return bool(re.search(r"\b(plot|line graph|bar chart|curve)\b", text))
    if target == "line_plot":
        return bool(re.search(r"\b(line plot|line graph|curve)\b", text))
    if target == "bar_plot":
        return bool(re.search(r"\b(bar plot|bar chart|histogram)\b", text))
    return False


def _appendix_start_page(working_dir: str | None) -> int | None:
    if not working_dir:
        return None
    path = Path(working_dir) / "mm_chunk.json"
    if not path.exists():
        return None
    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    pages = []
    for chunk in chunks if isinstance(chunks, list) else []:
        text = " ".join(str(chunk.get(field) or "") for field in ("section_title", "text")).lower()
        if re.search(r"\bappendix\b", text):
            page = chunk.get("page_start") or chunk.get("page_end")
            try:
                pages.append(int(page))
            except (TypeError, ValueError):
                pass
    return min(pages) if pages else None


def _load_media(working_dir: str | None) -> list[dict[str, Any]]:
    if not working_dir:
        return []
    path = Path(working_dir) / "mm_media.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
