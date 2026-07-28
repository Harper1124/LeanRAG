from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..generation.table_utils import parse_table
from ..schema import MMMedia
from .prompts import CHART_PROMPT, IMAGE_PROMPT, TABLE_PROMPT


NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?%?")


class ImageProcessor:
    media_type = "image"

    def __init__(self, vlm_func: Callable | None = None) -> None:
        self.vlm_func = vlm_func

    def process(self, media: MMMedia, media_context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        structured = _image_file_metadata(media.path)
        response = _call_json_model(
            self.vlm_func,
            IMAGE_PROMPT.replace("{context}", _json_for_prompt(media_context)),
            image_path=media.path,
        )
        semantic = {
            "visual_facts": _string_list(response.get("visual_facts")),
            "visible_text": _string_list(response.get("visible_text")),
            "objects": _json_list(response.get("objects")),
            "spatial_relations": _json_list(response.get("spatial_relations")),
            "image_type": _choice(response.get("image_type"), {"photo", "diagram", "screenshot", "map", "other"}, "other"),
            "caption_consistency": str(response.get("caption_consistency") or ""),
            "grounded_summary": str(response.get("grounded_summary") or ""),
            "uncertain_items": _string_list(response.get("uncertain_items")),
            "confidence": _confidence(response.get("confidence")),
        }
        if not response:
            semantic["uncertain_items"].append("VLM analysis unavailable or returned invalid JSON")
        confidence = {
            "overall": semantic["confidence"],
            "visual_model_available": bool(self.vlm_func),
            "model_response_valid": bool(response),
            "evidence_separation": "visual fields use image pixels; caption/OCR/nearby text remain in media_context",
        }
        return structured, semantic, confidence


class ChartProcessor:
    media_type = "chart"

    def __init__(self, vlm_func: Callable | None = None, ocr_func: Callable | None = None) -> None:
        self.vlm_func = vlm_func
        self.ocr_func = ocr_func

    def process(self, media: MMMedia, media_context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        structured = _parse_chart_evidence(media, self.ocr_func)
        response = _call_json_model(
            self.vlm_func,
            CHART_PROMPT.replace("{structured}", _json_for_prompt(structured)).replace(
                "{context}", _json_for_prompt(media_context)
            ),
            image_path=media.path,
        )
        allowed_numbers = set(structured["numeric_tokens"])
        semantic = {
            "chart_type": _safe_numeric_text(response.get("chart_type"), allowed_numbers),
            "title": structured["title"] or _safe_numeric_text(response.get("title"), allowed_numbers),
            "x_axis": structured["x_axis"],
            "y_axis": structured["y_axis"],
            "legends": structured["legends"],
            "series": _safe_numeric_items(response.get("series"), allowed_numbers),
            # Readable values are program-derived only; model values are intentionally ignored.
            "readable_data_points": structured["readable_data_points"],
            "qualitative_trends": _validated_trends(response.get("qualitative_trends"), allowed_numbers),
            "extrema_and_intersections": _safe_numeric_items(response.get("extrema_and_intersections"), allowed_numbers),
            "caption_consistency": _safe_numeric_text(
                response.get("caption_consistency"),
                allowed_numbers.union(NUMBER_RE.findall(media.caption or "")),
            ),
            "unreadable_regions": _safe_numeric_items(response.get("unreadable_regions"), allowed_numbers),
            "grounded_summary": _safe_numeric_text(response.get("grounded_summary"), allowed_numbers),
            "confidence": _confidence(response.get("confidence")),
        }
        if not response:
            semantic["unreadable_regions"].append("VLM interpretation unavailable or returned invalid JSON")
        confidence = {
            "overall": semantic["confidence"],
            "programmatic_parse": structured["parse_confidence"],
            "visual_model_available": bool(self.vlm_func),
            "model_response_valid": bool(response),
        }
        return structured, semantic, confidence


class TableProcessor:
    media_type = "table"

    def __init__(self, llm_func: Callable | None = None) -> None:
        self.llm_func = llm_func

    def process(self, media: MMMedia, media_context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        structured = _parse_table_evidence(media)
        response = _call_json_model(
            self.llm_func,
            TABLE_PROMPT.replace("{structured}", _json_for_prompt(structured)).replace(
                "{context}", _json_for_prompt(media_context)
            ),
        )
        semantic = _validated_table_semantics(response, structured, media.caption)
        confidence = {
            "overall": semantic["confidence"],
            "source_parse": structured["parse_confidence"],
            "language_model_available": bool(self.llm_func),
            "model_response_valid": bool(response),
            "numeric_provenance_complete": _table_numeric_provenance_complete(semantic, structured),
        }
        return structured, semantic, confidence


def _image_file_metadata(path: str) -> dict[str, Any]:
    result: dict[str, Any] = {"path_available": bool(path and Path(path).exists()), "width": None, "height": None}
    if not result["path_available"]:
        return result
    try:
        from PIL import Image

        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            result["format"] = image.format or ""
    except Exception:
        result["format"] = ""
    return result


def _parse_chart_evidence(media: MMMedia, ocr_func: Callable | None = None) -> dict[str, Any]:
    ocr = str(media.ocr_text or "").strip()
    ocr_source = "mineru_ocr" if ocr else ""
    if not ocr and media.path:
        ocr = _run_chart_ocr(media.path, ocr_func)
        ocr_source = "program_ocr" if ocr else "unavailable"
    lines = [" ".join(line.split()) for line in ocr.splitlines() if line.strip()]
    numeric_tokens = list(dict.fromkeys(token for line in lines for token in NUMBER_RE.findall(line)))
    x_lines = [line for line in lines if re.search(r"\bx[- ]?axis\b|horizontal axis", line, re.I)]
    y_lines = [line for line in lines if re.search(r"\by[- ]?axis\b|vertical axis", line, re.I)]
    legend_lines = [line for line in lines if re.search(r"\blegend\b|\bseries\b", line, re.I)]
    data_points = [
        {"text": line, "values": NUMBER_RE.findall(line), "source": "ocr"}
        for line in lines
        if NUMBER_RE.search(line)
    ]
    title = next((line for line in lines if not NUMBER_RE.search(line) and line not in x_lines + y_lines + legend_lines), "")
    return {
        **_image_file_metadata(media.path),
        "ocr_text": ocr,
        "ocr_source": ocr_source,
        "ocr_lines": lines,
        "numeric_tokens": numeric_tokens,
        "title": title,
        "x_axis": {"candidates": x_lines, "source": "ocr"} if x_lines else {},
        "y_axis": {"candidates": y_lines, "source": "ocr"} if y_lines else {},
        "legends": [{"text": line, "source": "ocr"} for line in legend_lines],
        "readable_data_points": data_points,
        "parse_confidence": 0.8 if lines else 0.0,
    }


def _parse_table_evidence(media: MMMedia) -> dict[str, Any]:
    parsed = parse_table(media.table_html, media.table_markdown)
    cells = parsed.get("cells") or []
    lookup = {(int(cell["row"]), int(cell["col"])): str(cell.get("text") or "") for cell in cells}
    n_rows, n_cols = int(parsed.get("n_rows") or 0), int(parsed.get("n_cols") or 0)
    column_keys = [lookup.get((0, col), "") for col in range(n_cols)] if n_rows else []
    row_keys = [lookup.get((row, 0), "") for row in range(1, n_rows)] if n_cols else []
    numeric_cells = []
    for cell in cells:
        tokens = NUMBER_RE.findall(str(cell.get("text") or ""))
        if tokens:
            numeric_cells.append({**cell, "numeric_tokens": tokens})
    units = list(dict.fromkeys(
        match.group(0)
        for text in column_keys + row_keys
        for match in re.finditer(r"%|[$€£¥]|\b(?:kg|g|mg|km|m|cm|mm|ms|s|min|h|hz|mhz|ghz)\b", text, re.I)
    ))
    return {
        **parsed,
        "header_hierarchy": [column_keys] if any(column_keys) else [],
        "row_keys": [value for value in row_keys if value],
        "column_keys": [value for value in column_keys if value],
        "units": units,
        "numeric_cells": numeric_cells,
        "source_kind": parsed.get("format", "none"),
    }


def _validated_table_semantics(response: dict[str, Any], structured: dict[str, Any], caption: str) -> dict[str, Any]:
    lookup = {(int(cell["row"]), int(cell["col"])): str(cell.get("text") or "") for cell in structured.get("cells", [])}
    all_numbers = {token for value in lookup.values() for token in NUMBER_RE.findall(value)}
    important = []
    for item in _json_list(response.get("important_cells")):
        if not isinstance(item, dict):
            continue
        try:
            key = (int(item.get("row")), int(item.get("col")))
        except (TypeError, ValueError):
            continue
        value = str(item.get("value") or "")
        if key in lookup and value == lookup[key]:
            important.append({
                "row": key[0],
                "col": key[1],
                "value": value,
                "reason": _safe_numeric_text(item.get("reason"), set(NUMBER_RE.findall(value))),
            })
    comparisons = []
    for item in _json_list(response.get("comparisons")):
        if not isinstance(item, dict):
            continue
        coords = _valid_source_cells(item.get("source_cells"), lookup)
        statement = str(item.get("statement") or "")
        cited_numbers = {token for coord in coords for token in NUMBER_RE.findall(lookup[coord])}
        if coords and set(NUMBER_RE.findall(statement)).issubset(cited_numbers):
            comparisons.append({"statement": statement, "source_cells": [[row, col] for row, col in coords]})
    ambiguity = [str(item) for item in _safe_numeric_items(response.get("ambiguous_structure"), all_numbers)]
    if not structured.get("table_parse_available"):
        ambiguity.append("HTML/Markdown table source could not be parsed")
    title_and_purpose = _safe_numeric_text(response.get("title_and_purpose"), all_numbers)
    if not title_and_purpose:
        title_and_purpose = _safe_numeric_text(re.sub(r"^\s*table\s+\d+[a-z]?\s*[:.\-]?\s*", "", caption, flags=re.I), all_numbers)
    return {
        "title_and_purpose": title_and_purpose,
        "header_hierarchy": _safe_numeric_items(response.get("header_hierarchy") or structured.get("header_hierarchy"), all_numbers),
        "row_keys": _safe_numeric_items(response.get("row_keys") or structured.get("row_keys"), all_numbers),
        "column_keys": _safe_numeric_items(response.get("column_keys") or structured.get("column_keys"), all_numbers),
        "units": _safe_numeric_items(response.get("units") or structured.get("units"), all_numbers),
        "important_cells": important,
        "comparisons": comparisons,
        "grounded_summary": _safe_numeric_text(response.get("grounded_summary"), all_numbers),
        "ambiguous_structure": list(dict.fromkeys(ambiguity)),
        "confidence": _confidence(response.get("confidence", structured.get("parse_confidence", 0.0))),
    }


def _valid_source_cells(value: Any, lookup: dict[tuple[int, int], str]) -> list[tuple[int, int]]:
    result = []
    for coord in value if isinstance(value, list) else []:
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            continue
        try:
            key = (int(coord[0]), int(coord[1]))
        except (TypeError, ValueError):
            continue
        if key in lookup and key not in result:
            result.append(key)
    return result


def _run_chart_ocr(path: str, ocr_func: Callable | None) -> str:
    if ocr_func:
        try:
            return str(ocr_func(path) or "").strip()
        except Exception:
            return ""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(path) as image:
            return str(pytesseract.image_to_string(image) or "").strip()
    except Exception:
        return ""


def _table_numeric_provenance_complete(semantic: dict[str, Any], structured: dict[str, Any]) -> bool:
    lookup = {(int(cell["row"]), int(cell["col"])): str(cell.get("text") or "") for cell in structured.get("cells", [])}
    for item in semantic.get("important_cells", []):
        if lookup.get((item["row"], item["col"])) != item["value"]:
            return False
    for item in semantic.get("comparisons", []):
        coords = [(coord[0], coord[1]) for coord in item.get("source_cells", [])]
        cited = {token for coord in coords for token in NUMBER_RE.findall(lookup.get(coord, ""))}
        if not set(NUMBER_RE.findall(item.get("statement", ""))).issubset(cited):
            return False
    return True


def _validated_trends(value: Any, allowed_numbers: set[str]) -> list[dict[str, Any]]:
    result = []
    for item in _json_list(value):
        if not isinstance(item, dict) or not str(item.get("series") or "").strip() or not str(item.get("evidence") or "").strip():
            continue
        if not _numbers_supported(item, allowed_numbers):
            continue
        result.append({
            "series": str(item["series"]),
            "trend": str(item.get("trend") or ""),
            "evidence": str(item["evidence"]),
        })
    return result


def _safe_numeric_items(value: Any, allowed_numbers: set[str]) -> list[Any]:
    return [item for item in _json_list(value) if _numbers_supported(item, allowed_numbers)]


def _safe_numeric_text(value: Any, allowed_numbers: set[str]) -> str:
    text = str(value or "")
    return text if set(NUMBER_RE.findall(text)).issubset(allowed_numbers) else ""


def _numbers_supported(value: Any, allowed_numbers: set[str]) -> bool:
    if isinstance(value, dict):
        return all(_numbers_supported(item, allowed_numbers) for key, item in value.items() if key not in {"row", "col", "confidence"})
    if isinstance(value, list):
        return all(_numbers_supported(item, allowed_numbers) for item in value)
    if isinstance(value, str):
        return set(NUMBER_RE.findall(value)).issubset(allowed_numbers)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value) in allowed_numbers
    return True


def _call_json_model(func: Callable | None, prompt: str, image_path: str = "") -> dict[str, Any]:
    if not func:
        return {}
    kwargs = {"prompt": prompt, "query": prompt, "temperature": 0.0}
    if image_path:
        kwargs.update({"image_paths": [image_path], "image_path": image_path})
    try:
        response = func(**kwargs)
    except TypeError:
        try:
            response = func(prompt, [image_path]) if image_path else func(prompt)
        except Exception:
            return {}
    except Exception:
        return {}
    return _parse_json_object(response)


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except (TypeError, ValueError):
            continue
    return {}


def _json_for_prompt(value: Any, max_chars: int = 12000) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)[:max_chars]


def _json_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _choice(value: Any, allowed: set[str], default: str) -> str:
    candidate = str(value or "").lower()
    return candidate if candidate in allowed else default


def _confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
