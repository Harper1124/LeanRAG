from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..generation.table_utils import parse_table
from ..schema import MMMedia
from .chart_ocr import extract_chart_ocr, parse_chart_layout, text_only_ocr_result
from .prompts import (
    CHART_ANALYSIS_PROMPT,
    CHART_SUMMARY_PROMPT,
    IMAGE_PROMPT,
    TABLE_COMPARISON_PROMPT,
    TABLE_SUMMARY_PROMPT,
)


NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?%?")
TABLE_MAX_CANDIDATE_ROWS = 12
TABLE_MAX_COMPARISONS = 6
TABLE_MAX_SUMMARY_FACTS = 8
CHART_MAX_OCR_LINES = 40
CHART_MAX_TRENDS = 6
TABLE_COMPARABLE_NUMBER_RE = re.compile(
    r"^\s*([$€£¥])?\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(%|[kKmMbBtT])?\s*$"
)
TABLE_COMPARISON_OPERATORS = {">", "<", ">=", "<=", "==", "!="}
TABLE_COMPARATIVE_LANGUAGE_RE = re.compile(
    r"(?:[<>]=?|\b(?:higher|highest|lower|lowest|more|less|greater|smaller|exceed|exceeds|"
    r"outperform|outperforms|better|worse|equal|equals|same|rank|ranks|top|bottom)\b)",
    re.I,
)


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
        requested_sources = response.get("evidence_source") if isinstance(response.get("evidence_source"), dict) else {}
        visual_sources = _string_list(requested_sources.get("visual"))
        visible_evidence = semantic["visual_facts"] + semantic["visible_text"]
        semantic["evidence_source"] = {
            "visual": [item for item in visual_sources if item in visible_evidence] or visible_evidence,
            "caption": _source_quotes(
                requested_sources.get("caption"),
                str(media_context.get("direct_evidence", {}).get("caption") or ""),
                fallback_to_source=True,
            ),
            "nearby_text": _source_quotes(
                requested_sources.get("nearby_text"),
                str(media_context.get("layout_context", {}).get("nearby_text") or ""),
                fallback_to_source=True,
            ),
        }
        roles = response.get("semantic_role")
        if isinstance(roles, str):
            roles = [roles]
        allowed_roles = {"example", "architecture", "experiment", "illustration", "other"}
        semantic["semantic_role"] = [
            str(item).lower() for item in _json_list(roles) if str(item).lower() in allowed_roles
        ] or ["other"]
        semantic["semantic_confidence"] = _confidence(
            response.get("semantic_confidence", semantic["confidence"])
        )
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

    def __init__(
        self,
        vlm_func: Callable | None = None,
        ocr_func: Callable | None = None,
        ocr_config: dict[str, Any] | None = None,
        require_ocr_backend: bool = False,
    ) -> None:
        self.vlm_func = vlm_func
        self.ocr_func = ocr_func
        self.ocr_config = ocr_config or {}
        self.require_ocr_backend = require_ocr_backend

    def process(self, media: MMMedia, media_context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        structured = _parse_chart_evidence(media, self.ocr_func, self.ocr_config)
        if self.require_ocr_backend and structured.get("ocr_status") == "unavailable":
            raise RuntimeError(
                f"Chart OCR unavailable for {media.media_id}: {structured.get('ocr_error') or 'unknown error'}"
            )
        compact_evidence = _compact_chart_evidence(structured, media_context, media.caption)
        response, analysis_diagnostic = _call_json_model_diagnostic(
            self.vlm_func,
            CHART_ANALYSIS_PROMPT.replace("{evidence}", _json_for_prompt(compact_evidence)),
            image_path=media.path,
        )
        analysis_attempts = 1
        if not response and self.vlm_func:
            response, analysis_diagnostic = _call_json_model_diagnostic(
                self.vlm_func,
                CHART_ANALYSIS_PROMPT.replace("{evidence}", _json_for_prompt(compact_evidence)),
                image_path=media.path,
            )
            analysis_attempts = 2
        allowed_numbers = set(NUMBER_RE.findall(structured.get("ocr_text") or ""))
        response_grounding = response.get("chart_grounding") if isinstance(response.get("chart_grounding"), dict) else {}
        chart_grounding = {
            "visual_evidence": _safe_numeric_items(response_grounding.get("visual_evidence"), allowed_numbers),
            "ocr_evidence": _source_quotes(
                response_grounding.get("ocr_evidence"), structured.get("ocr_text") or ""
            ),
            "context_evidence": _source_quotes(
                response_grounding.get("context_evidence"), _flatten_text(media_context)
            ),
        }
        qualitative_trends = _validated_chart_trends(
            response.get("qualitative_trends"),
            allowed_numbers,
            structured.get("ocr_text") or "",
            chart_grounding,
        )
        semantic = {
            "chart_type": _safe_numeric_text(response.get("chart_type"), allowed_numbers) or "unknown",
            "title": _safe_numeric_text(response.get("title"), allowed_numbers) or structured["title"] or "unknown",
            # OCR/layout output remains in structured_content as candidates;
            # semantic axis and series meaning comes from the visual model.
            "x_axis": _validated_axis(response.get("x_axis"), allowed_numbers),
            "y_axis": _validated_axis(response.get("y_axis"), allowed_numbers),
            "legends": _safe_numeric_items(response.get("legends"), allowed_numbers) or ["unknown"],
            "series": _safe_numeric_items(response.get("series"), allowed_numbers) or ["unknown"],
            "qualitative_trends": qualitative_trends,
            "extrema_and_intersections": _safe_numeric_items(
                response.get("extrema_and_intersections"), allowed_numbers
            )[:4],
            "caption_consistency": _safe_numeric_text(
                response.get("caption_consistency"),
                allowed_numbers.union(NUMBER_RE.findall(media.caption or "")),
            ),
            "unreadable_regions": _safe_numeric_items(response.get("unreadable_regions"), allowed_numbers),
            "grounded_summary": "",
            "confidence": _confidence(response.get("confidence")),
            "chart_grounding": chart_grounding,
            "semantic_confidence": _confidence(
                response.get("semantic_confidence", response.get("confidence"))
            ),
        }
        if not response:
            semantic["unreadable_regions"].append("VLM interpretation unavailable or returned invalid JSON")
        summary_evidence = _chart_summary_evidence(semantic, structured)
        if self.vlm_func and summary_evidence.get("validated_claims"):
            summary_response, summary_diagnostic = _call_json_model_diagnostic(
                self.vlm_func,
                CHART_SUMMARY_PROMPT.replace("{evidence}", _json_for_prompt(summary_evidence)),
            )
        else:
            summary_response = {}
            summary_diagnostic = {
                "status": "skipped_no_grounded_claims",
                "error": "",
                "response_excerpt": "",
            }
        grounded_summary, summary_grounding = _validated_chart_summary(
            summary_response,
            summary_evidence,
            allowed_numbers,
        )
        if not grounded_summary:
            grounded_summary, summary_grounding = _fallback_chart_summary(summary_evidence)
        semantic["grounded_summary"] = grounded_summary
        semantic["summary_grounding"] = summary_grounding
        confidence = {
            "overall": semantic["confidence"],
            "programmatic_parse": structured["parse_confidence"],
            "ocr_backend": structured.get("ocr_backend"),
            "ocr_status": structured.get("ocr_status"),
            "ocr_error": structured.get("ocr_error"),
            "visual_model_available": bool(self.vlm_func),
            "model_response_valid": bool(response),
            "analysis_attempts": analysis_attempts,
            "analysis_response_status": analysis_diagnostic["status"],
            "analysis_response_error": analysis_diagnostic["error"],
            "analysis_response_excerpt": analysis_diagnostic["response_excerpt"],
            "summary_response_valid": bool(summary_response),
            "summary_response_status": summary_diagnostic["status"],
            "summary_response_error": summary_diagnostic["error"],
            "summary_response_excerpt": summary_diagnostic["response_excerpt"],
            "summary_numeric_provenance_complete": _chart_summary_numeric_provenance_complete(
                grounded_summary, summary_grounding
            ),
        }
        return structured, semantic, confidence


class TableProcessor:
    media_type = "table"

    def __init__(self, llm_func: Callable | None = None) -> None:
        self.llm_func = llm_func

    def process(self, media: MMMedia, media_context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        structured = _parse_table_evidence(media)
        compact_evidence = _compact_table_evidence(structured, media.caption)
        comparison_response = _call_json_model(
            self.llm_func,
            TABLE_COMPARISON_PROMPT.replace("{evidence}", _json_for_prompt(compact_evidence)),
        )
        semantic = _validated_table_semantics(comparison_response, structured, media.caption)
        summary_evidence = _table_summary_evidence(compact_evidence, semantic, structured)
        structurally_ambiguous = bool(structured.get("structural_ambiguity"))
        if structurally_ambiguous:
            summary_response = {}
            grounded_summary, summary_grounding = _ambiguous_table_description(
                compact_evidence, structured
            )
        else:
            summary_response = _call_json_model(
                self.llm_func,
                TABLE_SUMMARY_PROMPT.replace("{evidence}", _json_for_prompt(summary_evidence)),
            ) if self.llm_func and summary_evidence.get("grounded_facts") else {}
            grounded_summary, summary_grounding = _validated_table_summary(
                summary_response,
                structured,
                summary_evidence,
            )
            if not grounded_summary:
                grounded_summary, summary_grounding = _fallback_table_summary(summary_evidence, structured)
        semantic["grounded_summary"] = grounded_summary
        semantic["summary_grounding"] = summary_grounding
        semantic["cell_grounding"] = _dedupe_grounding(
            list(semantic.get("cell_grounding") or [])
            + [item for item in summary_grounding if item.get("source_cells")]
        )
        confidence = {
            "overall": semantic["confidence"],
            "source_parse": structured["parse_confidence"],
            "language_model_available": bool(self.llm_func),
            "model_response_valid": bool(comparison_response),
            "comparison_response_valid": bool(comparison_response),
            "summary_response_valid": bool(summary_response),
            "structurally_ambiguous": structurally_ambiguous,
            "comparisons_allowed": not structurally_ambiguous,
            "summary_mode": "descriptive_only" if structurally_ambiguous else "validated_facts",
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


def _parse_chart_evidence(
    media: MMMedia,
    ocr_func: Callable | None = None,
    ocr_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if str(media.ocr_text or "").strip():
        ocr_result = text_only_ocr_result(media.ocr_text, source="mineru_ocr")
    else:
        ocr_result = extract_chart_ocr(media.path, ocr_func=ocr_func, config=ocr_config)
    parsed = parse_chart_layout(ocr_result)
    return {
        **_image_file_metadata(media.path),
        **parsed,
    }


def _compact_chart_evidence(
    structured: dict[str, Any],
    media_context: dict[str, Any],
    caption: str,
) -> dict[str, Any]:
    """Keep chart semantics and bounded OCR geometry while dropping duplicated raw OCR payloads."""
    context = media_context if isinstance(media_context, dict) else {}
    layout = context.get("layout_context") if isinstance(context.get("layout_context"), dict) else {}
    return {
        "image": {
            "width": structured.get("width"),
            "height": structured.get("height"),
            "format": structured.get("format"),
        },
        "caption": str(caption or "")[:500],
        "document_context": {
            "section": str(layout.get("section") or "")[:200],
            "reference": str(context.get("reference_context") or "")[:800],
        },
        "program_ocr": {
            "status": structured.get("ocr_status"),
            "title": structured.get("title") or "",
            "x_axis": _compact_chart_axis(structured.get("x_axis")),
            "y_axis": _compact_chart_axis(structured.get("y_axis")),
            "legends": [
                _compact_chart_item(item) for item in (structured.get("legends") or [])[:12]
            ],
            "readable_data_points": [
                _compact_chart_item(item) for item in (structured.get("readable_data_points") or [])[:20]
            ],
            "ocr_lines": [
                _compact_chart_item(item) for item in (structured.get("ocr_lines") or [])[:CHART_MAX_OCR_LINES]
            ],
        },
    }


def _compact_chart_axis(value: Any) -> dict[str, Any]:
    axis = value if isinstance(value, dict) else {}
    return {
        "label": str(axis.get("label") or ""),
        "tick_labels": [
            _compact_chart_item(item) for item in (axis.get("tick_labels") or [])[:16]
        ],
    }


def _compact_chart_item(value: Any) -> dict[str, Any] | str:
    if not isinstance(value, dict):
        return str(value or "")
    return {
        key: value[key]
        for key in ("text", "bbox", "confidence", "role", "region")
        if key in value and value[key] not in (None, "", [])
    }


def _validated_chart_trends(
    value: Any,
    allowed_numbers: set[str],
    ocr_text: str,
    default_grounding: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    for item in _json_list(value)[:CHART_MAX_TRENDS]:
        if not isinstance(item, dict):
            continue
        series = str(item.get("series") or "").strip()
        trend = str(item.get("trend") or "").strip()
        basis = str(item.get("evidence") or "").strip()
        if not series or not trend or not basis or not _numbers_supported(item, allowed_numbers):
            continue
        visual_evidence = _safe_numeric_items(item.get("visual_evidence"), allowed_numbers)
        ocr_evidence = _source_quotes(item.get("ocr_evidence"), ocr_text)
        if not visual_evidence and not ocr_evidence:
            visual_evidence = list(default_grounding.get("visual_evidence") or [])
            ocr_evidence = list(default_grounding.get("ocr_evidence") or [])
        if not visual_evidence and not ocr_evidence:
            continue
        result.append({
            "series": series,
            "trend": trend,
            "evidence": basis,
            "grounding": {
                "visual_evidence": visual_evidence,
                "ocr_evidence": ocr_evidence,
            },
        })
    return result


def _chart_summary_evidence(
    semantic: dict[str, Any],
    structured: dict[str, Any],
) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    chart_grounding = semantic.get("chart_grounding") if isinstance(semantic.get("chart_grounding"), dict) else {}
    visual_evidence = list(chart_grounding.get("visual_evidence") or [])
    ocr_evidence = list(chart_grounding.get("ocr_evidence") or [])
    description = _chart_description_claim(semantic)
    if description and (visual_evidence or ocr_evidence):
        claims.append({
            "claim_id": "claim_0",
            "kind": "chart_description",
            "statement": description,
            "basis": "Validated chart type, axes, legends, and series.",
            "visual_evidence": visual_evidence,
            "ocr_evidence": ocr_evidence,
        })
    for trend in semantic.get("qualitative_trends", [])[:CHART_MAX_TRENDS]:
        grounding = trend.get("grounding") if isinstance(trend.get("grounding"), dict) else {}
        statement = " ".join(
            part for part in (str(trend.get("series") or "").strip(), str(trend.get("trend") or "").strip()) if part
        )
        if not statement:
            continue
        claims.append({
            "claim_id": f"claim_{len(claims)}",
            "kind": "qualitative_trend",
            "statement": statement,
            "basis": str(trend.get("evidence") or ""),
            "visual_evidence": list(grounding.get("visual_evidence") or []),
            "ocr_evidence": list(grounding.get("ocr_evidence") or []),
        })
    return {
        "program_ocr": {
            "title": structured.get("title") or "",
            "x_axis_label": (structured.get("x_axis") or {}).get("label") or "",
            "y_axis_label": (structured.get("y_axis") or {}).get("label") or "",
        },
        "validated_claims": claims[: CHART_MAX_TRENDS + 1],
    }


def _chart_description_claim(semantic: dict[str, Any]) -> str:
    parts = []
    chart_type = str(semantic.get("chart_type") or "").strip()
    if chart_type and chart_type.casefold() != "unknown":
        parts.append(f"Chart type: {chart_type}")
    for key, label in (("x_axis", "x-axis"), ("y_axis", "y-axis")):
        axis = semantic.get(key) if isinstance(semantic.get(key), dict) else {}
        value = str(axis.get("meaning") or axis.get("label") or "").strip()
        if value and value.casefold() != "unknown":
            parts.append(f"{label}: {value}")
    series = [text for text in (_chart_item_text(item) for item in semantic.get("series", [])) if text.casefold() != "unknown"]
    if series:
        parts.append(f"series: {', '.join(series[:8])}")
    return "; ".join(parts)


def _chart_item_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("name", "text", "label", "series"):
            candidate = value.get(key)
            if isinstance(candidate, list):
                return ", ".join(str(item) for item in candidate)
            if candidate:
                return str(candidate)
        return ""
    return str(value or "").strip()


def _validated_chart_summary(
    response: dict[str, Any],
    summary_evidence: dict[str, Any],
    allowed_numbers: set[str],
) -> tuple[str, list[dict[str, Any]]]:
    claims = {
        str(item.get("claim_id")): item
        for item in summary_evidence.get("validated_claims", [])
        if item.get("claim_id")
    }
    grounding = []
    for item in _json_list(response.get("summary_sentences")):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        claim_ids = [str(value) for value in _json_list(item.get("supporting_claim_ids"))]
        if not text or not claim_ids or any(claim_id not in claims for claim_id in claim_ids):
            continue
        cited_claims = [claims[claim_id] for claim_id in claim_ids]
        supported_numbers = {
            token
            for claim in cited_claims
            for token in NUMBER_RE.findall(json.dumps(claim, ensure_ascii=False))
        }
        required_numbers = set(NUMBER_RE.findall(text))
        if not required_numbers.issubset(allowed_numbers) or not required_numbers.issubset(supported_numbers):
            continue
        grounding.append({
            "claim": text,
            "supporting_claim_ids": claim_ids,
            "supporting_claims": [claim.get("statement") for claim in cited_claims],
            "visual_evidence": _dedupe_json_items(
                item for claim in cited_claims for item in claim.get("visual_evidence", [])
            ),
            "ocr_evidence": list(dict.fromkeys(
                str(item) for claim in cited_claims for item in claim.get("ocr_evidence", [])
            )),
        })
    if not grounding:
        legacy = _safe_numeric_text(response.get("grounded_summary"), allowed_numbers)
        if legacy and claims:
            cited_claims = list(claims.values())
            supported_numbers = set(NUMBER_RE.findall(json.dumps(cited_claims, ensure_ascii=False)))
            if set(NUMBER_RE.findall(legacy)).issubset(supported_numbers):
                grounding.append({
                    "claim": legacy,
                    "supporting_claim_ids": list(claims),
                    "supporting_claims": [claim.get("statement") for claim in cited_claims],
                    "visual_evidence": _dedupe_json_items(
                        item for claim in cited_claims for item in claim.get("visual_evidence", [])
                    ),
                    "ocr_evidence": list(dict.fromkeys(
                        str(item) for claim in cited_claims for item in claim.get("ocr_evidence", [])
                    )),
                })
    return " ".join(item["claim"] for item in grounding), grounding


def _fallback_chart_summary(summary_evidence: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    grounding = []
    for claim in summary_evidence.get("validated_claims", [])[:2]:
        statement = str(claim.get("statement") or "").strip()
        if not statement:
            continue
        grounding.append({
            "claim": statement,
            "supporting_claim_ids": [str(claim.get("claim_id"))],
            "supporting_claims": [statement],
            "visual_evidence": list(claim.get("visual_evidence") or []),
            "ocr_evidence": list(claim.get("ocr_evidence") or []),
        })
    return " ".join(item["claim"] for item in grounding), grounding


def _chart_summary_numeric_provenance_complete(summary: str, grounding: list[dict[str, Any]]) -> bool:
    supported = {
        token
        for item in grounding
        for token in NUMBER_RE.findall(json.dumps({
            "supporting_claims": item.get("supporting_claims") or [],
            "visual_evidence": item.get("visual_evidence") or [],
            "ocr_evidence": item.get("ocr_evidence") or [],
        }, ensure_ascii=False))
    }
    return set(NUMBER_RE.findall(str(summary or ""))).issubset(supported)


def _dedupe_json_items(values: Any) -> list[Any]:
    seen = set()
    result = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


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
    structural_ambiguity = _detect_table_structural_ambiguity(parsed, lookup)
    return {
        **parsed,
        "header_hierarchy": [column_keys] if any(column_keys) else [],
        "row_keys": [value for value in row_keys if value],
        "column_keys": [value for value in column_keys if value],
        "units": units,
        "numeric_cells": numeric_cells,
        "structural_ambiguity": structural_ambiguity,
    }


def _detect_table_structural_ambiguity(
    parsed: dict[str, Any],
    lookup: dict[tuple[int, int], str],
) -> list[str]:
    reasons = []
    if parsed.get("has_rowspan"):
        reasons.append("rowspan_present")
    if parsed.get("has_colspan"):
        reasons.append("colspan_present")
    row_widths = [int(value) for value in parsed.get("row_widths", []) if int(value) >= 0]
    if row_widths and len(set(row_widths)) > 1:
        reasons.append("inconsistent_row_widths_or_missing_columns")
    n_rows, n_cols = int(parsed.get("n_rows") or 0), int(parsed.get("n_cols") or 0)
    header = [str(lookup.get((0, col), "")).strip() for col in range(n_cols)] if n_rows else []
    normalized_header = [normalize for value in header if (normalize := _normalize_table_label(value))]
    if len(normalized_header) != len(set(normalized_header)):
        reasons.append("duplicate_header_labels")
    for col in range(1, n_cols):
        if not header[col] and any(str(lookup.get((row, col), "")).strip() for row in range(1, n_rows)):
            reasons.append("missing_column_header")
            break
    header_signature = tuple(_normalize_table_label(value) for value in header)
    if any(
        tuple(_normalize_table_label(lookup.get((row, col), "")) for col in range(n_cols)) == header_signature
        for row in range(1, n_rows)
    ):
        reasons.append("repeated_header_row")
    return list(dict.fromkeys(reasons))


def _normalize_table_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _compact_table_evidence(structured: dict[str, Any], caption: str) -> dict[str, Any]:
    """Select a bounded, deterministic set of rows without duplicating the raw table."""
    rows: dict[int, list[dict[str, Any]]] = {}
    for cell in structured.get("cells", []):
        try:
            row, col = int(cell["row"]), int(cell["col"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(cell.get("text") or "").strip()
        if text:
            rows.setdefault(row, []).append({"row": row, "col": col, "value": text})
    for cells in rows.values():
        cells.sort(key=lambda item: item["col"])

    header_rows = [0] if 0 in rows else []
    data_rows = [row for row in sorted(rows) if row not in header_rows]
    reasons: dict[int, set[str]] = {row: set() for row in data_rows}
    if len(data_rows) <= TABLE_MAX_CANDIDATE_ROWS:
        for row in data_rows:
            reasons[row].add("complete_table")
    else:
        for row in data_rows[:2]:
            reasons[row].add("leading_row")
        for row in data_rows[-2:]:
            reasons[row].add("trailing_row")
        for row in data_rows:
            label = str(rows[row][0].get("value") or "").casefold()
            if re.search(r"\b(overall|total|average|avg|mean|baseline|aggregate|all)\b", label):
                reasons[row].add("aggregate_or_baseline")
        numeric_by_col: dict[int, list[tuple[float, int]]] = {}
        for row in data_rows:
            for cell in rows[row]:
                numeric = _single_numeric_value(cell["value"])
                if numeric is not None:
                    numeric_by_col.setdefault(cell["col"], []).append((numeric, row))
        for values in numeric_by_col.values():
            if len(values) < 2:
                continue
            minimum = min(value for value, _ in values)
            maximum = max(value for value, _ in values)
            for value, row in values:
                if value == minimum:
                    reasons[row].add("column_minimum")
                if value == maximum:
                    reasons[row].add("column_maximum")

    ranked_rows = sorted(
        data_rows,
        key=lambda row: (-len(reasons[row]), row),
    )[:TABLE_MAX_CANDIDATE_ROWS]
    selected_rows = ranked_rows
    return {
        "title": str(caption or "").strip(),
        "shape": {"rows": int(structured.get("n_rows") or 0), "columns": int(structured.get("n_cols") or 0)},
        "structural_ambiguity": list(structured.get("structural_ambiguity") or []),
        "comparisons_allowed": not bool(structured.get("structural_ambiguity")),
        "headers": [
            {"row": row, "cells": rows[row]}
            for row in header_rows
        ],
        "candidate_rows": [
            {
                "row": row,
                "selection_reasons": sorted(reasons[row]) or ["representative_row"],
                "cells": rows[row],
            }
            for row in selected_rows
        ],
        "units": list(structured.get("units") or []),
    }


def _single_numeric_value(text: str) -> float | None:
    tokens = NUMBER_RE.findall(str(text or ""))
    if len(tokens) != 1:
        return None
    normalized = tokens[0].rstrip("%").replace(",", "")
    try:
        return float(normalized)
    except ValueError:
        return None


def _comparable_table_value(text: str) -> tuple[float, str] | None:
    match = TABLE_COMPARABLE_NUMBER_RE.fullmatch(str(text or ""))
    if not match:
        return None
    currency, number, suffix = match.groups()
    if currency and suffix == "%":
        return None
    try:
        value = float(number.replace(",", ""))
    except ValueError:
        return None
    multiplier = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(str(suffix or "").casefold(), 1.0)
    unit = f"currency:{currency}" if currency else ("percent" if suffix == "%" else "number")
    return value * multiplier, unit


def _validated_table_comparison(
    item: dict[str, Any],
    lookup: dict[tuple[int, int], str],
) -> dict[str, Any] | None:
    left_coords = _valid_source_cells([item.get("left_cell")], lookup)
    right_coords = _valid_source_cells([item.get("right_cell")], lookup)
    if len(left_coords) != 1 or len(right_coords) != 1:
        return None
    left, right = left_coords[0], right_coords[0]
    if left == right or left[0] <= 0 or right[0] <= 0 or left[1] <= 0 or right[1] <= 0:
        return None
    operator = str(item.get("operator") or "").strip()
    if operator not in TABLE_COMPARISON_OPERATORS:
        return None
    left_parsed = _comparable_table_value(lookup[left])
    right_parsed = _comparable_table_value(lookup[right])
    if not left_parsed or not right_parsed or left_parsed[1] != right_parsed[1]:
        return None
    if not _table_relation_holds(left_parsed[0], operator, right_parsed[0]):
        return None
    left_row_label = str(lookup.get((left[0], 0), "")).strip()
    right_row_label = str(lookup.get((right[0], 0), "")).strip()
    left_header = str(lookup.get((0, left[1]), "")).strip()
    right_header = str(lookup.get((0, right[1]), "")).strip()
    if not all((left_row_label, right_row_label, left_header, right_header)):
        return None
    statement = (
        f"{left_row_label} {left_header} ({lookup[left]}) {operator} "
        f"{right_row_label} {right_header} ({lookup[right]})."
    )
    source_cells = []
    for coord in (left, right, (left[0], 0), (right[0], 0), (0, left[1]), (0, right[1])):
        if coord in lookup and coord not in source_cells:
            source_cells.append(coord)
    return {
        "left_cell": [left[0], left[1]],
        "operator": operator,
        "right_cell": [right[0], right[1]],
        "left_value": lookup[left],
        "right_value": lookup[right],
        "statement": statement,
        "source_cells": [[row, col] for row, col in source_cells],
    }


def _table_relation_holds(left: float, operator: str, right: float) -> bool:
    if operator == ">":
        return left > right
    if operator == "<":
        return left < right
    if operator == ">=":
        return left >= right
    if operator == "<=":
        return left <= right
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    return False


def _table_summary_evidence(
    compact: dict[str, Any],
    semantic: dict[str, Any],
    structured: dict[str, Any],
) -> dict[str, Any]:
    lookup = _table_cell_lookup(structured)
    facts: list[dict[str, Any]] = []
    for comparison in semantic.get("comparisons", [])[:TABLE_MAX_COMPARISONS]:
        coords = _valid_source_cells(comparison.get("source_cells"), lookup)
        if not coords:
            continue
        facts.append({
            "kind": "validated_comparison",
            "statement": str(comparison.get("statement") or ""),
            "left_cell": comparison.get("left_cell"),
            "operator": comparison.get("operator"),
            "right_cell": comparison.get("right_cell"),
            "source_cells": [[row, col] for row, col in coords],
            "source_values": [lookup[coord] for coord in coords],
        })
    candidate_facts = _candidate_row_facts(compact)
    if not facts:
        facts.extend(candidate_facts[:TABLE_MAX_SUMMARY_FACTS])
    else:
        facts.extend(candidate_facts[: max(0, TABLE_MAX_SUMMARY_FACTS - len(facts))])
    for important in semantic.get("important_cells", []):
        if len(facts) >= TABLE_MAX_SUMMARY_FACTS:
            break
        coord = (int(important["row"]), int(important["col"]))
        if coord not in lookup:
            continue
        reason = str(important.get("reason") or "").strip()
        value = lookup[coord]
        statement = f"{reason}: {value}" if reason else value
        facts.append({
            "kind": "validated_important_cell",
            "statement": statement,
            "source_cells": [[coord[0], coord[1]]],
            "source_values": [value],
        })
    facts = facts[:TABLE_MAX_SUMMARY_FACTS]
    for index, fact in enumerate(facts):
        fact["fact_id"] = f"fact_{index}"
        fact["descriptive_only"] = fact.get("kind") != "validated_comparison"
    return {
        "title": semantic.get("title_and_purpose") or compact.get("title") or "",
        "headers": compact.get("headers") or [],
        "structural_ambiguity": compact.get("structural_ambiguity") or [],
        "grounded_facts": facts,
    }


def _candidate_row_facts(compact: dict[str, Any]) -> list[dict[str, Any]]:
    headers = {
        int(cell["col"]): str(cell.get("value") or "")
        for header in compact.get("headers", [])
        for cell in header.get("cells", [])
    }
    facts = []
    for row in compact.get("candidate_rows", []):
        cells = list(row.get("cells") or [])
        if not cells:
            continue
        label = str(cells[0].get("value") or "").strip()
        parts = []
        for cell in cells[1:]:
            value = str(cell.get("value") or "").strip()
            header = headers.get(int(cell["col"]), f"column {cell['col']}")
            if value:
                parts.append(f"{header} = {value}")
        statement = f"{label}: {'; '.join(parts)}" if label and parts else "; ".join(
            str(cell.get("value") or "") for cell in cells
        )
        facts.append({
            "kind": "programmatic_candidate_row",
            "statement": statement,
            "source_cells": [[int(cell["row"]), int(cell["col"])] for cell in cells],
            "source_values": [str(cell.get("value") or "") for cell in cells],
        })
    return facts


def _validated_table_summary(
    response: dict[str, Any],
    structured: dict[str, Any],
    summary_evidence: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    lookup = _table_cell_lookup(structured)
    allowed_cells = {
        (int(row), int(col))
        for fact in summary_evidence.get("grounded_facts", [])
        for row, col in fact.get("source_cells", [])
    }
    allowed_cells.update(
        (int(cell["row"]), int(cell["col"]))
        for header in summary_evidence.get("headers", [])
        for cell in header.get("cells", [])
    )
    facts_by_id = {
        str(fact.get("fact_id")): fact
        for fact in summary_evidence.get("grounded_facts", [])
        if fact.get("fact_id")
    }
    grounding = []
    for item in _json_list(response.get("summary_sentences")):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or item.get("statement") or "").strip()
        coords = _valid_source_cells(item.get("source_cells"), lookup)
        fact_ids = [str(value) for value in _json_list(item.get("supporting_fact_ids"))]
        if (
            not text or not coords or not fact_ids
            or any(coord not in allowed_cells for coord in coords)
            or any(fact_id not in facts_by_id for fact_id in fact_ids)
        ):
            continue
        cited_facts = [facts_by_id[fact_id] for fact_id in fact_ids]
        required_cells = {
            (int(row), int(col))
            for fact in cited_facts
            for row, col in fact.get("source_cells", [])
        }
        if not required_cells.issubset(set(coords)):
            continue
        comparison_facts = [fact for fact in cited_facts if fact.get("kind") == "validated_comparison"]
        if comparison_facts:
            if len(cited_facts) != 1 or text != str(comparison_facts[0].get("statement") or ""):
                continue
        elif TABLE_COMPARATIVE_LANGUAGE_RE.search(text):
            continue
        cited_numbers = {token for coord in coords for token in NUMBER_RE.findall(lookup[coord])}
        if not set(NUMBER_RE.findall(text)).issubset(cited_numbers):
            continue
        grounding.append({
            "claim": text,
            "supporting_fact_ids": fact_ids,
            "source_cells": [[row, col] for row, col in coords],
            "source_values": [lookup[coord] for coord in coords],
        })
    return " ".join(item["claim"] for item in grounding), grounding


def _fallback_table_summary(
    summary_evidence: dict[str, Any],
    structured: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    lookup = _table_cell_lookup(structured)
    facts = list(summary_evidence.get("grounded_facts") or [])
    comparisons = [fact for fact in facts if fact.get("kind") == "validated_comparison"]
    selected = (comparisons or facts)[:2]
    grounding = []
    for fact in selected:
        coords = _valid_source_cells(fact.get("source_cells"), lookup)
        statement = str(fact.get("statement") or "").strip()
        if not statement or not coords:
            continue
        grounding.append({
            "claim": statement,
            "supporting_fact_ids": [str(fact.get("fact_id"))],
            "source_cells": [[row, col] for row, col in coords],
            "source_values": [lookup[coord] for coord in coords],
        })
    return " ".join(item["claim"] for item in grounding), grounding


def _ambiguous_table_description(
    compact: dict[str, Any],
    structured: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    caption = str(compact.get("title") or "").strip()
    title = re.sub(r"^\s*table\s+\d+[a-z]?\s*[:.\-]?\s*", "", caption, flags=re.I).strip()
    if title:
        claim = f"Table title: {title}"
        return claim, [{
            "claim": claim,
            "source_kind": "caption",
            "source_text": caption,
            "source_cells": [],
            "source_values": [],
            "descriptive_only": True,
            "structural_ambiguity": list(structured.get("structural_ambiguity") or []),
        }]
    lookup = _table_cell_lookup(structured)
    header_cells = [
        (coord, value) for coord, value in sorted(lookup.items()) if coord[0] == 0 and value.strip()
    ]
    if not header_cells:
        return "", []
    claim = "Table columns: " + ", ".join(value for _, value in header_cells)
    return claim, [{
        "claim": claim,
        "source_kind": "table_headers",
        "source_cells": [[row, col] for (row, col), _ in header_cells],
        "source_values": [value for _, value in header_cells],
        "descriptive_only": True,
        "structural_ambiguity": list(structured.get("structural_ambiguity") or []),
    }]


def _table_cell_lookup(structured: dict[str, Any]) -> dict[tuple[int, int], str]:
    return {
        (int(cell["row"]), int(cell["col"])): str(cell.get("text") or "")
        for cell in structured.get("cells", [])
    }


def _validated_table_semantics(response: dict[str, Any], structured: dict[str, Any], caption: str) -> dict[str, Any]:
    lookup = _table_cell_lookup(structured)
    all_numbers = {token for value in lookup.values() for token in NUMBER_RE.findall(value)}
    important = []
    for item in _json_list(response.get("important_cells"))[:8]:
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
    if not structured.get("structural_ambiguity"):
        for item in _json_list(response.get("comparisons"))[:TABLE_MAX_COMPARISONS]:
            if not isinstance(item, dict):
                continue
            comparison = _validated_table_comparison(item, lookup)
            if comparison:
                comparisons.append(comparison)
    ambiguity = list(structured.get("structural_ambiguity") or [])
    ambiguity.extend(str(item) for item in _safe_numeric_items(response.get("ambiguous_structure"), all_numbers))
    if not structured.get("table_parse_available"):
        ambiguity.append("HTML/Markdown table source could not be parsed")
    title_and_purpose = _safe_numeric_text(response.get("title_and_purpose"), all_numbers)
    if not title_and_purpose:
        title_and_purpose = _safe_numeric_text(re.sub(r"^\s*table\s+\d+[a-z]?\s*[:.\-]?\s*", "", caption, flags=re.I), all_numbers)
    table_structure_response = response.get("table_structure") if isinstance(response.get("table_structure"), dict) else {}
    table_structure = {
        "header_meaning": _validated_cell_grounded_items(table_structure_response.get("header_meaning"), lookup),
        "column_semantics": _validated_cell_grounded_items(table_structure_response.get("column_semantics"), lookup),
        "row_semantics": _validated_cell_grounded_items(table_structure_response.get("row_semantics"), lookup),
    }
    cell_grounding = _validated_cell_grounded_items(response.get("cell_grounding"), lookup)
    for item in important:
        cell_grounding.append({
            "claim": item.get("reason") or item["value"],
            "source_cells": [[item["row"], item["col"]]],
            "source_values": [item["value"]],
        })
    for item in comparisons:
        coords = [(int(row), int(col)) for row, col in item["source_cells"]]
        cell_grounding.append({
            "claim": item["statement"],
            "source_cells": item["source_cells"],
            "source_values": [lookup[coord] for coord in coords],
        })
    cell_grounding = _dedupe_grounding(cell_grounding)
    if title_and_purpose and not _numeric_text_traceable(title_and_purpose, cell_grounding):
        title_and_purpose = ""
    grounded_summary = _safe_numeric_text(response.get("grounded_summary"), all_numbers)
    if grounded_summary and not _numeric_text_traceable(grounded_summary, cell_grounding):
        grounded_summary = ""
    semantic_confidence = _confidence(response.get("semantic_confidence", response.get("confidence", structured.get("parse_confidence", 0.0))))
    return {
        "title_and_purpose": title_and_purpose,
        "header_hierarchy": _safe_numeric_items(response.get("header_hierarchy"), all_numbers),
        "row_keys": _safe_numeric_items(response.get("row_keys"), all_numbers),
        "column_keys": _safe_numeric_items(response.get("column_keys"), all_numbers),
        "units": _safe_numeric_items(response.get("units"), all_numbers),
        "important_cells": important,
        "comparisons": comparisons,
        "grounded_summary": grounded_summary,
        "ambiguous_structure": [
            item for item in dict.fromkeys(ambiguity) if _numeric_text_traceable(item, cell_grounding)
        ],
        "confidence": _confidence(response.get("confidence", structured.get("parse_confidence", 0.0))),
        "table_structure": table_structure,
        "cell_grounding": cell_grounding,
        "semantic_confidence": semantic_confidence,
    }


def _validated_cell_grounded_items(value: Any, lookup: dict[tuple[int, int], str]) -> list[dict[str, Any]]:
    result = []
    for item in _json_list(value):
        if not isinstance(item, dict):
            continue
        coords = _valid_source_cells(item.get("source_cells"), lookup)
        if not coords:
            continue
        source_values = [lookup[coord] for coord in coords]
        allowed_numbers = {token for source in source_values for token in NUMBER_RE.findall(source)}
        claim = str(item.get("claim") or item.get("meaning") or item.get("semantic") or "")
        if not claim or not set(NUMBER_RE.findall(claim)).issubset(allowed_numbers):
            continue
        grounded = {
            key: value
            for key, value in item.items()
            if key not in {"source_cells", "source_values"} and _numbers_supported(value, allowed_numbers)
        }
        grounded["source_cells"] = [[row, col] for row, col in coords]
        grounded["source_values"] = source_values
        result.append(grounded)
    return result


def _dedupe_grounding(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = (str(item.get("claim") or item.get("meaning") or ""), json.dumps(item.get("source_cells"), sort_keys=True))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _numeric_text_traceable(text: str, grounding: list[dict[str, Any]]) -> bool:
    required = set(NUMBER_RE.findall(text))
    supported = {
        token
        for item in grounding
        for source in item.get("source_values", [])
        for token in NUMBER_RE.findall(str(source))
    }
    return required.issubset(supported)


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


def _table_numeric_provenance_complete(semantic: dict[str, Any], structured: dict[str, Any]) -> bool:
    lookup = {(int(cell["row"]), int(cell["col"])): str(cell.get("text") or "") for cell in structured.get("cells", [])}
    for item in semantic.get("important_cells", []):
        if lookup.get((item["row"], item["col"])) != item["value"]:
            return False
    for item in semantic.get("comparisons", []):
        revalidated = _validated_table_comparison(item, lookup)
        if not revalidated or revalidated.get("statement") != item.get("statement"):
            return False
    for item in semantic.get("cell_grounding", []):
        coords = _valid_source_cells(item.get("source_cells"), lookup)
        if not coords:
            return False
        cited = {token for coord in coords for token in NUMBER_RE.findall(lookup[coord])}
        claim = str(item.get("claim") or item.get("meaning") or "")
        if not set(NUMBER_RE.findall(claim)).issubset(cited):
            return False
    summary_grounding = semantic.get("summary_grounding", [])
    for item in summary_grounding:
        if item.get("source_kind") == "caption":
            cited = set(NUMBER_RE.findall(str(item.get("source_text") or "")))
            if not set(NUMBER_RE.findall(str(item.get("claim") or ""))).issubset(cited):
                return False
            continue
        coords = _valid_source_cells(item.get("source_cells"), lookup)
        if not coords:
            return False
        cited = {token for coord in coords for token in NUMBER_RE.findall(lookup[coord])}
        if not set(NUMBER_RE.findall(str(item.get("claim") or ""))).issubset(cited):
            return False
    summary_numbers = set(NUMBER_RE.findall(str(semantic.get("grounded_summary") or "")))
    supported_summary_numbers = {
        token
        for item in summary_grounding
        for source in (
            [item.get("source_text")] if item.get("source_kind") == "caption"
            else item.get("source_values", [])
        )
        for token in NUMBER_RE.findall(str(source))
    }
    if not summary_numbers.issubset(supported_summary_numbers):
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


def _validated_axis(value: Any, allowed_numbers: set[str]) -> dict[str, Any]:
    if isinstance(value, str):
        value = {"label": value, "meaning": value}
    if not isinstance(value, dict) or not _numbers_supported(value, allowed_numbers):
        return {"label": "unknown", "meaning": "unknown", "unit": "", "confidence": 0.0}
    result = dict(value)
    result["label"] = str(result.get("label") or "unknown")
    result["meaning"] = str(result.get("meaning") or result["label"] or "unknown")
    result["unit"] = str(result.get("unit") or "")
    result["confidence"] = _confidence(result.get("confidence"))
    return result


def _source_quotes(value: Any, source: str, fallback_to_source: bool = False) -> list[str]:
    source = str(source or "").strip()
    if not source:
        return []
    normalized_source = " ".join(source.lower().split())
    result = []
    for item in _string_list(value):
        normalized_item = " ".join(item.lower().split())
        if normalized_item and normalized_item in normalized_source:
            result.append(item)
    if not result and fallback_to_source:
        result.append(source)
    return result


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_flatten_text(item) for item in value)
    return str(value or "")


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
    response, _ = _call_json_model_diagnostic(func, prompt, image_path)
    return response


def _call_json_model_diagnostic(
    func: Callable | None,
    prompt: str,
    image_path: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    if not func:
        return {}, {"status": "model_unavailable", "error": "", "response_excerpt": ""}
    kwargs = {"prompt": prompt, "query": prompt, "temperature": 0.0}
    if image_path:
        kwargs.update({"image_paths": [image_path], "image_path": image_path})
    try:
        raw_response = func(**kwargs)
    except TypeError:
        try:
            raw_response = func(prompt, [image_path]) if image_path else func(prompt)
        except Exception as exc:
            return {}, {
                "status": "model_exception",
                "error": _diagnostic_text(exc),
                "response_excerpt": "",
            }
    except Exception as exc:
        return {}, {
            "status": "model_exception",
            "error": _diagnostic_text(exc),
            "response_excerpt": "",
        }
    parsed = _parse_json_object(raw_response)
    if parsed:
        return parsed, {"status": "ok", "error": "", "response_excerpt": ""}
    raw_text = str(raw_response or "").strip()
    return {}, {
        "status": "empty_response" if not raw_text else "invalid_json",
        "error": "",
        "response_excerpt": raw_text[:1000],
    }


def _diagnostic_text(value: Exception) -> str:
    return f"{type(value).__name__}: {value}"[:1000]


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
