from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .id_utils import normalize_text, stable_id
from .input_adapter import JoinedMedia
from .schema import (
    GENERATOR_VERSION,
    MEDIA_SEMANTIC_UNIT_SCHEMA_VERSION,
    GenerationInfo,
    Grounding,
    GroundingKind,
    MediaSemanticUnit,
    RefType,
    SourceReference,
    bounded_number,
)


NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?%?")
RANK_OR_TREND_RE = re.compile(
    r"\b(?:highest|lowest|top|bottom|rank(?:s|ed|ing)?|increase[sd]?|decrease[sd]?|"
    r"rise[sd]?|fall[se]?|trend|outperform(?:s|ed)?|better|worse|greater|less)\b",
    re.I,
)
LAYOUT_OBJECTS = {
    "axis", "axes", "line", "lines", "legend", "legends", "rectangle", "rectangles",
    "arrow", "arrows", "circle", "circles", "box", "boxes", "shape", "shapes",
}
IDENTITY_COLUMN_RE = re.compile(
    r"\b(?:model|method|dataset|system|approach|component|category|task|metric|benchmark|name)\b",
    re.I,
)
TABLE_FACT_LIMIT = 16


@dataclass
class _BuildContent:
    retrieval_parts: list[str]
    graph_facts: list[str]
    evidence_refs: list[SourceReference]
    warnings: list[str]


class SemanticUnitBuildError(ValueError):
    pass


class MediaSemanticUnitBuilder:
    """Build deterministic Phase 3 text only from validated Phase 2 evidence."""

    def __init__(self, generator_version: str = GENERATOR_VERSION) -> None:
        self.generator_version = generator_version
        self._strategies: dict[str, Callable[[JoinedMedia], _BuildContent]] = {
            "image": self._build_image,
            "chart": self._build_chart,
            "table": self._build_table,
        }

    def build(self, item: JoinedMedia) -> MediaSemanticUnit:
        try:
            strategy = self._strategies[item.media_type]
        except KeyError as exc:
            raise SemanticUnitBuildError(f"Unsupported semantic unit media_type: {item.media_type}") from exc
        content = strategy(item)
        content.retrieval_parts = _dedupe_text(content.retrieval_parts)
        content.graph_facts = _dedupe_text(content.graph_facts)
        content.evidence_refs = _dedupe_refs(content.evidence_refs)
        retrieval_text = "\n".join(content.retrieval_parts).strip()
        graph_text = "\n".join(content.graph_facts).strip()
        if not retrieval_text or not content.evidence_refs:
            raise SemanticUnitBuildError(f"Insufficient grounded evidence for media {item.media_id}")
        confidence = bounded_number(
            item.processed.get("semantic_confidence", (item.processed.get("confidence") or {}).get("overall", 0.0)),
            "processed_media.semantic_confidence",
        )
        return MediaSemanticUnit(
            chunk_id=stable_id(
                "media_chunk",
                "media_semantic_unit.v1",
                {"schema_version": MEDIA_SEMANTIC_UNIT_SCHEMA_VERSION, "media_id": item.media_id},
            ),
            media_id=item.media_id,
            retrieval_text=retrieval_text,
            graph_text=graph_text,
            evidence_refs=content.evidence_refs,
            generation=GenerationInfo(
                schema_version=MEDIA_SEMANTIC_UNIT_SCHEMA_VERSION,
                generator_version=self.generator_version,
                confidence=confidence,
                warnings=sorted(set(content.warnings)),
            ),
        )

    def _build_image(self, item: JoinedMedia) -> _BuildContent:
        processed = item.processed
        semantic = _dict(processed.get("semantic_content"))
        context = _dict(processed.get("media_context"))
        direct = _dict(context.get("direct_evidence"))
        layout = _dict(context.get("layout_context"))
        retrieval: list[str] = ["Modality: image"]
        graph: list[str] = []
        refs: list[SourceReference] = []
        warnings: list[str] = []

        caption = _clean_excerpt(direct.get("caption"), 500)
        if caption:
            retrieval.append(f"Caption: {caption}")
            refs.append(_media_ref(item, GroundingKind.VISUAL_FACT.value, "/media_context/direct_evidence/caption", "caption"))
        section = _clean_excerpt(layout.get("section"), 200)
        if section:
            retrieval.append(f"Document context: {section}")
        reference = _clean_excerpt(context.get("reference_context"), 500)
        if reference:
            retrieval.append(f"Reference context: {reference}")
            refs.append(_media_ref(item, GroundingKind.TEXT_SPAN.value, "/media_context/reference_context", "reference_context"))

        for key, label, kind in (
            ("visual_facts", "Visual fact", GroundingKind.VISUAL_FACT.value),
            ("visible_text", "Visible text", GroundingKind.OCR_SPAN.value),
        ):
            for index, raw in enumerate(_list(semantic.get(key))[:12]):
                text = _clean_excerpt(raw, 350)
                if not text:
                    continue
                retrieval.append(f"{label}: {text}")
                refs.append(_media_ref(item, kind, f"/semantic_content/{key}/{index}", key))
                if not _layout_only(text):
                    graph.append(text)

        summary = _clean_excerpt(semantic.get("grounded_summary"), 500)
        visual_sources = _list(_dict(semantic.get("evidence_source")).get("visual"))
        if summary and visual_sources:
            retrieval.append(f"Grounded summary: {summary}")
            graph.append(summary)
            refs.append(_media_ref(item, GroundingKind.VISUAL_FACT.value, "/semantic_content/grounded_summary", "grounded_summary"))
        elif summary:
            warnings.append("image_grounded_summary_rejected_without_visual_source")
        if not graph:
            warnings.append("no_grounded_graph_facts")
        return _BuildContent(retrieval, graph, refs, warnings)

    def _build_table(self, item: JoinedMedia) -> _BuildContent:
        processed = item.processed
        semantic = _dict(processed.get("semantic_content"))
        structured = _dict(processed.get("structured_content"))
        context = _dict(processed.get("media_context"))
        direct = _dict(context.get("direct_evidence"))
        retrieval: list[str] = ["Modality: table"]
        graph: list[str] = []
        refs: list[SourceReference] = []
        warnings: list[str] = []
        caption = _clean_excerpt(direct.get("caption"), 500)
        title = _clean_excerpt(semantic.get("title_and_purpose"), 500)
        if caption:
            retrieval.append(f"Caption: {caption}")
            refs.append(_media_ref(item, GroundingKind.VISUAL_FACT.value, "/media_context/direct_evidence/caption", "caption"))
        if title:
            retrieval.append(f"Purpose: {title}")
            refs.append(_media_ref(item, GroundingKind.VISUAL_FACT.value, "/semantic_content/title_and_purpose", "table_title"))

        claims, reconstruction_warnings = _reconstruct_table_facts(structured, semantic)
        if claims:
            warnings.append("table_facts_reconstructed_from_structured_cells")
        warnings.extend(reconstruction_warnings)
        for index, comparison in enumerate(_list(semantic.get("comparisons"))):
            if not isinstance(comparison, dict):
                continue
            statement = _clean_excerpt(comparison.get("statement"), 400)
            cells = _valid_cells(comparison.get("source_cells"))
            if statement and cells and not claims:
                claims.append((statement, cells, f"/semantic_content/comparisons/{index}"))
        for index, grounding in enumerate(_list(semantic.get("cell_grounding"))):
            if not isinstance(grounding, dict):
                continue
            statement = _clean_excerpt(
                grounding.get("claim") or grounding.get("meaning") or grounding.get("semantic"), 400
            )
            cells = _valid_cells(grounding.get("source_cells"))
            if statement and cells and not claims:
                claims.append((statement, cells, f"/semantic_content/cell_grounding/{index}"))
        claims = _dedupe_claims(claims)[:TABLE_FACT_LIMIT]
        for statement, cells, path in claims:
            retrieval.append(f"Grounded table fact: {statement}")
            graph.append(statement)
            refs.append(_media_ref(
                item, GroundingKind.TABLE_CELLS.value, path, "table_fact", {"cells": cells}
            ))

        summary = _clean_excerpt(semantic.get("grounded_summary"), 500)
        if summary:
            if NUMBER_RE.search(summary) and not _numbers_supported_by_table_claims(summary, semantic):
                warnings.append("table_numeric_summary_rejected_without_cell_grounding")
            else:
                retrieval.append(f"Grounded summary: {summary}")
        if not claims:
            warnings.append("no_cell_grounded_graph_facts")
        if not refs:
            raise SemanticUnitBuildError(f"Table {item.media_id} has no locator-backed evidence")
        return _BuildContent(retrieval, graph, refs, warnings)

    def _build_chart(self, item: JoinedMedia) -> _BuildContent:
        processed = item.processed
        semantic = _dict(processed.get("semantic_content"))
        structured = _dict(processed.get("structured_content"))
        context = _dict(processed.get("media_context"))
        direct = _dict(context.get("direct_evidence"))
        chart_grounding = _dict(semantic.get("chart_grounding"))
        has_chart_grounding = any(_list(chart_grounding.get(key)) for key in ("visual_evidence", "ocr_evidence"))
        allowed_numbers = {
            token
            for point in _list(structured.get("readable_data_points"))
            for token in NUMBER_RE.findall(json.dumps(point, ensure_ascii=False))
        }
        allowed_numbers.update(NUMBER_RE.findall(str(structured.get("ocr_text") or "")))

        retrieval: list[str] = ["Modality: chart"]
        graph: list[str] = []
        refs: list[SourceReference] = []
        warnings: list[str] = []
        caption = _clean_excerpt(direct.get("caption"), 500)
        title = _clean_excerpt(semantic.get("title"), 300)
        if caption:
            retrieval.append(f"Caption: {caption}")
            refs.append(_media_ref(item, GroundingKind.VISUAL_FACT.value, "/media_context/direct_evidence/caption", "caption"))
        if title and title.casefold() != "unknown":
            retrieval.append(f"Title: {title}")
            refs.append(_media_ref(item, GroundingKind.CHART_EVIDENCE.value, "/semantic_content/title", "chart_title"))
        series = [normalize_text(value) for value in _list(semantic.get("series")) if normalize_text(value).casefold() != "unknown"]
        if series:
            retrieval.append(f"Series: {', '.join(series[:10])}")
        for index, trend in enumerate(_list(semantic.get("qualitative_trends"))):
            if not isinstance(trend, dict):
                continue
            series_name = normalize_text(trend.get("series"))
            trend_text = normalize_text(trend.get("trend"))
            evidence = normalize_text(trend.get("evidence"))
            statement = normalize_text(f"{series_name} {trend_text}")
            if not statement or not evidence or not has_chart_grounding:
                warnings.append("chart_trend_rejected_without_grounding")
                continue
            if not set(NUMBER_RE.findall(statement + " " + evidence)).issubset(allowed_numbers):
                warnings.append("chart_trend_rejected_with_unsupported_number")
                continue
            retrieval.append(f"Grounded trend: {statement}")
            graph.append(statement)
            refs.append(_media_ref(
                item,
                GroundingKind.CHART_EVIDENCE.value,
                f"/semantic_content/qualitative_trends/{index}",
                "chart_trend",
                {"evidence": evidence},
            ))
        summary = _clean_excerpt(semantic.get("grounded_summary"), 500)
        if summary:
            risky = bool(NUMBER_RE.search(summary) or RANK_OR_TREND_RE.search(summary))
            numbers_supported = set(NUMBER_RE.findall(summary)).issubset(allowed_numbers)
            if not has_chart_grounding or (risky and not numbers_supported):
                warnings.append("chart_summary_rejected_without_sufficient_grounding")
            else:
                retrieval.append(f"Grounded summary: {summary}")
                graph.append(summary)
                refs.append(_media_ref(
                    item, GroundingKind.CHART_EVIDENCE.value, "/semantic_content/grounded_summary", "chart_summary"
                ))
        if not graph:
            warnings.append("no_grounded_graph_facts")
        if not refs:
            raise SemanticUnitBuildError(f"Chart {item.media_id} has no locator-backed evidence")
        return _BuildContent(retrieval, graph, refs, warnings)


def build_media_semantic_units(joined: list[JoinedMedia]) -> tuple[list[MediaSemanticUnit], list[dict[str, Any]]]:
    builder = MediaSemanticUnitBuilder()
    units = []
    errors = []
    for item in sorted(joined, key=lambda value: value.media_id):
        try:
            units.append(builder.build(item))
        except Exception as exc:
            errors.append({
                "stage": "semantic_units", "code": "semantic_unit_failed", "message": str(exc),
                "media_id": item.media_id, "retryable": False, "attempt": None,
            })
    return units, errors


def _media_ref(
    item: JoinedMedia,
    kind: str,
    json_path: str,
    evidence_role: str,
    extra_locator: dict[str, Any] | None = None,
) -> SourceReference:
    locator = {
        "source_file": "processed_media.json",
        "json_path": json_path,
        "evidence_role": evidence_role,
    }
    locator.update(extra_locator or {})
    confidence = item.processed.get("semantic_confidence", (item.processed.get("confidence") or {}).get("overall", 0.0))
    return SourceReference(
        ref_type=RefType.MEDIA.value,
        ref_id=item.media_id,
        media_id=item.media_id,
        grounding=Grounding(kind=kind, locator=locator),
        confidence=bounded_number(confidence, "evidence confidence"),
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_excerpt(value: Any, max_chars: int) -> str:
    return normalize_text(value)[:max_chars]


def _valid_cells(value: Any) -> list[list[int]]:
    result = []
    for coord in value if isinstance(value, list) else []:
        if not isinstance(coord, (list, tuple)) or len(coord) != 2:
            continue
        try:
            normalized = [int(coord[0]), int(coord[1])]
        except (TypeError, ValueError):
            continue
        if normalized not in result:
            result.append(normalized)
    return result


def _reconstruct_table_facts(
    structured: dict[str, Any], semantic: dict[str, Any]
) -> tuple[list[tuple[str, list[list[int]], str]], list[str]]:
    """Build complete, cell-grounded row facts without semantic inference."""
    cells = []
    for raw in _list(structured.get("cells")):
        if not isinstance(raw, dict):
            continue
        try:
            row, col = int(raw["row"]), int(raw["col"])
        except (KeyError, TypeError, ValueError):
            continue
        text = normalize_text(raw.get("text"))
        if row >= 0 and col >= 0 and text:
            cells.append((row, col, text))
    columns = [normalize_text(value) for value in _list(structured.get("column_keys")) if normalize_text(value)]
    if not cells or not columns:
        return [], []

    header_depth = max(1, len(_list(structured.get("header_hierarchy"))))
    header_cells = [cell for cell in cells if cell[0] < header_depth]
    header_coords: dict[str, list[int]] = {}
    for column in columns:
        match = next((cell for cell in header_cells if cell[2].casefold() == column.casefold()), None)
        if match:
            header_coords[column.casefold()] = [match[0], match[1]]

    important_rows = {
        int(item["row"])
        for item in _list(semantic.get("important_cells"))
        if isinstance(item, dict) and isinstance(item.get("row"), int)
    }
    rows: dict[int, list[tuple[int, int, str]]] = {}
    for cell in cells:
        if cell[0] >= header_depth:
            rows.setdefault(cell[0], []).append(cell)

    reconstructed = []
    reconstruction_warnings: set[str] = set()
    identity_column = bool(IDENTITY_COLUMN_RE.search(columns[0]))
    for row, row_cells in sorted(rows.items(), key=lambda pair: pair[0]):
        ordered = sorted(row_cells, key=lambda cell: cell[1])
        if len(ordered) < len(columns):
            if len(ordered) > 1:
                reconstruction_warnings.add("table_rows_skipped_due_to_ambiguous_alignment")
            continue
        coordinates = [cell[1] for cell in ordered]
        if coordinates != list(range(coordinates[0], coordinates[0] + len(coordinates))):
            reconstruction_warnings.add("table_rows_skipped_due_to_ambiguous_alignment")
            continue
        leading = ordered[:-len(columns)]
        data = ordered[-len(columns):]
        if identity_column:
            subject_cell = data[0]
            subject = subject_cell[2]
            metric_pairs = list(zip(columns[1:], data[1:]))
            context_cells = leading
        else:
            if not leading:
                continue
            subject_cell = leading[-1]
            subject = " / ".join(cell[2] for cell in leading)
            metric_pairs = list(zip(columns, data))
            context_cells = leading
        if not subject or _looks_like_missing_value(subject):
            continue

        clauses = []
        grounding_cells = [[cell[0], cell[1]] for cell in context_cells]
        if [subject_cell[0], subject_cell[1]] not in grounding_cells:
            grounding_cells.append([subject_cell[0], subject_cell[1]])
        for column, value_cell in metric_pairs[:8]:
            value = value_cell[2]
            if _looks_like_missing_value(value):
                continue
            if _shot_setting_without_measurement(value, column):
                reconstruction_warnings.add("table_cells_skipped_due_to_incomplete_value")
                continue
            clauses.append(f"{column} = {value}")
            header_coord = header_coords.get(column.casefold())
            if header_coord and header_coord not in grounding_cells:
                grounding_cells.append(header_coord)
            value_coord = [value_cell[0], value_cell[1]]
            if value_coord not in grounding_cells:
                grounding_cells.append(value_coord)
        if clauses:
            reconstructed.append((
                f"{subject}: {'; '.join(clauses)}.",
                grounding_cells,
                "/structured_content/cells",
                row in important_rows,
                row,
            ))

    prioritized = sorted(reconstructed, key=lambda item: (not item[3], item[4]))[:TABLE_FACT_LIMIT]
    prioritized.sort(key=lambda item: item[4])
    return (
        [(statement, coords, path) for statement, coords, path, _, _ in prioritized],
        sorted(reconstruction_warnings),
    )


def _looks_like_missing_value(value: str) -> bool:
    return normalize_text(value).casefold() in {"", "-", "--", "—", "n/a", "na", "none", "unknown"}


def _shot_setting_without_measurement(value: str, column: str) -> bool:
    setting_only = re.fullmatch(r"\d+\s*-\s*shot", normalize_text(value), re.I) is not None
    setting_column = re.search(r"\b(?:shot|shots|setting)\b", normalize_text(column), re.I) is not None
    return setting_only and not setting_column


def _numbers_supported_by_table_claims(text: str, semantic: dict[str, Any]) -> bool:
    supported = set()
    for item in _list(semantic.get("cell_grounding")):
        if isinstance(item, dict) and _valid_cells(item.get("source_cells")):
            for value in _list(item.get("source_values")):
                supported.update(NUMBER_RE.findall(str(value)))
    return set(NUMBER_RE.findall(text)).issubset(supported)


def _layout_only(text: str) -> bool:
    tokens = set(re.findall(r"[A-Za-z]+", text.casefold()))
    ignored = {"a", "an", "the", "is", "are", "visible", "shown", "contains", "and", "of"}
    content = tokens - ignored
    return bool(content) and content.issubset(LAYOUT_OBJECTS)


def _dedupe_text(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        clean = normalize_text(item)
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def _dedupe_refs(items: list[SourceReference]) -> list[SourceReference]:
    seen = set()
    result = []
    for item in items:
        key = json.dumps(
            {"ref_type": item.ref_type, "ref_id": item.ref_id, "grounding": item.grounding.locator},
            sort_keys=True,
            ensure_ascii=False,
        )
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _dedupe_claims(items: list[tuple[str, list[list[int]], str]]) -> list[tuple[str, list[list[int]], str]]:
    seen = set()
    result = []
    for statement, cells, path in items:
        key = (statement.casefold(), tuple(tuple(cell) for cell in cells))
        if key not in seen:
            seen.add(key)
            result.append((statement, cells, path))
    return result
