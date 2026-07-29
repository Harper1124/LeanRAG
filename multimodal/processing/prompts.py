"""Phase 2 prompts: structured understanding without graph or retrieval prose."""

IMAGE_PROMPT = """Analyze the supplied document image and return one JSON object only.

Evidence policy:
- visual_facts, visible_text, objects, spatial_relations, image_type, and grounded_summary must use only pixels visible in the image.
- Caption, OCR, nearby text, section, and references are text evidence. Use them only for caption_consistency; never convert them into visual facts.
- evidence_source.visual must cite visual_facts or visible_text. Caption and nearby_text entries must quote the corresponding context source.
- Do not infer a visually absent object from the caption. Use \"unknown\" for an unclear object.
- uncertain_items must list ambiguity. Do not produce entities, graph knowledge, graph_text, retrieval_text, or a long free-form description.

Return this schema:
{
  "visual_facts": [], "visible_text": [], "objects": [], "spatial_relations": [],
  "image_type": "photo|diagram|screenshot|map|other",
  "caption_consistency": "", "grounded_summary": "", "uncertain_items": [], "confidence": 0.0,
  "evidence_source": {"visual": [], "caption": [], "nearby_text": []},
  "semantic_role": ["example|architecture|experiment|illustration|other"], "semantic_confidence": 0.0
}

Text/layout context (not visual evidence):
{context}
"""

CHART_PROMPT = """Analyze the supplied chart image and return one JSON object only.

Program OCR text, positions, and readable numbers are factual evidence. Program axis and legend assignments are candidates, not semantic truth.
- Use chart pixels with OCR positions to determine real x/y meaning, legend-to-series mapping, and chart meaning.
- Do not copy a candidate role when pixels do not support it. Return "unknown" for an uncertain field.
- Do not invent or interpolate blurred numbers.
- Every qualitative_trend item must identify its series and state its visual/programmatic basis.
- Record caption/image conflicts in caption_consistency.
- Use unreadable_regions for unclear regions.
- Do not produce entities, graph knowledge, graph_text, retrieval_text, or a long free-form description.

Return this schema:
{
  "chart_type": "", "title": "", "x_axis": {}, "y_axis": {}, "legends": [], "series": [],
  "readable_data_points": [], "qualitative_trends": [], "extrema_and_intersections": [],
  "caption_consistency": "", "unreadable_regions": [], "grounded_summary": "", "confidence": 0.0,
  "chart_grounding": {"visual_evidence": [], "ocr_evidence": [], "context_evidence": []},
  "semantic_confidence": 0.0
}

Program-derived chart evidence:
{structured}

Text/layout context (not visual evidence):
{context}
"""

TABLE_PROMPT = """Interpret the parsed table and return one JSON object only.

Evidence policy:
- Parsed HTML/Markdown cells are the sole factual source for table values.
- Every important_cells item must contain row, col, value, and reason; value must exactly match that source cell.
- Every comparisons item must contain statement and source_cells ([[row,col], ...]).
- Every table_structure and cell_grounding item must contain source_cells. Numbers in a claim or meaning must occur in those cells.
- Do not add missing values, unverifiable rankings, extrapolated trends, entities, graph knowledge, graph_text, or retrieval_text.
- Put structural uncertainty in ambiguous_structure.

Return this schema:
{
  "title_and_purpose": "", "header_hierarchy": [], "row_keys": [], "column_keys": [], "units": [],
  "important_cells": [], "comparisons": [], "grounded_summary": "", "ambiguous_structure": [], "confidence": 0.0,
  "table_structure": {"header_meaning": [], "column_semantics": [], "row_semantics": []},
  "cell_grounding": [], "semantic_confidence": 0.0
}

Parsed table source:
{structured}

Text/layout context:
{context}
"""
