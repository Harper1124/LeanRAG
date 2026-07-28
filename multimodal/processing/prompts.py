"""Phase 2 prompts: structured understanding without graph or retrieval prose."""

IMAGE_PROMPT = """Analyze the supplied document image and return one JSON object only.

Evidence policy:
- visual_facts, visible_text, objects, spatial_relations, image_type, and grounded_summary must use only pixels visible in the image.
- Caption, OCR, nearby text, section, and references are text evidence. Use them only for caption_consistency; never convert them into visual facts.
- Do not infer a visually absent object from the caption. Use \"unknown\" for an unclear object.
- uncertain_items must list ambiguity. Do not produce entities, graph knowledge, graph_text, retrieval_text, or a long free-form description.

Return this schema:
{
  "visual_facts": [], "visible_text": [], "objects": [], "spatial_relations": [],
  "image_type": "photo|diagram|screenshot|map|other",
  "caption_consistency": "", "grounded_summary": "", "uncertain_items": [], "confidence": 0.0
}

Text/layout context (not visual evidence):
{context}
"""

CHART_PROMPT = """Analyze the supplied chart image and return one JSON object only.

The program-derived OCR/axis/legend/value evidence below is authoritative for readable text and numbers.
- Do not invent or interpolate blurred numbers.
- Every qualitative_trend item must identify its series and state its visual/programmatic basis.
- Record caption/image conflicts in caption_consistency.
- Use unreadable_regions for unclear regions.
- Do not produce entities, graph knowledge, graph_text, retrieval_text, or a long free-form description.

Return this schema:
{
  "chart_type": "", "title": "", "x_axis": {}, "y_axis": {}, "legends": [], "series": [],
  "readable_data_points": [], "qualitative_trends": [], "extrema_and_intersections": [],
  "caption_consistency": "", "unreadable_regions": [], "grounded_summary": "", "confidence": 0.0
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
- Do not add missing values, unverifiable rankings, extrapolated trends, entities, graph knowledge, graph_text, or retrieval_text.
- Put structural uncertainty in ambiguous_structure.

Return this schema:
{
  "title_and_purpose": "", "header_hierarchy": [], "row_keys": [], "column_keys": [], "units": [],
  "important_cells": [], "comparisons": [], "grounded_summary": "", "ambiguous_structure": [], "confidence": 0.0
}

Parsed table source:
{structured}

Text/layout context:
{context}
"""
