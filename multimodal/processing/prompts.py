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

CHART_ANALYSIS_PROMPT = """Analyze the supplied chart image and compact OCR evidence, then return one JSON object only.

Program OCR text, positions, and readable numbers are factual evidence. Program axis and legend assignments are candidates, not semantic truth.
- Use chart pixels with OCR positions to determine real x/y meaning, legend-to-series mapping, and chart meaning.
- Caption and document context may clarify purpose or conflicts, but they are not evidence for plotted values or trends.
- Do not copy a candidate role when pixels do not support it. Return "unknown" for an uncertain field.
- Do not invent or interpolate blurred numbers.
- Produce at most 6 qualitative_trends and at most 4 extrema_and_intersections.
- Every qualitative_trend item must identify its series, state its basis, and include visual_evidence or exact OCR quotes.
- ocr_evidence items must be exact text from the supplied OCR lines.
- Every number in an output field must occur in the supplied OCR evidence.
- Record caption/image conflicts in caption_consistency.
- Use unreadable_regions for unclear regions.
- Do not produce a summary, entities, graph knowledge, graph_text, retrieval_text, or a long free-form description.

Return this schema:
{
  "chart_type": "", "title": "", "x_axis": {}, "y_axis": {}, "legends": [], "series": [],
  "qualitative_trends": [
    {"series": "", "trend": "", "evidence": "", "visual_evidence": [], "ocr_evidence": []}
  ],
  "extrema_and_intersections": [],
  "caption_consistency": "", "unreadable_regions": [], "confidence": 0.0,
  "chart_grounding": {"visual_evidence": [], "ocr_evidence": [], "context_evidence": []},
  "semantic_confidence": 0.0
}

Compact chart evidence:
{evidence}
"""


CHART_SUMMARY_PROMPT = """Write a concise grounded chart summary and return one JSON object only.

Evidence policy:
- Use only the supplied validated claims. Do not inspect or reinterpret the original chart.
- Write 1 to 3 concise sentences describing the chart and its most important supported trends.
- Every sentence must cite one or more supporting_claim_ids from the supplied claims.
- Do not introduce a number, series, comparison, ranking, cause, or conclusion absent from those cited claims.
- Every number in a sentence must occur verbatim in its cited claims.
- Do not claim that one configuration causes an outcome unless the validated claim explicitly states causality.

Return this schema:
{
  "summary_sentences": [
    {"text": "", "supporting_claim_ids": ["claim_0"]}
  ],
  "confidence": 0.0
}

Validated chart evidence:
{evidence}
"""


# Backward-compatible import for callers outside the Phase 2 pipeline.
CHART_PROMPT = CHART_ANALYSIS_PROMPT

TABLE_COMPARISON_PROMPT = """Extract a small set of useful, verifiable facts from the compact table evidence and return one JSON object only.

Evidence policy:
- The supplied cells are the sole factual source for table values.
- Produce at most 8 important_cells and at most 6 comparisons; prefer overall rows, extrema, baselines, and material differences.
- Every important_cells item must contain row, col, value, and reason; value must exactly match that source cell.
- Every comparison must contain left_cell, operator, right_cell, and statement.
- left_cell and right_cell must each point directly to one numeric table cell.
- operator must be one of >, <, >=, <=, ==, or !=. The program will execute this relation and discard false comparisons.
- Each comparison is binary. Do not claim that multiple rows all outperform other rows in one comparison.
- The statement must identify the corresponding row labels and metric headers; do not generalize a metric comparison into overall model superiority.
- When structural_ambiguity is non-empty, comparisons must be an empty list.
- Do not infer missing values, rankings without cited values, extrapolated trends, entities, graph knowledge, graph_text, or retrieval_text.
- Put structural uncertainty in ambiguous_structure.

Return this schema:
{
  "title_and_purpose": "",
  "important_cells": [{"row": 0, "col": 0, "value": "", "reason": ""}],
  "comparisons": [
    {"left_cell": [3, 2], "operator": ">", "right_cell": [4, 2], "statement": ""}
  ],
  "ambiguous_structure": [], "confidence": 0.0, "semantic_confidence": 0.0
}

Compact table evidence:
{evidence}
"""


TABLE_SUMMARY_PROMPT = """Write a concise grounded summary from the validated table evidence and return one JSON object only.

Evidence policy:
- Use only the supplied validated comparisons and grounded candidate facts.
- Write 1 to 3 concise sentences describing the table's purpose and most important supported findings.
- Do not introduce a comparison, ranking, number, category, or conclusion absent from the supplied evidence.
- Every summary sentence must include source_cells citing all cells needed to verify it.
- Every summary sentence must include supporting_fact_ids from the supplied grounded_facts.
- Every number in a sentence must occur verbatim in one of its source cells.
- A validated comparison statement must be copied verbatim; do not reverse, broaden, or paraphrase its relation.
- Facts marked descriptive_only must not be turned into comparisons or rankings.

Return this schema:
{
  "summary_sentences": [
    {"text": "", "supporting_fact_ids": ["fact_0"], "source_cells": [[0, 0]]}
  ],
  "confidence": 0.0
}

Validated table evidence:
{evidence}
"""


# Backward-compatible import for callers outside the Phase 2 pipeline.
TABLE_PROMPT = TABLE_COMPARISON_PROMPT
