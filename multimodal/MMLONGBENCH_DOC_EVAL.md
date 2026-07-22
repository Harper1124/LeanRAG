# MMLongBench-Doc Evaluation for MM-LeanRAG

MMLongBench-Doc is a document QA benchmark with 1,090 questions over 135 PDFs. The useful fields for this pipeline are `doc_id`, `doc_type`, `question`, `answer`, `evidence_pages`, `evidence_sources`, and `answer_format`.

## Recommended Flow

1. Prepare the local dataset directory from the GitHub repository:

```bash
python -m multimodal.prepare_mmlongbench_doc --source github --output_dir datasets/mmlongbench_doc
```

If the server cannot access GitHub directly, clone or upload the repository data first, then prepare offline:

```bash
python -m multimodal.prepare_mmlongbench_doc \
  --output_dir datasets/mmlongbench_doc \
  --local_data_file /path/to/MMLongBench-Doc/data/samples.json \
  --local_documents_dir /path/to/MMLongBench-Doc/data/documents
```

For a fast smoke test with only the first two documents:

```bash
python -m multimodal.prepare_mmlongbench_doc --source github --output_dir datasets/mmlongbench_doc_smoke --max_docs 2
```

2. Build MM-LeanRAG workspaces:

```bash
python -m multimodal.build_docbench --docbench_dir datasets/mmlongbench_doc --working_root exp/mm_mmlongbench_doc --config config.yaml
```

3. Run prediction:

```bash
python -m multimodal.evaluate_docbench --dataset_dir datasets/mmlongbench_doc --working_root exp/mm_mmlongbench_doc --output_file results/mmlongbench_doc_predictions.jsonl --config config.yaml
```

4. Score predictions:

```bash
python -m multimodal.score_mmlongbench_doc --dataset_dir datasets/mmlongbench_doc --predictions_file results/mmlongbench_doc_predictions.jsonl --output_file results/mmlongbench_doc_scores.json
```

To follow the original MMLongBench-Doc evaluation flow, first extract a canonical answer from each free-form model response with an evaluation model, then score the extracted answer with the MMLongBench-Doc-style rules:

```bash
python -m multimodal.score_mmlongbench_doc \
  --dataset_dir datasets/mmlongbench_doc \
  --predictions_file results/mmlongbench_doc_predictions.jsonl \
  --output_file results/mmlongbench_doc_scores.json \
  --extract_answers \
  --evaluation_model qwen2.5:7b \
  --evaluation_base_url http://127.0.0.1:11444/v1 \
  --evaluation_api_key ollama
```

The evaluation model can also be configured in `config.yaml`:

```yaml
evaluation_model:
  model: "qwen2.5:7b"
  api_key: "ollama"
  base_url: "http://127.0.0.1:11444/v1"
  temperature: 0.0
  max_tokens: 256

evaluation_extraction:
  short_string_max_chars: 80
  short_identifier_max_chars: 80
  enable_guard: true
```

Or run the full chain:

```bash
python -m multimodal.run_mmlongbench_doc_eval --prepare --build --dataset_dir datasets/mmlongbench_doc --working_root exp/mm_mmlongbench_doc --output_dir results --config config.yaml
```

The end-to-end runner accepts the same extraction switch:

```bash
python -m multimodal.run_mmlongbench_doc_eval \
  --dataset_dir datasets/mmlongbench_doc \
  --working_root exp/mm_mmlongbench_doc \
  --output_dir results \
  --config config.yaml \
  --extract_answers
```

## Evaluation Answer Extraction Guard

When `--extract_answers` is enabled, scoring follows this chain:

```text
raw prediction
-> deterministic pre-check
-> evaluation model extraction when needed
-> parse extracted answer
-> normalize
-> extraction guard
-> scored_prediction
```

The pre-check bypasses the evaluation model when the raw prediction is already a stable answer, including standard `Not answerable`, short integers/floats/percentages, a single URL, a short identifier or filename, a safe short string answer for `Str`, or a clearly parseable list for `List`.

The guard normalizes template pollution such as `Not answerable|String`, removes extraction labels and bare format tags, falls back to the raw answer on empty or invalid extraction output, protects short canonical answers from unnecessary rewrites, and prevents `List` answers from being downgraded to `Not answerable` or a single scalar when the raw response contains multiple stable items.

Set `evaluation_extraction.enable_guard: false` to restore the older behavior after parsing and normalization. This disables deterministic pre-check and guard fallback decisions.

## Metrics

- `answer_score`: main answer metric. With `--extract_answers`, it scores `scored_prediction` from `extracted_answer` using exact integer match, tolerant float/percentage match, ANLS for strings, and partial list F1 for list answers.
- `official_raw_answer_score`: MMLongBench-Doc-style score on the original free-form prediction before answer extraction.
- `official_extracted_answer_score`: MMLongBench-Doc-style score on the canonical extracted answer, including strict length-matched list scoring.
- `exact_match`: normalized exact answer match.
- `token_f1`: normalized token overlap F1.
- `numeric_match`: numeric correctness for `Int` and `Float` answers.
- `list_f1`: item-level F1 for list answers.
- `list_partial_f1`: partial list F1 on the extracted answer. For `List` questions this is also used as `answer_score`.
- `anls`: approximate normalized Levenshtein score for string answers.
- `extracted_answer` / `extracted_answer_format`: canonical answer and format produced by the evaluation model before rule scoring.
- `scored_prediction`: final answer used by `answer_score` after extraction guard.
- `extraction_guard_applied`: whether the guard changed or bypassed extraction.
- `extraction_guard_action`: guard action, for example `bypass_short_numeric`, `bypass_raw_list`, `normalize_not_answerable`, `fallback_raw_prediction`, `fallback_raw_list`, `preserve_short_answer`, `coerce_numeric_format`, or `parse_error_fallback`.
- `extraction_guard_reason`: short reason for the guard action.
- `precheck_bypassed_llm`: whether deterministic pre-check avoided calling the evaluation model.
- `raw_normalized_prediction`: normalized raw prediction before LLM extraction.
- `pre_guard_extracted_answer`: parsed and normalized evaluation-model answer before guard fallback.
- `post_guard_scored_prediction`: final answer used for scoring.
- `page_hit`: whether any retrieved evidence page overlaps the gold `evidence_pages`.
- `page_precision`, `page_recall`, `page_f1`: page-level evidence retrieval quality.
- `missing_workspace_rate`: share of questions whose document workspace was not built.

The score JSON contains `overall`, `by_answer_format`, `by_doc_type`, and `by_evidence_source` summaries, plus per-question rows for error analysis.
