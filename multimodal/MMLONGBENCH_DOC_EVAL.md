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

## Metrics

- `answer_score`: main answer metric. With `--extract_answers`, it scores `scored_prediction` from `extracted_answer` using MMLongBench-Doc-style rules: exact integer match, tolerant float/percentage match, ANLS for strings, and length-matched list scoring.
- `exact_match`: normalized exact answer match.
- `token_f1`: normalized token overlap F1.
- `numeric_match`: numeric correctness for `Int` and `Float` answers.
- `list_f1`: item-level F1 for list answers.
- `anls`: approximate normalized Levenshtein score for string answers.
- `extracted_answer` / `extracted_answer_format`: canonical answer and format produced by the evaluation model before rule scoring.
- `page_hit`: whether any retrieved evidence page overlaps the gold `evidence_pages`.
- `page_precision`, `page_recall`, `page_f1`: page-level evidence retrieval quality.
- `missing_workspace_rate`: share of questions whose document workspace was not built.

The score JSON contains `overall`, `by_answer_format`, `by_doc_type`, and `by_evidence_source` summaries, plus per-question rows for error analysis.
