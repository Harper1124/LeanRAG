# MMLongBench-Doc Evaluation for MM-LeanRAG

MMLongBench-Doc is a document QA benchmark with 1,090 questions over 135 PDFs. The useful fields for this pipeline are `doc_id`, `doc_type`, `question`, `answer`, `evidence_pages`, `evidence_sources`, and `answer_format`.

## Recommended Flow

1. Prepare the local dataset directory:

```bash
python -m multimodal.prepare_mmlongbench_doc --output_dir datasets/mmlongbench_doc
```

For a fast smoke test:

```bash
python -m multimodal.prepare_mmlongbench_doc --output_dir datasets/mmlongbench_doc_smoke --max_docs 2
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

Or run the full chain:

```bash
python -m multimodal.run_mmlongbench_doc_eval --prepare --build --dataset_dir datasets/mmlongbench_doc --working_root exp/mm_mmlongbench_doc --output_dir results --config config.yaml
```

## Metrics

- `answer_score`: main answer metric. It uses list F1 for `List`, tolerant numeric match for `Int`/`Float`, and max of exact match/token F1 for string answers.
- `exact_match`: normalized exact answer match.
- `token_f1`: normalized token overlap F1.
- `numeric_match`: numeric correctness for `Int` and `Float` answers.
- `list_f1`: item-level F1 for list answers.
- `page_hit`: whether any retrieved evidence page overlaps the gold `evidence_pages`.
- `page_precision`, `page_recall`, `page_f1`: page-level evidence retrieval quality.
- `missing_workspace_rate`: share of questions whose document workspace was not built.

The score JSON contains `overall`, `by_answer_format`, `by_doc_type`, and `by_evidence_source` summaries, plus per-question rows for error analysis.
