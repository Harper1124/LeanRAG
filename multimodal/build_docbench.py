from __future__ import annotations

import argparse
import json
import re
import traceback
from collections import defaultdict
from pathlib import Path

from .chunk_builder import build_mm_chunks_from_mineru, save_mm_artifacts
from .docbench_loader import load_docbench
from .io_utils import save_dataclasses, write_json, write_jsonl
from .media_captioner import caption_images, summarize_tables
from .media_linker import link_media_to_chunks, link_media_to_entities
from .mineru_parser import parse_pdf_with_mineru


def build_docbench(
    docbench_dir: str,
    working_root: str,
    mineru_backend: str = "pipeline",
    max_token_size: int = 1024,
    overlap_token_size: int = 128,
    force: bool = False,
    build_graph: bool = True,
    use_media_caption: bool = False,
    use_table_summary: bool = False,
    model_config: dict | None = None,
) -> list[dict]:
    samples = load_docbench(docbench_dir)
    docs = _group_by_doc(samples)
    manifests = []
    for doc_id, rows in docs.items():
        pdf_path = rows[0]["pdf_path"]
        working_dir = Path(working_root) / doc_id
        mineru_dir = working_dir / "mineru_output"
        working_dir.mkdir(parents=True, exist_ok=True)

        mineru_info = parse_pdf_with_mineru(
            pdf_path=pdf_path,
            output_dir=str(mineru_dir),
            mineru_backend=mineru_backend,
            force=force,
        )
        chunks, media_items = build_mm_chunks_from_mineru(
            mineru_output_dir=str(mineru_dir),
            doc_id=doc_id,
            source_pdf=pdf_path,
            max_token_size=max_token_size,
            overlap_token_size=overlap_token_size,
        )
        if use_media_caption:
            media_items = caption_images(media_items, _default_vlm_captioner)
        if use_table_summary:
            media_items = summarize_tables(media_items, _default_table_summarizer)
        chunks, media_items = link_media_to_chunks(chunks, media_items, embedding_func=None)
        artifact_paths = save_mm_artifacts(chunks, media_items, str(working_dir))

        graph_status = "skipped"
        if build_graph:
            _ensure_minimal_triples(working_dir, artifact_paths["leanrag_chunk_file"])
            media_items = link_media_to_entities(str(working_dir), chunks, media_items, embedding_func=None)
            save_dataclasses(media_items, working_dir / "mm_media.json")
            graph_status = _try_build_leanrag_graph(working_dir, model_config or {})

        manifest = {
            "doc_id": doc_id,
            "source_pdf": pdf_path,
            "mineru_output_dir": mineru_info["mineru_output_dir"],
            "mm_chunk_file": artifact_paths["mm_chunk_file"],
            "mm_media_file": artifact_paths["mm_media_file"],
            "leanrag_chunk_file": artifact_paths["leanrag_chunk_file"],
            "working_dir": str(working_dir),
            "entity_vector_db": str(working_dir / "milvus_demo.db"),
            "evidence_vector_db": str(working_dir / "evidence_milvus.db"),
            "graph_status": graph_status,
            "qa_count": len([row for row in rows if row.get("question")]),
        }
        write_json(manifest, working_dir / "manifest.json")
        manifests.append(manifest)
    write_json(manifests, Path(working_root) / "manifest.json")
    return manifests


def _group_by_doc(samples: list[dict]) -> dict[str, list[dict]]:
    docs = defaultdict(list)
    for sample in samples:
        docs[str(sample["doc_id"])].append(sample)
    return dict(docs)


def _ensure_minimal_triples(working_dir: Path, chunk_file: str) -> None:
    entity_path = working_dir / "entity.jsonl"
    relation_path = working_dir / "relation.jsonl"
    if entity_path.exists() and relation_path.exists():
        return
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    entities = {}
    relations = []
    for chunk in chunks:
        source_id = chunk["hash_code"]
        names = _extract_candidate_entities(chunk.get("text", ""))
        for name in names:
            current = entities.setdefault(
                name,
                {"entity_name": name, "entity_type": "CONCEPT", "description": "", "source_id": source_id, "degree": 0},
            )
            if source_id not in current["source_id"].split("|"):
                current["source_id"] += "|" + source_id
            if len(current["description"]) < 500:
                current["description"] = (current["description"] + " " + _sentence_around(chunk.get("text", ""), name)).strip()
        for left, right in zip(names, names[1:]):
            relations.append(
                {
                    "src_tgt": left,
                    "tgt_src": right,
                    "description": f"{left} and {right} co-occur in the same text chunk.",
                    "weight": 1,
                    "source_id": source_id,
                }
            )
    if not entities and chunks:
        text = chunks[0].get("text", "")[:200]
        entities["document"] = {
            "entity_name": "document",
            "entity_type": "DOCUMENT",
            "description": text,
            "source_id": chunks[0]["hash_code"],
            "degree": 0,
        }
    write_jsonl(entities.values(), entity_path)
    write_jsonl(relations, relation_path)


def _try_build_leanrag_graph(working_dir: Path, model_config: dict) -> str:
    try:
        import build_graph as lean_build_graph
        from query_graph import embedding

        use_llm_func = None
        if model_config.get("deepseek", {}).get("base_url"):
            from .openai_clients import make_chat_func

            use_llm_func = make_chat_func(model_config["deepseek"])

        lean_build_graph.WORKING_DIR = str(working_dir)
        global_config = {
            "max_workers": 4,
            "working_dir": str(working_dir),
            "embeddings_func": embedding,
            "use_llm_func": use_llm_func,
        }
        lean_build_graph.hierarchical_clustering(global_config)
        return "built"
    except Exception as exc:
        write_json(
            {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()},
            working_dir / "graph_build_error.json",
        )
        _write_lightweight_graph_artifacts(working_dir)
        return "lightweight"


def _write_lightweight_graph_artifacts(working_dir: Path) -> None:
    entity_path = working_dir / "entity.jsonl"
    relation_path = working_dir / "relation.jsonl"
    if not entity_path.exists():
        return
    entities = []
    with entity_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("parent", "root")
            item.setdefault("degree", 0)
            entities.append(item)
    relations = []
    if relation_path.exists():
        with relation_path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                item.setdefault("level", 0)
                relations.append(item)
    write_jsonl([entities], working_dir / "all_entities.json")
    write_jsonl(relations, working_dir / "generate_relations.json")
    write_jsonl(
        [
            {
                "entity_name": "root",
                "entity_description": "Lightweight fallback graph generated from text chunks.",
                "findings": [],
            }
        ],
        working_dir / "community.json",
    )


def _extract_candidate_entities(text: str, limit: int = 8) -> list[str]:
    phrases = re.findall(r"\b[A-Z][A-Za-z0-9%/-]*(?:\s+[A-Z][A-Za-z0-9%/-]*){0,4}\b", text)
    if not phrases:
        phrases = re.findall(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9%/-]{2,20}", text)
    seen = []
    for phrase in phrases:
        phrase = phrase.strip()
        if phrase.lower() in {"the", "this", "that", "table", "figure"}:
            continue
        if phrase not in seen:
            seen.append(phrase)
        if len(seen) >= limit:
            break
    return seen


def _sentence_around(text: str, entity: str) -> str:
    for sentence in re.split(r"(?<=[.!?。！？])\s+", text):
        if entity in sentence:
            return sentence[:300]
    return text[:300]


def _default_vlm_captioner(prompt=None, image_paths=None, image_path=None, **kwargs):
    del prompt, kwargs
    path = image_path or (image_paths[0] if image_paths else "")
    return f"Image evidence from {path}"


def _default_table_summarizer(prompt=None, **kwargs):
    del kwargs
    text = str(prompt or "")
    return " ".join(text.split())[:1000]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal DocBench artifacts for LeanRAG.")
    parser.add_argument("--docbench_dir", required=True)
    parser.add_argument("--working_root", required=True)
    parser.add_argument("--mineru_backend", default="pipeline")
    parser.add_argument("--max_token_size", type=int, default=1024)
    parser.add_argument("--overlap_token_size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_graph", action="store_true")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    full_config = _load_config(args.config)
    config = full_config.get("multimodal", {})
    manifests = build_docbench(
        docbench_dir=args.docbench_dir,
        working_root=args.working_root,
        mineru_backend=args.mineru_backend,
        max_token_size=args.max_token_size,
        overlap_token_size=args.overlap_token_size,
        force=args.force,
        build_graph=not args.skip_graph,
        use_media_caption=bool(config.get("use_media_caption", False)),
        use_table_summary=bool(config.get("use_table_summary", False)),
        model_config=full_config,
    )
    print(f"Built {len(manifests)} document workspaces under {args.working_root}")


def _load_config(path: str) -> dict:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        return {}


if __name__ == "__main__":
    main()
