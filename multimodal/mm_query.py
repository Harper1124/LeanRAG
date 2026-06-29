from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .io_utils import load_dataclasses, read_json
from .schema import MMChunk, MMMedia, dataclass_to_dict


def query_mm_graph(
    global_config: dict,
    db,
    query: str,
    doc_id: str | None = None,
) -> tuple[str, dict]:
    """
    Multimodal LeanRAG query entry.

    Text retrieval is delegated to existing LeanRAG graph search when available.
    Returned text chunk ids are used to backfill attached images/tables.
    """
    del db
    working_dir = global_config["working_dir"]
    chunks, media_items = _load_mm_artifacts(working_dir)
    text_evidence, graph_evidence, selected_entities = _retrieve_text_evidence(global_config, query, chunks)
    if not text_evidence:
        text_evidence = _keyword_retrieve(query, chunks, topk=global_config.get("text_topk", 5))

    visual_evidence, table_evidence = _media_for_text_evidence(text_evidence, chunks, media_items)
    visual_evidence = visual_evidence[: global_config.get("max_images_per_query", 4)]
    table_evidence = table_evidence[: global_config.get("max_tables_per_query", 4)]
    context = _format_context(text_evidence, graph_evidence, visual_evidence, table_evidence)

    if visual_evidence and global_config.get("answer_with_vlm_when_media", True):
        answer = _call_vlm(global_config.get("use_vlm_func"), query, context, [item["path"] for item in visual_evidence])
        if answer is None:
            answer = _call_llm(global_config.get("use_llm_func"), query, context)
    else:
        answer = _call_llm(global_config.get("use_llm_func"), query, context)
    if answer is None:
        answer = context

    trace = {
        "text_evidence": text_evidence,
        "graph_evidence": graph_evidence,
        "visual_evidence": visual_evidence,
        "table_evidence": table_evidence,
        "selected_entities": selected_entities,
    }
    return str(answer), trace


def _retrieve_text_evidence(global_config: dict, query: str, chunks: list[MMChunk]):
    try:
        from database_utils import get_text_units, search_vector_search
    except Exception:
        return [], [], []
    embedding_func = global_config.get("embeddings_func")
    if not embedding_func:
        return [], [], []
    try:
        query_embedding = embedding_func(query)
        entity_results = search_vector_search(
            global_config["working_dir"],
            query_embedding,
            topk=global_config.get("topk", 10),
            level_mode=global_config.get("level_mode", 2),
        )
        source_ids = [item[-1] for item in entity_results]
        chunk_file = global_config.get("chunks_file") or str(Path(global_config["working_dir"]) / "leanrag_chunk.json")
        text_units = get_text_units(global_config["working_dir"], source_ids, chunk_file, k=global_config.get("text_topk", 5))
        chunk_by_hash = {chunk.hash_code: chunk for chunk in chunks}
        evidence = []
        for item in text_units:
            chunk = chunk_by_hash.get(item.get("hash_code"))
            evidence.append(_chunk_evidence(chunk, item.get("score", 0)) if chunk else item)
        selected = [{"entity_name": item[0], "parent": item[1], "description": item[2], "source_id": item[3]} for item in entity_results]
        return evidence, [], selected
    except Exception:
        return [], [], []


def _keyword_retrieve(query: str, chunks: list[MMChunk], topk: int = 5) -> list[dict]:
    query_terms = [term.lower() for term in query.split() if len(term) > 2]
    scored = []
    for chunk in chunks:
        text = chunk.text.lower()
        score = sum(text.count(term) for term in query_terms)
        if score:
            scored.append((score, chunk))
    if not scored:
        scored = [(1, chunk) for chunk in chunks[:topk]]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [_chunk_evidence(chunk, score) for score, chunk in scored[:topk]]


def _media_for_text_evidence(text_evidence: list[dict], chunks: list[MMChunk], media_items: list[MMMedia]):
    chunks_by_hash = {chunk.hash_code: chunk for chunk in chunks}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    media_by_id = {item.media_id: item for item in media_items}
    counter = Counter()
    for evidence in text_evidence:
        chunk = chunks_by_hash.get(evidence.get("hash_code")) or chunks_by_id.get(evidence.get("chunk_id"))
        if not chunk:
            continue
        counter.update(chunk.attached_media_ids)
    media = [media_by_id[media_id] for media_id, _ in counter.most_common() if media_id in media_by_id]
    visual = [_media_evidence(item, counter[item.media_id]) for item in media if item.modality == "image"]
    tables = [_media_evidence(item, counter[item.media_id]) for item in media if item.modality == "table"]
    return visual, tables


def _chunk_evidence(chunk: MMChunk, score: float = 0.0) -> dict:
    item = dataclass_to_dict(chunk)
    item["score"] = score
    return item


def _media_evidence(media: MMMedia, score: float = 0.0) -> dict:
    item = dataclass_to_dict(media)
    item["score"] = score
    return item


def _format_context(text_evidence, graph_evidence, visual_evidence, table_evidence) -> str:
    payload = {
        "text_evidence": text_evidence,
        "graph_evidence": graph_evidence,
        "visual_evidence": visual_evidence,
        "table_evidence": table_evidence,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _call_llm(func: Callable | None, query: str, context: str):
    if not func:
        return None
    prompt = "Answer the question using the provided evidence. Cite pages and media paths when useful."
    try:
        return func(query, system_prompt=f"{prompt}\n\n{context}")
    except TypeError:
        return func(f"{prompt}\n\nQuestion: {query}\n\nEvidence:\n{context}")


def _call_vlm(func: Callable | None, query: str, context: str, image_paths: list[str]):
    if not func:
        return None
    prompt = f"Answer the question using text and visual evidence.\n\nQuestion: {query}\n\nEvidence:\n{context}"
    try:
        return func(query=query, context=context, image_paths=image_paths)
    except TypeError:
        pass
    try:
        return func(prompt, image_paths)
    except TypeError:
        return func(prompt)


def _load_mm_artifacts(working_dir: str) -> tuple[list[MMChunk], list[MMMedia]]:
    working = Path(working_dir)
    chunks = load_dataclasses(working / "mm_chunk.json", MMChunk)
    media = load_dataclasses(working / "mm_media.json", MMMedia) if (working / "mm_media.json").exists() else []
    return chunks, media


def _load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return {}


def _install_default_model_funcs(config: dict[str, Any], full_config: dict[str, Any]) -> None:
    try:
        from .openai_clients import make_chat_func, make_vlm_func
    except Exception:
        return
    if "use_llm_func" not in config and full_config.get("deepseek", {}).get("base_url"):
        llm_conf = dict(full_config["deepseek"])
        llm_conf.setdefault("api_key_env", "DASHSCOPE_API_KEY")
        config["use_llm_func"] = make_chat_func(llm_conf)
    if "use_vlm_func" not in config:
        vlm_conf = {
            "model": config.get("vlm_model"),
            "base_url": config.get("vlm_base_url"),
            "api_key": config.get("vlm_api_key", ""),
            "api_key_env": config.get("vlm_api_key_env", "DASHSCOPE_API_KEY"),
        }
        if vlm_conf["model"] and vlm_conf["base_url"]:
            config["use_vlm_func"] = make_vlm_func(vlm_conf)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a multimodal LeanRAG working directory.")
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--doc_id", default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    full_config = _load_config(args.config)
    config = full_config.get("multimodal", {})
    _install_default_model_funcs(config, full_config)
    config.update(
        {
            "working_dir": args.working_dir,
            "chunks_file": str(Path(args.working_dir) / "leanrag_chunk.json"),
        }
    )
    answer, trace = query_mm_graph(config, None, args.query, doc_id=args.doc_id)
    print(answer)
    print(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
