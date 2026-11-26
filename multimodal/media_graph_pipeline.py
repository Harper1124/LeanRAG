from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import traceback
import unicodedata
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .evidence_store import build_evidence_vector_store
from .graph.mm_edge_builder import build_phase4_edges
from .io_utils import load_dataclasses
from .mm_node_builder import build_phase1_mm_graph
from .openai_clients import make_async_chat_func, make_chat_func, make_embedding_func
from .phase3.input_adapter import (
    atomic_write_json,
    atomic_write_jsonl,
    join_media_records,
    read_json_array,
    read_jsonl,
)
from .phase3.semantic_unit_builder import build_media_semantic_units
from .schema import MMMedia


LAYOUT_RELATION_TYPES = {
    "document_contains_page", "page_contains_node", "text_caption_of_media",
    "text_refers_to_media", "entity_link_media", "media_near_text",
    "media_same_page_media", "page_next_page", "page_prev_page",
}
DEFAULT_DESCRIPTION_TOKEN_LIMIT = 512
HIERARCHY_OUTPUTS = ("all_entities.json", "community.json", "generate_relations.json", "milvus_demo.db")


class MediaGraphPipelineError(RuntimeError):
    def __init__(self, message: str, manifest: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.manifest = manifest or {}


def normalize_entity_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip().strip("\"'“”‘’` ")
    return re.sub(r"\s+", " ", text).upper()


def normalize_description(value: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value or ""))).strip()


def split_source_ids(value: Any) -> list[str]:
    return sorted({part.strip() for part in str(value or "").split("|") if part.strip()})


def overwrite_grounded_summaries(
    media_rows: list[dict[str, Any]], processed_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    processed_by_id: dict[str, dict[str, Any]] = {}
    for row in processed_rows:
        media_id = str(row.get("media_id") or "").strip()
        if not media_id or media_id in processed_by_id:
            raise ValueError(f"processed_media.json has missing or duplicate media_id: {media_id!r}")
        processed_by_id[media_id] = row
    updated, trace = [], []
    seen: set[str] = set()
    for original in media_rows:
        row = dict(original)
        media_id = str(row.get("media_id") or "").strip()
        if not media_id or media_id in seen:
            raise ValueError(f"mm_media.json has missing or duplicate media_id: {media_id!r}")
        seen.add(media_id)
        processed = processed_by_id.get(media_id) or {}
        semantic = processed.get("semantic_content") or {}
        summary = normalize_description(semantic.get("grounded_summary")) if isinstance(semantic, dict) else ""
        if summary:
            changed = row.get("summary") != summary
            row["summary"] = summary
            trace.append({"media_id": media_id, "event": "summary_overwritten", "changed": changed})
        else:
            trace.append({"media_id": media_id, "event": "summary_preserved", "changed": False})
        updated.append(row)
    return updated, trace


def semantic_units_to_graph_chunks(units: Iterable[Any]) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    chunks, trace = [], []
    for unit in units:
        data = asdict(unit) if is_dataclass(unit) else dict(unit)
        media_id = str(data.get("media_id") or "").strip()
        graph_text = normalize_description(data.get("graph_text"))
        if graph_text:
            if not media_id:
                raise ValueError("semantic unit with non-empty graph_text is missing media_id")
            chunks.append({"hash_code": media_id, "text": graph_text})
        else:
            trace.append({"media_id": media_id, "event": "empty_graph_text_skipped"})
    return sorted(chunks, key=lambda item: item["hash_code"]), trace


def _token_count(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(re.findall(r"\w+|[^\w\s]", text, re.UNICODE))


def _bounded_join(
    values: Iterable[Any], label: str, token_limit: int,
    summarizer: Callable[[str], str] | None, warnings: list[dict[str, Any]], context: dict[str, Any],
) -> str:
    descriptions = sorted({normalize_description(value) for value in values if normalize_description(value)})
    joined = " | ".join(descriptions)
    if _token_count(joined) <= token_limit:
        return joined
    if summarizer is not None:
        prompt = f"Summarize these {label} descriptions without dropping factual claims:\n{joined}"
        try:
            summarized = normalize_description(summarizer(prompt))
            if summarized:
                return summarized
        except Exception as exc:
            warnings.append({**context, "code": "description_summary_failed", "message": str(exc)})
    warnings.append({**context, "code": "description_deterministically_truncated"})
    parts, tokens = [], 0
    for description in descriptions:
        cost = _token_count(description)
        if parts and tokens + cost > token_limit:
            break
        parts.append(description)
        tokens += cost
    return " | ".join(parts) or joined[: max(1, token_limit * 4)]


def merge_legacy_graphs(
    text_entities: list[dict[str, Any]], text_relations: list[dict[str, Any]],
    media_entities: list[dict[str, Any]], media_relations: list[dict[str, Any]],
    summarizer: Callable[[str], str] | None = None,
    token_limit: int = DEFAULT_DESCRIPTION_TOKEN_LIMIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    grouped_entities: dict[str, list[dict[str, Any]]] = {}
    for row in [*text_entities, *media_entities]:
        name = normalize_entity_name(row.get("entity_name"))
        if not name:
            warnings.append({"stage": "merge", "code": "empty_entity_name_skipped"})
            continue
        grouped_entities.setdefault(name, []).append(row)

    entities = []
    for name in sorted(grouped_entities):
        rows = grouped_entities[name]
        types: dict[str, int] = {}
        for row in rows:
            entity_type = normalize_entity_name(row.get("entity_type") or "OTHER")
            types[entity_type] = types.get(entity_type, 0) + max(1, len(split_source_ids(row.get("source_id"))))
        chosen_type = sorted(types, key=lambda item: (-types[item], item))[0]
        if len(types) > 1:
            warnings.append({"stage": "merge", "code": "entity_type_conflict", "entity_name": name,
                             "types": sorted(types), "selected": chosen_type})
        context = {"stage": "merge", "entity_name": name}
        entities.append({
            "entity_name": name,
            "entity_type": chosen_type,
            "description": _bounded_join((row.get("description") for row in rows), "entity", token_limit,
                                         summarizer, warnings, context),
            "source_id": "|".join(sorted({source for row in rows for source in split_source_ids(row.get("source_id"))})),
            "degree": 0,
        })

    grouped_relations: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for raw in [*text_relations, *media_relations]:
        rows = raw if isinstance(raw, list) else [raw]
        for row in rows:
            if not isinstance(row, dict):
                continue
            relation_kind = str(row.get("relation_type") or row.get("source") or "").strip().lower()
            if relation_kind in LAYOUT_RELATION_TYPES:
                warnings.append({"stage": "merge", "code": "layout_relation_excluded", "relation_type": relation_kind})
                continue
            src = normalize_entity_name(row.get("src_tgt") or row.get("src_id"))
            tgt = normalize_entity_name(row.get("tgt_src") or row.get("tgt_id"))
            if src and tgt:
                grouped_relations.setdefault((src, tgt), []).append(row)

    relations = []
    for (src, tgt) in sorted(grouped_relations):
        rows = grouped_relations[(src, tgt)]
        sources = sorted({source for row in rows for source in split_source_ids(row.get("source_id"))})
        descriptions = {normalize_description(row.get("description")) for row in rows if normalize_description(row.get("description"))}
        context = {"stage": "merge", "src_tgt": src, "tgt_src": tgt}
        relations.append({
            "src_tgt": src,
            "tgt_src": tgt,
            "description": _bounded_join(descriptions, "relation", token_limit, summarizer, warnings, context),
            "weight": max(len(sources), len(descriptions), 1),
            "source_id": "|".join(sources),
        })
    return entities, relations, warnings


def _normalize_extraction_output(
    entities: Iterable[dict[str, Any]], relations: Iterable[Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_entities = []
    for row in entities:
        normalized_entities.append({
            "entity_name": normalize_entity_name(row.get("entity_name")),
            "entity_type": normalize_entity_name(row.get("entity_type") or "OTHER"),
            "description": normalize_description(row.get("description")),
            "source_id": "|".join(split_source_ids(row.get("source_id"))),
            "degree": int(row.get("degree", 0) or 0),
        })
    normalized_relations = []
    for raw in relations:
        for row in raw if isinstance(raw, list) else [raw]:
            if isinstance(row, dict):
                relation_kind = str(row.get("relation_type") or row.get("source") or "").strip().lower()
                if relation_kind in LAYOUT_RELATION_TYPES:
                    continue
                normalized_relations.append({
                    "src_tgt": normalize_entity_name(row.get("src_tgt") or row.get("src_id")),
                    "tgt_src": normalize_entity_name(row.get("tgt_src") or row.get("tgt_id")),
                    "description": normalize_description(row.get("description")),
                    "weight": float(row.get("weight", 1) or 1),
                    "source_id": "|".join(split_source_ids(row.get("source_id"))),
                })
    return normalized_entities, normalized_relations


async def extract_legacy_media_graph(
    chunks: list[dict[str, str]], llm_callable: Callable[..., Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from GraphExtraction.chunk import triple_extraction

    entities, relations = await triple_extraction(
        {item["hash_code"]: item["text"] for item in chunks}, llm_callable
    )
    return _normalize_extraction_output(entities, relations)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_hierarchy_runner(working: Path, entity_path: Path, relation_path: Path,
                              config: dict[str, Any], llm: Callable, embedding: Callable) -> None:
    import build_graph

    mm_config = config.get("multimodal") or {}
    build_graph.hierarchical_clustering({
        "working_dir": str(working), "entity_path": str(entity_path), "relation_path": str(relation_path),
        "max_workers": int(mm_config.get("graph_max_workers", 1)),
        "embedding_max_workers": int(mm_config.get("graph_embedding_max_workers", 1)),
        "use_llm_func": llm, "embeddings_func": embedding,
    })


def _record_hierarchy_success(working: Path) -> None:
    """Remove stale lightweight failure state after the merged hierarchy succeeds."""
    (working / "graph_build_error.json").unlink(missing_ok=True)
    workspace_manifest = working / "manifest.json"
    if not workspace_manifest.exists():
        return
    try:
        value = json.loads(workspace_manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(value, dict) or "graph_status" not in value:
        return
    value["graph_status"] = "built"
    value["hierarchy_status"] = "completed"
    value["graph_input"] = "phase3/merged_graph"
    atomic_write_json(value, workspace_manifest)


def _require_hierarchy_outputs(working: Path) -> None:
    missing = [str(working / name) for name in HIERARCHY_OUTPUTS if not (working / name).exists()]
    if missing:
        raise MediaGraphPipelineError(f"hierarchy runner returned without required outputs: {missing}")


def run_media_graph_pipeline(
    working_dir: str | Path, config: dict[str, Any] | None = None, llm_mode: str = "none",
    skip_hierarchy: bool = False, force: bool = False,
    extraction_runner: Callable[[list[dict[str, str]]], Any] | None = None,
    summarizer: Callable[[str], str] | None = None,
    hierarchy_runner: Callable[..., None] | None = None,
    rebuild_retrieval: bool = True,
) -> dict[str, Any]:
    working = Path(working_dir).resolve()
    config = config or {}
    phase3 = working / "phase3"
    legacy_media = phase3 / "legacy_media_graph"
    merged = phase3 / "merged_graph"
    manifest_path = phase3 / "media_graph_manifest.json"
    manifest: dict[str, Any] = {"schema_version": "media_graph_pipeline.v1", "status": "running", "stages": {},
                                "warnings": [], "errors": [], "config": {"llm_mode": llm_mode,
                                "skip_hierarchy": skip_hierarchy,
                                "llm_model": (config.get("deepseek") or {}).get("model"),
                                "embedding_model": (config.get("glm") or {}).get("embedding_model")
                                                   or (config.get("glm") or {}).get("model")}}
    required = [working / name for name in ("mm_media.json", "processed_media.json", "entity.jsonl", "relation.jsonl")]
    try:
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing required inputs: {missing}")
        media_rows = read_json_array(working / "mm_media.json")
        processed_rows = read_json_array(working / "processed_media.json")
        media_rows, summary_trace = overwrite_grounded_summaries(media_rows, processed_rows)
        atomic_write_json(media_rows, working / "mm_media.json")
        manifest["stages"]["join_and_summary"] = {"status": "completed", "count": len(media_rows), "trace": summary_trace}

        input_paths = required + ([working / "mm_chunk.json"] if (working / "mm_chunk.json").exists() else [])
        build_signature = json.dumps({"llm_mode": llm_mode, "skip_hierarchy": skip_hierarchy,
                                      "rebuild_retrieval": rebuild_retrieval,
                                      "models": manifest["config"]}, sort_keys=True)
        fingerprint = hashlib.sha256(("".join(f"{path.name}:{_sha256(path)}\n" for path in input_paths)
                                      + build_signature).encode()).hexdigest()
        expected = [phase3 / "media_semantic_units.jsonl", phase3 / "media_graph_chunks.json",
                    legacy_media / "entity.jsonl", legacy_media / "relation.jsonl",
                    merged / "entity.jsonl", merged / "relation.jsonl"]
        if not skip_hierarchy:
            expected.extend(working / name for name in HIERARCHY_OUTPUTS)
        if rebuild_retrieval:
            expected.extend(working / name for name in ("mm_nodes.jsonl", "mm_edges_seed.jsonl", "mm_edges.jsonl"))
        if not force and manifest_path.exists() and all(path.exists() for path in expected):
            old = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if old.get("input_fingerprint") == fingerprint and old.get("status") == "completed":
                if not skip_hierarchy:
                    _record_hierarchy_success(working)
                old["reused"] = True
                return old
        manifest["input_fingerprint"] = fingerprint
        manifest["inputs"] = {path.name: {"path": str(path), "sha256": _sha256(path)} for path in input_paths}

        joined = join_media_records(processed_rows, media_rows)
        manifest["warnings"].extend(joined.errors)
        units, unit_errors = build_media_semantic_units(joined.joined)
        manifest["warnings"].extend(unit_errors)
        atomic_write_jsonl(units, phase3 / "media_semantic_units.jsonl")
        chunks, chunk_trace = semantic_units_to_graph_chunks(units)
        atomic_write_json(chunks, phase3 / "media_graph_chunks.json")
        manifest["stages"]["semantic_units"] = {"status": "completed", "units": len(units),
                                                   "graph_chunks": len(chunks), "trace": chunk_trace}

        if extraction_runner is not None:
            result = extraction_runner(chunks)
            if asyncio.iscoroutine(result):
                media_entities, media_relations = asyncio.run(result)
            else:
                media_entities, media_relations = result
            media_entities, media_relations = _normalize_extraction_output(media_entities, media_relations)
        elif chunks:
            if llm_mode != "configured":
                raise MediaGraphPipelineError("graph_text exists but --llm-mode none supplies no GraphExtraction LLM")
            async_llm = make_async_chat_func(config["deepseek"])
            media_entities, media_relations = asyncio.run(extract_legacy_media_graph(chunks, async_llm))
        else:
            media_entities, media_relations = [], []
        atomic_write_jsonl(media_entities, legacy_media / "entity.jsonl")
        atomic_write_jsonl(media_relations, legacy_media / "relation.jsonl")
        manifest["stages"]["media_graph_extraction"] = {"status": "completed", "entities": len(media_entities),
                                                          "relations": len(media_relations)}

        text_entities = read_jsonl(working / "entity.jsonl")
        text_relations = read_jsonl(working / "relation.jsonl")
        if summarizer is None and llm_mode == "configured":
            summarizer = make_chat_func(config["deepseek"])
        token_limit = int(((config.get("multimodal") or {}).get("media_graph") or {}).get(
            "description_token_limit", DEFAULT_DESCRIPTION_TOKEN_LIMIT))
        merged_entities, merged_relations, merge_warnings = merge_legacy_graphs(
            text_entities, text_relations, media_entities, media_relations, summarizer, token_limit
        )
        manifest["warnings"].extend(merge_warnings)
        atomic_write_jsonl(merged_entities, merged / "entity.jsonl")
        atomic_write_jsonl(merged_relations, merged / "relation.jsonl")
        manifest["stages"]["merge"] = {"status": "completed", "entities": len(merged_entities),
                                          "relations": len(merged_relations)}

        chat = make_chat_func(config["deepseek"]) if llm_mode == "configured" else None
        embedding = make_embedding_func(config["glm"]) if llm_mode == "configured" else None
        if skip_hierarchy:
            manifest["stages"]["hierarchy"] = {"status": "skipped"}
        else:
            if chat is None or embedding is None:
                raise MediaGraphPipelineError("hierarchy requires --llm-mode configured (or --skip-hierarchy)")
            (hierarchy_runner or _default_hierarchy_runner)(working, merged / "entity.jsonl",
                                                            merged / "relation.jsonl", config, chat, embedding)
            _require_hierarchy_outputs(working)
            _record_hierarchy_success(working)
            manifest["stages"]["hierarchy"] = {"status": "completed", "input": "phase3/merged_graph"}

        if rebuild_retrieval:
            retrieval_by_media = {unit.media_id: unit.retrieval_text for unit in units}
            trace = build_phase1_mm_graph(str(working), entity_file="phase3/merged_graph/entity.jsonl",
                                          media_retrieval_text=retrieval_by_media)
            edges = build_phase4_edges(working)
            retrieval_stage: dict[str, Any] = {"status": "completed", "nodes": trace.get("counts", {}).get("total_nodes"),
                                                "edges": len(edges)}
            if embedding is not None:
                media_items = load_dataclasses(working / "mm_media.json", MMMedia)
                dimension = int((config.get("multimodal") or {}).get("embedding_dim", 1024))
                build_evidence_vector_store(media_items, str(working), embedding, dimension, retrieval_by_media)
                retrieval_stage["evidence_store"] = "evidence_milvus.db or evidence_vectors.json"
            else:
                retrieval_stage["evidence_store"] = "skipped_without_configured_embedding"
            manifest["stages"]["retrieval_and_mm_graph"] = retrieval_stage

        manifest["status"] = "completed"
        manifest["reused"] = False
        manifest["counts"] = {
            "warnings": len(manifest["warnings"]), "errors": len(manifest["errors"]),
            "semantic_units": len(units), "media_graph_chunks": len(chunks),
            "media_entities": len(media_entities), "media_relations": len(media_relations),
            "merged_entities": len(merged_entities), "merged_relations": len(merged_relations),
        }
        manifest["outputs"] = {"media_semantic_units": "phase3/media_semantic_units.jsonl",
                               "media_graph_chunks": "phase3/media_graph_chunks.json",
                               "legacy_media_entity": "phase3/legacy_media_graph/entity.jsonl",
                               "legacy_media_relation": "phase3/legacy_media_graph/relation.jsonl",
                               "merged_entity": "phase3/merged_graph/entity.jsonl",
                               "merged_relation": "phase3/merged_graph/relation.jsonl"}
        if not skip_hierarchy:
            manifest["outputs"].update({
                "all_entities": "all_entities.json",
                "community": "community.json",
                "generate_relations": "generate_relations.json",
                "entity_vector_store": "milvus_demo.db",
            })
        if rebuild_retrieval:
            manifest["outputs"].update({
                "mm_nodes": "mm_nodes.jsonl",
                "mm_edges_seed": "mm_edges_seed.jsonl",
                "mm_edges": "mm_edges.jsonl",
            })
            for candidate in ("evidence_milvus.db", "evidence_vectors.json", "evidence_records.json"):
                if (working / candidate).exists():
                    manifest["outputs"][candidate.rsplit(".", 1)[0]] = candidate
        atomic_write_json(manifest, manifest_path)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["errors"].append({"message": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()})
        manifest["counts"] = {"warnings": len(manifest["warnings"]), "errors": len(manifest["errors"])}
        atomic_write_json(manifest, manifest_path)
        raise MediaGraphPipelineError(str(exc), manifest) from exc


def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8-sig") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError("config must be a YAML object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build media semantics into the legacy LeanRAG hierarchy.")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--llm-mode", choices=("none", "configured"), default="none")
    parser.add_argument("--skip-hierarchy", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = run_media_graph_pipeline(args.working_dir, load_config(args.config), args.llm_mode,
                                            args.skip_hierarchy, args.force)
    except MediaGraphPipelineError as exc:
        print(json.dumps(exc.manifest, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
