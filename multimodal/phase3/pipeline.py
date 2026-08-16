from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .entity_extractor import ExtractionResult, extract_media_graph
from .id_utils import file_sha256
from .input_adapter import (
    adapt_legacy_entities,
    adapt_legacy_relations,
    atomic_write_json,
    atomic_write_jsonl,
    join_media_records,
    read_json_array,
    read_jsonl,
)
from .schema import CANONICAL_SCHEMA_VERSION, GENERATOR_VERSION, BuildStatus, SchemaValidationError
from .semantic_unit_builder import build_media_semantic_units
from .validators import (
    REQUIRED_INPUT_FILES,
    validate_chunk_rows,
    validate_media_extraction,
    validate_required_files,
    validate_semantic_units,
)


DEFAULT_CONFIG = {
    "enabled": True,
    "output_dir": "phase3",
    "media_entity_min_confidence": 0.75,
    "media_relation_min_confidence": 0.75,
    "require_grounding": True,
    "schema_version": CANONICAL_SCHEMA_VERSION,
    "generator_version": GENERATOR_VERSION,
    "llm_max_attempts": 2,
}


class Phase3BuildError(RuntimeError):
    def __init__(self, message: str, manifest: dict[str, Any]) -> None:
        super().__init__(message)
        self.manifest = manifest


def run_phase3(
    working_dir: str | Path,
    llm_callable: Callable | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    working = Path(working_dir)
    phase3_config = _phase3_config(config)
    output_dir = working / str(phase3_config["output_dir"])
    manifest_path = output_dir / "build_manifest.json"
    stages: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    inputs = _input_summaries(working)
    try:
        validate_required_files(working)
        if phase3_config["schema_version"] != CANONICAL_SCHEMA_VERSION:
            raise SchemaValidationError(
                f"Configured schema_version {phase3_config['schema_version']!r} does not match {CANONICAL_SCHEMA_VERSION!r}"
            )
        if int(phase3_config["llm_max_attempts"]) != 2:
            raise SchemaValidationError("llm_max_attempts must be 2 under the confirmed extraction contract")
        chunks = read_json_array(working / "mm_chunk.json")
        media_rows = read_json_array(working / "mm_media.json")
        processed_rows = read_json_array(working / "processed_media.json")
        legacy_entity_rows = read_jsonl(working / "entity.jsonl")
        legacy_relation_rows = read_jsonl(working / "relation.jsonl")
        chunk_hashes = validate_chunk_rows(chunks)
        text_entities = adapt_legacy_entities(legacy_entity_rows, chunk_hashes)
        text_relations = adapt_legacy_relations(legacy_relation_rows, text_entities, chunk_hashes)
        join_result = join_media_records(processed_rows, media_rows)
        errors.extend(join_result.errors)
        trace.extend({"stage": "input_validation", "event": "media_skipped", **item} for item in join_result.skipped)
        stages["step0_contract_and_inputs"] = {
            "status": BuildStatus.PARTIAL_FAILED.value if join_result.errors else BuildStatus.COMPLETED.value,
            "counts": {
                "mm_chunks": len(chunks),
                "mm_media": len(media_rows),
                "processed_media": len(processed_rows),
                "joined_media": len(join_result.joined),
                "skipped_media": len(join_result.skipped),
                "canonical_text_entities_validated": len(text_entities),
                "canonical_text_relations_validated": len(text_relations),
            },
            "errors": list(join_result.errors),
            "trace": [item for item in trace if item.get("stage") == "input_validation"],
        }
    except Exception as exc:
        fatal = _error("step0_contract_and_inputs", "input_validation_failed", str(exc))
        errors.append(fatal)
        stages["step0_contract_and_inputs"] = {
            "status": BuildStatus.FAILED.value,
            "counts": {},
            "errors": [fatal],
            "trace": [],
        }
        manifest = _manifest(phase3_config, BuildStatus.FAILED.value, inputs, {}, stages, errors, trace)
        atomic_write_json(manifest, manifest_path)
        raise Phase3BuildError(str(exc), manifest) from exc

    units, semantic_errors = build_media_semantic_units(join_result.joined)
    errors.extend(semantic_errors)
    joined_media_ids = {item.media_id for item in join_result.joined}
    try:
        validate_semantic_units(units, {str(row["media_id"]) for row in media_rows})
    except Exception as exc:
        fatal = _error("step1_semantic_units", "semantic_output_validation_failed", str(exc))
        errors.append(fatal)
        stages["step1_semantic_units"] = {
            "status": BuildStatus.FAILED.value,
            "counts": {"semantic_units": len(units)},
            "errors": semantic_errors + [fatal],
            "trace": [],
        }
        manifest = _manifest(phase3_config, BuildStatus.FAILED.value, inputs, {}, stages, errors, trace)
        atomic_write_json(manifest, manifest_path)
        raise Phase3BuildError(str(exc), manifest) from exc
    stages["step1_semantic_units"] = {
        "status": _stage_status(len(units), semantic_errors),
        "counts": {
            "eligible_joined_media": len(joined_media_ids),
            "semantic_units": len(units),
            "semantic_unit_failures": len(semantic_errors),
        },
        "errors": semantic_errors,
        "trace": [],
    }

    media_types = {item.media_id: item.media_type for item in join_result.joined}
    extraction = extract_media_graph(
        units,
        llm_callable,
        media_types,
        entity_min_confidence=phase3_config["media_entity_min_confidence"],
        relation_min_confidence=phase3_config["media_relation_min_confidence"],
        max_attempts=int(phase3_config["llm_max_attempts"]),
    )
    errors.extend(extraction.errors)
    trace.extend(extraction.trace)
    canonical_validation_failed = False
    try:
        validate_media_extraction(
            extraction.entities,
            extraction.relations,
            units,
            {str(row["media_id"]) for row in media_rows},
        )
    except Exception as exc:
        fatal = _error("step2_media_graph", "canonical_output_validation_failed", str(exc))
        errors.append(fatal)
        extraction.errors.append(fatal)
        canonical_validation_failed = True
    stages["step2_media_graph"] = {
        "status": _stage_status(len(extraction.successful_media_ids), extraction.errors),
        "counts": {
            "media_attempted": len(units),
            "media_completed": len(extraction.successful_media_ids),
            "entities": len(extraction.entities),
            "relations": len(extraction.relations),
            "filtered_items": sum(item.get("event") == "filtered" for item in extraction.trace),
            "media_failures": len({item.get("media_id") for item in extraction.errors if item.get("media_id")}),
        },
        "errors": extraction.errors,
        "trace": extraction.trace,
    }

    completed_media = len(extraction.successful_media_ids)
    target_media = len(join_result.joined) + len({item.get("media_id") for item in join_result.errors if item.get("media_id")})
    overall_status = (
        BuildStatus.FAILED.value
        if canonical_validation_failed
        else _overall_status(target_media, completed_media, errors)
    )
    if overall_status == BuildStatus.FAILED.value:
        manifest = _manifest(phase3_config, overall_status, inputs, {}, stages, errors, trace)
        atomic_write_json(manifest, manifest_path)
        raise Phase3BuildError("Phase 3 build failed; no complete media result was available", manifest)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "media_semantic_units": output_dir / "media_semantic_units.jsonl",
        "media_entity": output_dir / "media_entity.jsonl",
        "media_relation": output_dir / "media_relation.jsonl",
    }
    try:
        atomic_write_jsonl(units, output_paths["media_semantic_units"])
        atomic_write_jsonl(extraction.entities, output_paths["media_entity"])
        atomic_write_jsonl(extraction.relations, output_paths["media_relation"])
    except Exception as exc:
        fatal = _error("write_outputs", "atomic_output_write_failed", str(exc))
        errors.append(fatal)
        stages["write_outputs"] = {
            "status": BuildStatus.FAILED.value,
            "counts": {},
            "errors": [fatal],
            "trace": [],
        }
        manifest = _manifest(phase3_config, BuildStatus.FAILED.value, inputs, {}, stages, errors, trace)
        atomic_write_json(manifest, manifest_path)
        raise Phase3BuildError(str(exc), manifest) from exc
    outputs = {
        name: {
            "path": path.relative_to(working).as_posix(),
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
            "records": len(units) if name == "media_semantic_units" else len(extraction.entities) if name == "media_entity" else len(extraction.relations),
        }
        for name, path in output_paths.items()
    }
    manifest = _manifest(phase3_config, overall_status, inputs, outputs, stages, errors, trace)
    atomic_write_json(manifest, manifest_path)
    return manifest


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required only when loading a YAML config file") from exc
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError("config file must contain a YAML object")
    return value


def _phase3_config(config: dict[str, Any] | None) -> dict[str, Any]:
    provided = config or {}
    if "multimodal" in provided:
        provided = (provided.get("multimodal") or {}).get("phase3") or {}
    elif "phase3" in provided:
        provided = provided.get("phase3") or {}
    if not isinstance(provided, dict):
        raise ValueError("multimodal.phase3 config must be an object")
    unsupported = set(provided) - set(DEFAULT_CONFIG)
    if unsupported:
        raise ValueError(f"Unsupported Step 0-2 phase3 config keys: {sorted(unsupported)}")
    merged = {**DEFAULT_CONFIG, **provided}
    if not merged["enabled"]:
        raise ValueError("multimodal.phase3.enabled is false")
    return merged


def _input_summaries(working: Path) -> dict[str, Any]:
    summaries = {}
    for name in REQUIRED_INPUT_FILES:
        path = working / name
        if path.is_file():
            summaries[name] = {"path": name, "sha256": file_sha256(path), "size": path.stat().st_size}
        else:
            summaries[name] = {"path": name, "missing": True}
    return summaries


def _manifest(
    config: dict[str, Any],
    status: str,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    stages: dict[str, Any],
    errors: list[dict[str, Any]],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "generator_version": str(config["generator_version"]),
        "status": status,
        "config": dict(config),
        "inputs": inputs,
        "outputs": outputs,
        "stages": stages,
        "errors": errors,
        "trace": trace,
    }


def _stage_status(success_count: int, stage_errors: list[dict[str, Any]]) -> str:
    if not stage_errors:
        return BuildStatus.COMPLETED.value
    return BuildStatus.PARTIAL_FAILED.value if success_count else BuildStatus.FAILED.value


def _overall_status(target_media: int, completed_media: int, errors: list[dict[str, Any]]) -> str:
    if not errors:
        return BuildStatus.COMPLETED.value
    if completed_media > 0:
        return BuildStatus.PARTIAL_FAILED.value
    if target_media == 0 and not errors:
        return BuildStatus.COMPLETED.value
    return BuildStatus.FAILED.value


def _error(stage: str, code: str, message: str) -> dict[str, Any]:
    return {
        "stage": stage, "code": code, "message": message,
        "media_id": None, "retryable": False, "attempt": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build LeanRAG Phase 3 Step 0-2 artifacts.")
    parser.add_argument("--working-dir", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--llm-mode", choices=("none", "configured"), default="none")
    args = parser.parse_args(argv)
    full_config = load_config(args.config)
    llm_callable = None
    if args.llm_mode == "configured":
        from ..openai_clients import make_chat_func

        llm_callable = make_chat_func(full_config["deepseek"])
    try:
        manifest = run_phase3(args.working_dir, llm_callable=llm_callable, config=full_config)
    except Phase3BuildError as exc:
        print(json.dumps(exc.manifest, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
