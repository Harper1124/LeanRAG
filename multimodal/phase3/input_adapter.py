from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from .id_utils import (
    canonical_json,
    normalize_name,
    normalize_relation_type,
    normalize_text,
    normalized_name_key,
    stable_id,
)
from .schema import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalEntity,
    CanonicalRelation,
    EntityType,
    Grounding,
    GroundingKind,
    RefType,
    SchemaValidationError,
    SourceReference,
    bounded_number,
)


LEGACY_ENTITY_FIELDS = {"entity_name", "entity_type", "description", "source_id", "degree"}
LEGACY_RELATION_FIELDS = {
    "src_tgt", "tgt_src", "description", "weight", "source_id", "source", "relation_type"
}
SUPPORTED_MEDIA_TYPES = {"image", "chart", "table"}
ALL_MEDIA_TYPES = SUPPORTED_MEDIA_TYPES | {"noise", "generic"}

LEGACY_ENTITY_TYPE_MAP = {
    "MODEL": "MODEL",
    "DATASET": "DATASET",
    "METRIC": "METRIC",
    "METHOD": "METHOD",
    "TECHNOLOGY": "METHOD",
    "COMPONENT": "COMPONENT",
    "ORGANIZATION": "ORGANIZATION",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "GEO": "LOCATION",
    "CONCEPT": "CONCEPT",
    "INDUSTRY": "CONCEPT",
    "MATHEMATICS": "CONCEPT",
    "SOCIAL SCIENCES": "CONCEPT",
    "SOCIAL_SCIENCES": "CONCEPT",
    "PRODUCT": "OTHER",
    "EVENT": "OTHER",
    "NORMAL_ENTITY": "OTHER",
    "NORMAL ENTITY": "OTHER",
    "OTHER": "OTHER",
}


@dataclass(frozen=True)
class JoinedMedia:
    media_id: str
    media_type: str
    media: dict[str, Any]
    processed: dict[str, Any]


@dataclass
class MediaJoinResult:
    joined: list[JoinedMedia]
    skipped: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def read_json_array(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise SchemaValidationError(f"{path} must contain a JSON array of objects")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise SchemaValidationError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(value)
    return rows


def atomic_write_json(data: Any, path: str | Path) -> None:
    _atomic_write(path, json.dumps(_jsonable(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_write_jsonl(rows: Iterable[Any], path: str | Path) -> None:
    text = "".join(
        json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    _atomic_write(path, text)


def _atomic_write(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def normalize_entity_type(value: Any) -> str:
    key = normalize_text(value).upper().replace("-", "_")
    if key in {item.value for item in EntityType}:
        return key
    try:
        return LEGACY_ENTITY_TYPE_MAP[key]
    except KeyError as exc:
        raise SchemaValidationError(f"Unsupported legacy entity_type: {value!r}") from exc


def split_legacy_source_id(value: Any) -> list[str]:
    if not isinstance(value, str):
        raise SchemaValidationError("legacy source_id must be a string")
    source_ids = sorted({normalize_text(item) for item in value.split("|") if normalize_text(item)})
    if not source_ids:
        raise SchemaValidationError("legacy source_id must contain at least one source")
    return source_ids


def adapt_legacy_entities(
    rows: list[dict[str, Any]],
    valid_chunk_ids: set[str] | None = None,
) -> list[CanonicalEntity]:
    entities = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        extra = set(row) - LEGACY_ENTITY_FIELDS
        required = {"entity_name", "entity_type", "description", "source_id"}
        missing = required - set(row)
        if extra or missing:
            raise SchemaValidationError(f"legacy entity row {index} fields invalid; missing={sorted(missing)}, extra={sorted(extra)}")
        name = normalize_name(row["entity_name"])
        description = normalize_text(row["description"])
        if not name or not description:
            raise SchemaValidationError(f"legacy entity row {index} has empty name or description")
        entity_type = normalize_entity_type(row["entity_type"])
        source_ids = split_legacy_source_id(row["source_id"])
        _validate_chunk_sources(source_ids, valid_chunk_ids, f"legacy entity row {index}")
        refs = [_text_source_ref(source_id) for source_id in source_ids]
        entity_id = stable_id(
            "ent",
            "legacy_text_entity.v1",
            {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "name": normalized_name_key(name),
                "entity_type": entity_type,
                "description": description,
                "source_refs": source_ids,
            },
        )
        if entity_id in seen_ids:
            raise SchemaValidationError(f"duplicate canonical legacy entity ID at row {index}: {entity_id}")
        seen_ids.add(entity_id)
        entities.append(CanonicalEntity(
            entity_id=entity_id,
            entity_name=name,
            entity_type=entity_type,
            description=description,
            source_refs=refs,
            origin_modalities=["text"],
            confidence=1.0,
            aliases=[],
        ))
    return sorted(entities, key=lambda item: (normalized_name_key(item.entity_name), item.entity_type, item.entity_id))


def adapt_legacy_relations(
    rows: list[dict[str, Any]],
    entities: list[CanonicalEntity],
    valid_chunk_ids: set[str] | None = None,
) -> list[CanonicalRelation]:
    by_exact_name: dict[str, list[CanonicalEntity]] = {}
    by_normalized_name: dict[str, list[CanonicalEntity]] = {}
    for entity in entities:
        by_exact_name.setdefault(normalize_name(entity.entity_name), []).append(entity)
        by_normalized_name.setdefault(normalized_name_key(entity.entity_name), []).append(entity)
    relations = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        extra = set(row) - LEGACY_RELATION_FIELDS
        required = {"src_tgt", "tgt_src", "description", "weight", "source_id"}
        missing = required - set(row)
        if extra or missing:
            raise SchemaValidationError(f"legacy relation row {index} fields invalid; missing={sorted(missing)}, extra={sorted(extra)}")
        source = _resolve_legacy_endpoint(
            row["src_tgt"], by_exact_name, by_normalized_name, index, "src_tgt"
        )
        target = _resolve_legacy_endpoint(
            row["tgt_src"], by_exact_name, by_normalized_name, index, "tgt_src"
        )
        description = normalize_text(row["description"])
        if not description:
            raise SchemaValidationError(f"legacy relation row {index} has empty description")
        weight = bounded_number(row["weight"], f"legacy relation row {index} weight")
        relation_type = normalize_relation_type(row.get("relation_type") or row.get("source") or "RELATED_TO")
        source_ids = split_legacy_source_id(row["source_id"])
        _validate_chunk_sources(source_ids, valid_chunk_ids, f"legacy relation row {index}")
        refs = [_text_source_ref(source_id) for source_id in source_ids]
        relation_id = stable_id(
            "rel",
            "legacy_text_relation.v1",
            {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "source_entity_id": source.entity_id,
                "target_entity_id": target.entity_id,
                "relation_type": relation_type,
                "description": description,
                "source_refs": source_ids,
            },
        )
        if relation_id in seen_ids:
            raise SchemaValidationError(f"duplicate canonical legacy relation ID at row {index}: {relation_id}")
        seen_ids.add(relation_id)
        relations.append(CanonicalRelation(
            relation_id=relation_id,
            source_entity_id=source.entity_id,
            target_entity_id=target.entity_id,
            relation_type=relation_type,
            description=description,
            weight=weight,
            source_refs=refs,
            origin_modalities=["text"],
            confidence=1.0,
        ))
    return sorted(relations, key=lambda item: (item.source_entity_id, item.target_entity_id, item.relation_type, item.relation_id))


def join_media_records(
    processed_rows: list[dict[str, Any]],
    media_rows: list[dict[str, Any]],
) -> MediaJoinResult:
    media_by_id = _unique_media_index(media_rows, "mm_media.json")
    processed_by_id = _unique_media_index(processed_rows, "processed_media.json")
    joined: list[JoinedMedia] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for media_id in sorted(media_by_id):
        media = media_by_id[media_id]
        media_type = normalize_text(media.get("mapped_type") or media.get("type") or media.get("modality") or "generic").lower()
        if media_type not in ALL_MEDIA_TYPES:
            errors.append(_join_error(media_id, "unsupported_modality", f"Unsupported mm_media modality: {media_type}"))
            continue
        processed = processed_by_id.get(media_id)
        if media_type == "noise":
            skipped.append({"media_id": media_id, "reason": "noise"})
            if processed is not None:
                errors.append(_join_error(media_id, "noise_has_processed_record", "noise media must not have a Phase 2 record"))
            continue
        if media_type == "generic" and processed is None:
            skipped.append({"media_id": media_id, "reason": "generic_without_supported_phase2_mapping"})
            continue
        if processed is None:
            errors.append(_join_error(media_id, "missing_processed_media", "No processed_media record for indexable media"))
            continue
        processed_type = normalize_text(processed.get("media_type")).lower()
        if processed_type not in SUPPORTED_MEDIA_TYPES:
            errors.append(_join_error(media_id, "unsupported_processed_modality", f"Unsupported processed media_type: {processed_type}"))
            continue
        if media_type != "generic" and processed_type != media_type:
            errors.append(_join_error(
                media_id, "modality_conflict", f"mm_media modality {media_type} conflicts with processed media_type {processed_type}"
            ))
            continue
        joined.append(JoinedMedia(media_id=media_id, media_type=processed_type, media=media, processed=processed))
    for media_id in sorted(set(processed_by_id) - set(media_by_id)):
        errors.append(_join_error(media_id, "missing_mm_media", "processed_media record has no mm_media record"))
    return MediaJoinResult(joined=joined, skipped=skipped, errors=errors)


def _unique_media_index(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result = {}
    for index, row in enumerate(rows):
        media_id = normalize_text(row.get("media_id"))
        if not media_id:
            raise SchemaValidationError(f"{source} row {index} is missing media_id")
        if media_id in result:
            raise SchemaValidationError(f"{source} contains duplicate media_id: {media_id}")
        result[media_id] = row
    return result


def _resolve_legacy_endpoint(
    value: Any,
    by_exact_name: dict[str, list[CanonicalEntity]],
    by_normalized_name: dict[str, list[CanonicalEntity]],
    row_index: int,
    field_name: str,
) -> CanonicalEntity:
    exact_matches = by_exact_name.get(normalize_name(value), [])
    matches = exact_matches if exact_matches else by_normalized_name.get(normalized_name_key(value), [])
    if len(matches) != 1:
        reason = "missing" if not matches else "ambiguous"
        raise SchemaValidationError(f"legacy relation row {row_index} {field_name} is {reason}: {value!r}")
    return matches[0]


def _validate_chunk_sources(source_ids: list[str], valid: set[str] | None, context: str) -> None:
    if valid is None:
        return
    missing = sorted(set(source_ids) - valid)
    if missing:
        raise SchemaValidationError(f"{context} has unresolved text source_refs: {missing}")


def _text_source_ref(source_id: str) -> SourceReference:
    return SourceReference(
        ref_type=RefType.TEXT_CHUNK.value,
        ref_id=source_id,
        media_id=None,
        grounding=Grounding(
            kind=GroundingKind.TEXT_SPAN.value,
            locator={"source_file": "mm_chunk.json", "hash_code": source_id},
        ),
        confidence=1.0,
    )


def _join_error(media_id: str, code: str, message: str) -> dict[str, Any]:
    return {"stage": "join_media", "code": code, "message": message, "media_id": media_id, "retryable": False, "attempt": None}
