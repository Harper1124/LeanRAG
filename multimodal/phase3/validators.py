from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .id_utils import canonical_json
from .schema import (
    CanonicalEntity,
    CanonicalRelation,
    MediaSemanticUnit,
    RefType,
    SchemaValidationError,
    SourceReference,
)


REQUIRED_INPUT_FILES = (
    "mm_chunk.json",
    "mm_media.json",
    "processed_media.json",
    "entity.jsonl",
    "relation.jsonl",
)

CANONICAL_LEGACY_ALIASES = {
    "name", "type", "source_id", "source", "target", "relation", "src_tgt", "tgt_src",
    "src_id", "tgt_id", "source_media_id", "source_chunk_id",
}


def validate_required_files(working_dir: str | Path) -> list[Path]:
    working = Path(working_dir)
    missing = [name for name in REQUIRED_INPUT_FILES if not (working / name).is_file()]
    if missing:
        raise SchemaValidationError(f"Missing required Phase 3 input files: {missing}")
    return [working / name for name in REQUIRED_INPUT_FILES]


def validate_chunk_rows(rows: list[dict[str, Any]]) -> set[str]:
    hashes = set()
    for index, row in enumerate(rows):
        hash_code = str(row.get("hash_code") or "").strip()
        if not hash_code:
            raise SchemaValidationError(f"mm_chunk.json row {index} is missing hash_code")
        if hash_code in hashes:
            raise SchemaValidationError(f"mm_chunk.json contains duplicate hash_code: {hash_code}")
        hashes.add(hash_code)
    return hashes


def validate_semantic_units(
    units: Iterable[MediaSemanticUnit],
    media_ids: set[str],
) -> None:
    seen_chunks = set()
    seen_media = set()
    for unit in units:
        MediaSemanticUnit.from_dict(asdict(unit))
        if unit.chunk_id in seen_chunks:
            raise SchemaValidationError(f"Duplicate semantic unit chunk_id: {unit.chunk_id}")
        if unit.media_id in seen_media:
            raise SchemaValidationError(f"Duplicate semantic unit media_id: {unit.media_id}")
        if unit.media_id not in media_ids:
            raise SchemaValidationError(f"Semantic unit references unknown media_id: {unit.media_id}")
        seen_chunks.add(unit.chunk_id)
        seen_media.add(unit.media_id)
        for source_ref in unit.evidence_refs:
            _validate_source_ref(source_ref, media_ids, {}, set())


def validate_canonical_graph(
    entities: Iterable[CanonicalEntity],
    relations: Iterable[CanonicalRelation],
    chunk_ids: set[str],
    media_ids: set[str],
    text_chunk_ids: set[str] | None = None,
) -> None:
    entity_list = list(entities)
    relation_list = list(relations)
    entity_ids = set()
    entity_payloads = set()
    for entity in entity_list:
        payload = asdict(entity)
        validate_no_legacy_aliases(payload)
        CanonicalEntity.from_dict(payload)
        if entity.entity_id in entity_ids:
            raise SchemaValidationError(f"Duplicate canonical entity_id: {entity.entity_id}")
        serialized = canonical_json(payload)
        if serialized in entity_payloads:
            raise SchemaValidationError(f"Duplicate canonical entity payload: {entity.entity_id}")
        entity_ids.add(entity.entity_id)
        entity_payloads.add(serialized)
        for source_ref in entity.source_refs:
            _validate_source_ref(source_ref, media_ids, _unit_media_map(chunk_ids, entity_list), text_chunk_ids or set())
    relation_ids = set()
    for relation in relation_list:
        payload = asdict(relation)
        validate_no_legacy_aliases(payload)
        CanonicalRelation.from_dict(payload)
        if relation.relation_id in relation_ids:
            raise SchemaValidationError(f"Duplicate canonical relation_id: {relation.relation_id}")
        if relation.source_entity_id not in entity_ids or relation.target_entity_id not in entity_ids:
            raise SchemaValidationError(f"Relation {relation.relation_id} has a missing entity endpoint")
        relation_ids.add(relation.relation_id)
        for source_ref in relation.source_refs:
            _validate_source_ref(source_ref, media_ids, {}, text_chunk_ids or set(), chunk_ids)


def validate_media_extraction(
    entities: Iterable[CanonicalEntity],
    relations: Iterable[CanonicalRelation],
    units: Iterable[MediaSemanticUnit],
    media_ids: set[str],
) -> None:
    unit_list = list(units)
    unit_map = {unit.chunk_id: unit.media_id for unit in unit_list}
    entity_list = list(entities)
    relation_list = list(relations)
    entity_ids = set()
    for entity in entity_list:
        payload = asdict(entity)
        validate_no_legacy_aliases(payload)
        CanonicalEntity.from_dict(payload)
        if entity.entity_id in entity_ids:
            raise SchemaValidationError(f"Duplicate media entity_id: {entity.entity_id}")
        entity_ids.add(entity.entity_id)
        for ref in entity.source_refs:
            _validate_source_ref(ref, media_ids, unit_map, set())
    relation_ids = set()
    for relation in relation_list:
        payload = asdict(relation)
        validate_no_legacy_aliases(payload)
        CanonicalRelation.from_dict(payload)
        if relation.relation_id in relation_ids:
            raise SchemaValidationError(f"Duplicate media relation_id: {relation.relation_id}")
        if relation.source_entity_id not in entity_ids or relation.target_entity_id not in entity_ids:
            raise SchemaValidationError(f"Media relation {relation.relation_id} has a missing entity endpoint")
        relation_ids.add(relation.relation_id)
        for ref in relation.source_refs:
            _validate_source_ref(ref, media_ids, unit_map, set())


def validate_no_legacy_aliases(record: dict[str, Any]) -> None:
    forbidden = sorted(set(record) & CANONICAL_LEGACY_ALIASES)
    if forbidden:
        raise SchemaValidationError(f"Canonical record contains legacy alias fields: {forbidden}")


def _validate_source_ref(
    source_ref: SourceReference,
    media_ids: set[str],
    unit_map: dict[str, str],
    text_chunk_ids: set[str],
    known_chunk_ids: set[str] | None = None,
) -> None:
    if source_ref.ref_type == RefType.MEDIA.value:
        if source_ref.ref_id not in media_ids or source_ref.media_id != source_ref.ref_id:
            raise SchemaValidationError(f"Unresolved media source_ref: {source_ref.ref_id}")
    elif source_ref.ref_type == RefType.MEDIA_SEMANTIC_UNIT.value:
        allowed = known_chunk_ids if known_chunk_ids is not None else set(unit_map)
        if source_ref.ref_id not in allowed:
            raise SchemaValidationError(f"Unresolved media semantic unit source_ref: {source_ref.ref_id}")
        if unit_map and unit_map.get(source_ref.ref_id) != source_ref.media_id:
            raise SchemaValidationError(f"source_ref media_id conflicts with semantic unit: {source_ref.ref_id}")
    elif source_ref.ref_type == RefType.TEXT_CHUNK.value:
        if source_ref.ref_id not in text_chunk_ids:
            raise SchemaValidationError(f"Unresolved text source_ref: {source_ref.ref_id}")


def _unit_media_map(chunk_ids: set[str], entities: list[CanonicalEntity]) -> dict[str, str]:
    result = {}
    for entity in entities:
        for ref in entity.source_refs:
            if ref.ref_type == RefType.MEDIA_SEMANTIC_UNIT.value and ref.ref_id in chunk_ids and ref.media_id:
                result[ref.ref_id] = ref.media_id
    return result
