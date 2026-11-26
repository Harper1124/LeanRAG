from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .id_utils import normalize_name, normalize_relation_type, normalize_text, normalized_name_key, stable_id
from .input_adapter import normalize_entity_type
from .schema import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalEntity,
    CanonicalRelation,
    Grounding,
    MediaSemanticUnit,
    RefType,
    SourceReference,
    bounded_number,
)


LAYOUT_OBJECTS = {
    "axis", "axes", "line", "lines", "legend", "legends", "rectangle", "rectangles",
    "arrow", "arrows", "circle", "circles", "box", "boxes", "shape", "shapes",
}
ENTITY_FIELDS = {
    "key", "entity_name", "entity_type", "description", "confidence", "aliases", "evidence_ref_indices"
}
RELATION_FIELDS = {
    "source_key", "target_key", "relation_type", "description", "confidence", "evidence_ref_indices"
}
ROOT_FIELDS = {"entities", "relations"}

EXTRACTION_PROMPT = """Extract business-semantic entities and directed Entity-to-Entity relations from GRAPH_TEXT only.

Rules:
- Do not use outside knowledge and do not infer facts absent from GRAPH_TEXT.
- Every entity and relation description must copy an exact supporting clause or sentence from GRAPH_TEXT.
- Do not return visual layout objects such as axis, line, legend, rectangle, arrow, circle, box, or shape.
- entity_type must be one of MODEL, DATASET, METRIC, METHOD, COMPONENT, ORGANIZATION, PERSON, LOCATION, CONCEPT, OTHER.
- Each entity and relation must cite one or more valid evidence_ref_indices from 0 through {max_ref_index}.
- Relation endpoints use entity key values declared in the same response.
- confidence is evidence support, not a default value. Use 0.90-1.00 only when the entity or relation and its exact description are explicit in GRAPH_TEXT; 0.75-0.89 when explicit but locally ambiguous; below 0.75 when uncertain. Never copy the example score mechanically.
- Return exactly one JSON object and no prose, using this schema:
{{"entities":[{{"key":"e1","entity_name":"Example Model","entity_type":"MODEL","description":"Example Model uses Example Method.","confidence":0.92,"aliases":[],"evidence_ref_indices":[0]}},{{"key":"e2","entity_name":"Example Method","entity_type":"METHOD","description":"Example Model uses Example Method.","confidence":0.92,"aliases":[],"evidence_ref_indices":[0]}}],"relations":[{{"source_key":"e1","target_key":"e2","relation_type":"USES","description":"Example Model uses Example Method.","confidence":0.90,"evidence_ref_indices":[0]}}]}}

EVIDENCE_REF_LOCATORS (metadata only; do not treat it as factual content):
{evidence_ref_locators}

GRAPH_TEXT:
{graph_text}
"""


@dataclass
class ExtractionResult:
    entities: list[CanonicalEntity] = field(default_factory=list)
    relations: list[CanonicalRelation] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    successful_media_ids: set[str] = field(default_factory=set)


class ExtractionResponseError(ValueError):
    pass


def extract_media_graph(
    units: list[MediaSemanticUnit],
    llm_callable: Callable | None,
    media_types: dict[str, str],
    entity_min_confidence: float,
    relation_min_confidence: float,
    max_attempts: int = 2,
) -> ExtractionResult:
    entity_min_confidence = bounded_number(entity_min_confidence, "media_entity_min_confidence")
    relation_min_confidence = bounded_number(relation_min_confidence, "media_relation_min_confidence")
    if max_attempts != 2:
        raise ValueError("Phase 3 extraction retry contract requires exactly two maximum attempts")
    result = ExtractionResult()
    for unit in sorted(units, key=lambda item: item.media_id):
        modality = media_types[unit.media_id]
        if not unit.graph_text.strip():
            result.successful_media_ids.add(unit.media_id)
            result.trace.append({
                "stage": "entity_extraction", "media_id": unit.media_id,
                "event": "skipped_no_graph_facts", "source": "deterministic_guard",
            })
            continue
        if llm_callable is None:
            result.errors.append(_error(unit.media_id, "llm_unavailable", "LLM callable is unavailable", None))
            continue
        materialized = None
        last_error = ""
        last_raw = ""
        for attempt in range(1, max_attempts + 1):
            prompt = EXTRACTION_PROMPT.format(
                max_ref_index=len(unit.evidence_refs) - 1,
                evidence_ref_locators=json.dumps([
                    {
                        "index": index,
                        "kind": ref.grounding.kind,
                        "locator": ref.grounding.locator,
                    }
                    for index, ref in enumerate(unit.evidence_refs)
                ], ensure_ascii=False, sort_keys=True),
                graph_text=unit.graph_text,
            )
            if attempt > 1:
                prompt += (
                    f"\nThe previous response failed validation: {last_error}\n"
                    f"Previous response (possibly truncated):\n{last_raw[:2000]}\n"
                    "Correct that specific error. Return the complete corrected JSON object only."
                )
            try:
                raw = _call_llm(llm_callable, prompt)
                last_raw = _response_excerpt(raw)
                response = _parse_and_validate_root(raw)
                entities, local_keys, entity_trace = _materialize_entities(
                    unit, response["entities"], modality, entity_min_confidence
                )
                relations, relation_trace = _materialize_relations(
                    unit, response["relations"], modality, local_keys, relation_min_confidence
                )
                materialized = (entities, relations, entity_trace, relation_trace)
                break
            except Exception as exc:
                last_error = str(exc)
                result.trace.append({
                    "stage": "entity_extraction", "media_id": unit.media_id,
                    "event": "llm_attempt_failed", "attempt": attempt, "message": last_error,
                    "source": "llm",
                })
        if materialized is None:
            result.errors.append(_error(unit.media_id, "invalid_llm_response", last_error, max_attempts))
            continue
        entities, relations, entity_trace, relation_trace = materialized
        if not entities and not entity_trace:
            entity_trace.append({
                "stage": "entity_extraction", "media_id": unit.media_id,
                "event": "no_canonical_candidates", "source": "llm",
                "capability_boundary": "valid response contained no grounded business-semantic entities",
            })
        result.entities.extend(entities)
        result.relations.extend(relations)
        result.trace.extend(entity_trace + relation_trace)
        result.successful_media_ids.add(unit.media_id)
    result.entities.sort(key=lambda item: (item.source_refs[0].media_id or "", normalized_name_key(item.entity_name), item.entity_type, item.entity_id))
    result.relations.sort(key=lambda item: (item.source_refs[0].media_id or "", item.source_entity_id, item.target_entity_id, item.relation_type, item.relation_id))
    return result


def _materialize_entities(
    unit: MediaSemanticUnit,
    candidates: list[dict[str, Any]],
    modality: str,
    threshold: float,
) -> tuple[list[CanonicalEntity], dict[str, CanonicalEntity], list[dict[str, Any]]]:
    retained_by_identity: dict[tuple[str, str, str], CanonicalEntity] = {}
    key_to_identity: dict[str, tuple[str, str, str]] = {}
    trace = []
    seen_candidate_keys = set()
    for index, candidate in enumerate(candidates):
        _exact_fields(candidate, ENTITY_FIELDS, f"entities[{index}]")
        local_key = normalize_text(candidate["key"])
        if not local_key or local_key in seen_candidate_keys:
            raise ExtractionResponseError(f"entities[{index}].key is empty or duplicated")
        seen_candidate_keys.add(local_key)
        name = normalize_name(candidate["entity_name"])
        description = normalize_text(candidate["description"])
        entity_type = normalize_entity_type(candidate["entity_type"])
        confidence = bounded_number(candidate["confidence"], f"entities[{index}].confidence")
        aliases = _normalize_aliases(candidate["aliases"], name, unit.graph_text, index)
        evidence_indices = _validate_evidence_indices(candidate["evidence_ref_indices"], unit, f"entities[{index}]")
        if not name or not description:
            raise ExtractionResponseError(f"entities[{index}] has empty name or description")
        if _is_layout_object(name):
            trace.append(_filtered_trace(unit.media_id, "entity", "pure_visual_layout_object", name, confidence))
            continue
        if not _appears_in_graph(name, unit.graph_text):
            raise ExtractionResponseError(f"Entity {name!r} does not appear in graph_text")
        if not _description_appears_in_graph(description, unit.graph_text):
            raise ExtractionResponseError(f"Entity description for {name!r} is not an exact graph_text clause")
        if confidence < threshold:
            trace.append(_filtered_trace(unit.media_id, "entity", "below_confidence_threshold", name, confidence))
            continue
        identity = (normalized_name_key(name), entity_type, description)
        entity_id = stable_id(
            "ent",
            "media_entity.v1",
            {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "media_id": unit.media_id,
                "name": identity[0],
                "entity_type": entity_type,
                "description": description,
            },
        )
        entity = CanonicalEntity(
            entity_id=entity_id,
            entity_name=name,
            entity_type=entity_type,
            description=description,
            source_refs=_semantic_source_refs(unit, evidence_indices),
            origin_modalities=[modality],
            confidence=confidence,
            aliases=aliases,
        )
        existing = retained_by_identity.get(identity)
        if existing is None:
            retained_by_identity[identity] = entity
        else:
            retained_by_identity[identity] = CanonicalEntity(
                entity_id=entity.entity_id,
                entity_name=min(existing.entity_name, entity.entity_name, key=lambda value: (value.casefold(), value)),
                entity_type=entity.entity_type,
                description=entity.description,
                source_refs=_merge_source_refs(existing.source_refs, entity.source_refs),
                origin_modalities=[modality],
                confidence=max(existing.confidence, entity.confidence),
                aliases=sorted(set(existing.aliases + entity.aliases), key=lambda value: (value.casefold(), value)),
            )
        key_to_identity[local_key] = identity
    entities = list(retained_by_identity.values())
    entity_by_key = {
        key: retained_by_identity[identity]
        for key, identity in key_to_identity.items()
        if identity in retained_by_identity
    }
    return entities, entity_by_key, trace


def _materialize_relations(
    unit: MediaSemanticUnit,
    candidates: list[dict[str, Any]],
    modality: str,
    entity_by_key: dict[str, CanonicalEntity],
    threshold: float,
) -> tuple[list[CanonicalRelation], list[dict[str, Any]]]:
    retained: dict[tuple[str, str, str, str], CanonicalRelation] = {}
    trace = []
    for index, candidate in enumerate(candidates):
        _exact_fields(candidate, RELATION_FIELDS, f"relations[{index}]")
        source_key = normalize_text(candidate["source_key"])
        target_key = normalize_text(candidate["target_key"])
        confidence = bounded_number(candidate["confidence"], f"relations[{index}].confidence")
        description = normalize_text(candidate["description"])
        relation_type = normalize_relation_type(candidate["relation_type"])
        evidence_indices = _validate_evidence_indices(candidate["evidence_ref_indices"], unit, f"relations[{index}]")
        if not description:
            raise ExtractionResponseError(f"relations[{index}] has empty description")
        if not _description_appears_in_graph(description, unit.graph_text):
            raise ExtractionResponseError(f"Relation description at index {index} is not an exact graph_text clause")
        if confidence < threshold:
            trace.append(_filtered_trace(unit.media_id, "relation", "below_confidence_threshold", relation_type, confidence))
            continue
        if source_key not in entity_by_key or target_key not in entity_by_key:
            trace.append(_filtered_trace(unit.media_id, "relation", "missing_or_filtered_endpoint", relation_type, confidence))
            continue
        source = entity_by_key[source_key]
        target = entity_by_key[target_key]
        identity = (source.entity_id, target.entity_id, relation_type, description)
        relation_id = stable_id(
            "rel",
            "media_relation.v1",
            {
                "schema_version": CANONICAL_SCHEMA_VERSION,
                "media_id": unit.media_id,
                "source_entity_id": source.entity_id,
                "target_entity_id": target.entity_id,
                "relation_type": relation_type,
                "description": description,
            },
        )
        relation = CanonicalRelation(
            relation_id=relation_id,
            source_entity_id=source.entity_id,
            target_entity_id=target.entity_id,
            relation_type=relation_type,
            description=description,
            weight=confidence,
            source_refs=_semantic_source_refs(unit, evidence_indices),
            origin_modalities=[modality],
            confidence=confidence,
        )
        existing = retained.get(identity)
        if existing is None:
            retained[identity] = relation
        else:
            confidence = max(existing.confidence, relation.confidence)
            retained[identity] = CanonicalRelation(
                relation_id=relation.relation_id,
                source_entity_id=relation.source_entity_id,
                target_entity_id=relation.target_entity_id,
                relation_type=relation.relation_type,
                description=relation.description,
                weight=confidence,
                source_refs=_merge_source_refs(existing.source_refs, relation.source_refs),
                origin_modalities=[modality],
                confidence=confidence,
            )
    return list(retained.values()), trace


def _semantic_source_refs(unit: MediaSemanticUnit, indices: list[int]) -> list[SourceReference]:
    refs = []
    seen = set()
    for index in indices:
        evidence = unit.evidence_refs[index]
        locator_key = json.dumps(evidence.grounding.locator, ensure_ascii=False, sort_keys=True)
        if locator_key in seen:
            continue
        seen.add(locator_key)
        refs.append(SourceReference(
            ref_type=RefType.MEDIA_SEMANTIC_UNIT.value,
            ref_id=unit.chunk_id,
            media_id=unit.media_id,
            grounding=Grounding(kind=evidence.grounding.kind, locator=dict(evidence.grounding.locator)),
            confidence=evidence.confidence,
        ))
    return refs


def _call_llm(func: Callable, prompt: str) -> Any:
    try:
        return func(
            prompt=prompt,
            query=prompt,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except TypeError:
        return func(prompt)


def _response_excerpt(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _parse_and_validate_root(value: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(value, dict):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            raise ExtractionResponseError("LLM returned an empty response")
        if text.startswith("```"):
            match = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S | re.I)
            if not match:
                raise ExtractionResponseError("Invalid fenced JSON response")
            text = match.group(1)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExtractionResponseError(f"Invalid LLM JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ExtractionResponseError("LLM response must be one JSON object")
    _exact_fields(parsed, ROOT_FIELDS, "root")
    if not isinstance(parsed["entities"], list) or not isinstance(parsed["relations"], list):
        raise ExtractionResponseError("entities and relations must be arrays")
    if any(not isinstance(item, dict) for item in parsed["entities"] + parsed["relations"]):
        raise ExtractionResponseError("entity and relation candidates must be objects")
    return parsed


def _exact_fields(value: dict[str, Any], expected: set[str], context: str) -> None:
    if set(value) != expected:
        raise ExtractionResponseError(
            f"{context} fields mismatch; missing={sorted(expected - set(value))}, extra={sorted(set(value) - expected)}"
        )


def _validate_evidence_indices(value: Any, unit: MediaSemanticUnit, context: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ExtractionResponseError(f"{context}.evidence_ref_indices must be a non-empty array")
    result = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw < len(unit.evidence_refs):
            raise ExtractionResponseError(f"{context} has invalid evidence_ref index: {raw!r}")
        if raw not in result:
            result.append(raw)
    return sorted(result)


def _normalize_aliases(value: Any, name: str, graph_text: str, index: int) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ExtractionResponseError(f"entities[{index}].aliases must be a string array")
    result = []
    for raw in value:
        alias = normalize_name(raw)
        if not alias or normalized_name_key(alias) == normalized_name_key(name):
            continue
        if not _appears_in_graph(alias, graph_text):
            raise ExtractionResponseError(f"Alias {alias!r} does not appear in graph_text")
        if alias not in result:
            result.append(alias)
    return sorted(result, key=lambda item: (item.casefold(), item))


def _merge_source_refs(left: list[SourceReference], right: list[SourceReference]) -> list[SourceReference]:
    by_key = {}
    for ref in left + right:
        key = json.dumps(
            {
                "ref_type": ref.ref_type,
                "ref_id": ref.ref_id,
                "media_id": ref.media_id,
                "kind": ref.grounding.kind,
                "locator": ref.grounding.locator,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        by_key[key] = ref
    return [by_key[key] for key in sorted(by_key)]


def _appears_in_graph(name: str, graph_text: str) -> bool:
    needle = _conservative_match_text(name)
    haystack = _conservative_match_text(graph_text)
    return bool(needle) and re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _conservative_match_text(value: Any) -> str:
    text = normalize_text(value).casefold()
    text = re.sub(r"\bw\s*/\s*o\b", " without ", text)
    text = re.sub(r"\bw\s*/\b", " with ", text)
    return re.sub(r"(?:[^\w]|_)+", " ", text, flags=re.UNICODE).strip()


def _description_appears_in_graph(description: str, graph_text: str) -> bool:
    return normalize_text(description).casefold() in normalize_text(graph_text).casefold()


def _is_layout_object(name: str) -> bool:
    clean = re.sub(r"[^a-z]+", " ", normalized_name_key(name)).strip()
    return clean in LAYOUT_OBJECTS


def _filtered_trace(media_id: str, item_type: str, reason: str, label: str, confidence: float) -> dict[str, Any]:
    return {
        "stage": "entity_extraction", "media_id": media_id, "event": "filtered",
        "item_type": item_type, "reason": reason, "label": label,
        "confidence": confidence, "source": "llm",
    }


def _error(media_id: str, code: str, message: str, attempt: int | None) -> dict[str, Any]:
    return {
        "stage": "entity_extraction", "code": code, "message": message,
        "media_id": media_id, "retryable": False, "attempt": attempt,
    }
