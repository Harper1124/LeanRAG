from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar


CANONICAL_SCHEMA_VERSION = "phase3.canonical.v1"
MEDIA_SEMANTIC_UNIT_SCHEMA_VERSION = "phase3.media_semantic_unit.v1"
GENERATOR_VERSION = "leanrag.phase3.steps0-2.v1"


class EntityType(str, Enum):
    MODEL = "MODEL"
    DATASET = "DATASET"
    METRIC = "METRIC"
    METHOD = "METHOD"
    COMPONENT = "COMPONENT"
    ORGANIZATION = "ORGANIZATION"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    CONCEPT = "CONCEPT"
    OTHER = "OTHER"


class RefType(str, Enum):
    TEXT_CHUNK = "text_chunk"
    MEDIA_SEMANTIC_UNIT = "media_semantic_unit"
    MEDIA = "media"


class GroundingKind(str, Enum):
    TEXT_SPAN = "text_span"
    VISUAL_FACT = "visual_fact"
    OCR_SPAN = "ocr_span"
    TABLE_CELLS = "table_cells"
    CHART_EVIDENCE = "chart_evidence"


class BuildStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL_FAILED = "partial_failed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SchemaValidationError(ValueError):
    pass


def bounded_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise SchemaValidationError(f"{field_name} must be a number in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{field_name} must be a number in [0, 1]") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise SchemaValidationError(f"{field_name} must be a finite number in [0, 1]")
    return number


def _strict_keys(data: dict[str, Any], expected: set[str], schema_name: str) -> None:
    actual = set(data)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SchemaValidationError(f"{schema_name} fields mismatch; missing={missing}, extra={extra}")


@dataclass(frozen=True)
class Grounding:
    kind: str
    locator: dict[str, Any]

    FIELDS: ClassVar[set[str]] = {"kind", "locator"}

    def __post_init__(self) -> None:
        if self.kind not in {item.value for item in GroundingKind}:
            raise SchemaValidationError(f"Unsupported grounding kind: {self.kind}")
        if not isinstance(self.locator, dict) or not self.locator:
            raise SchemaValidationError("grounding.locator must be a non-empty object")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Grounding":
        _strict_keys(data, cls.FIELDS, "Grounding")
        return cls(kind=str(data["kind"]), locator=dict(data["locator"]))


@dataclass(frozen=True)
class SourceReference:
    ref_type: str
    ref_id: str
    media_id: str | None
    grounding: Grounding
    confidence: float

    FIELDS: ClassVar[set[str]] = {"ref_type", "ref_id", "media_id", "grounding", "confidence"}

    def __post_init__(self) -> None:
        if self.ref_type not in {item.value for item in RefType}:
            raise SchemaValidationError(f"Unsupported source ref type: {self.ref_type}")
        if not self.ref_id:
            raise SchemaValidationError("source ref_id must not be empty")
        if self.ref_type == RefType.TEXT_CHUNK.value and self.media_id is not None:
            raise SchemaValidationError("text_chunk source_refs must set media_id to null")
        if self.ref_type != RefType.TEXT_CHUNK.value and not self.media_id:
            raise SchemaValidationError("media source_refs require media_id")
        object.__setattr__(self, "confidence", bounded_number(self.confidence, "source_ref.confidence"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceReference":
        _strict_keys(data, cls.FIELDS, "SourceReference")
        grounding = data["grounding"]
        if not isinstance(grounding, dict):
            raise SchemaValidationError("source_ref.grounding must be an object")
        return cls(
            ref_type=str(data["ref_type"]),
            ref_id=str(data["ref_id"]),
            media_id=None if data["media_id"] is None else str(data["media_id"]),
            grounding=Grounding.from_dict(grounding),
            confidence=data["confidence"],
        )


@dataclass(frozen=True)
class GenerationInfo:
    schema_version: str
    generator_version: str
    confidence: float
    warnings: list[str]

    FIELDS: ClassVar[set[str]] = {"schema_version", "generator_version", "confidence", "warnings"}

    def __post_init__(self) -> None:
        if not self.schema_version or not self.generator_version:
            raise SchemaValidationError("generation versions must not be empty")
        object.__setattr__(self, "confidence", bounded_number(self.confidence, "generation.confidence"))
        if not isinstance(self.warnings, list) or any(not isinstance(item, str) for item in self.warnings):
            raise SchemaValidationError("generation.warnings must be a string array")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationInfo":
        _strict_keys(data, cls.FIELDS, "GenerationInfo")
        return cls(
            schema_version=str(data["schema_version"]),
            generator_version=str(data["generator_version"]),
            confidence=data["confidence"],
            warnings=list(data["warnings"]),
        )


@dataclass(frozen=True)
class MediaSemanticUnit:
    chunk_id: str
    media_id: str
    retrieval_text: str
    graph_text: str
    evidence_refs: list[SourceReference]
    generation: GenerationInfo

    FIELDS: ClassVar[set[str]] = {
        "chunk_id", "media_id", "retrieval_text", "graph_text", "evidence_refs", "generation"
    }

    def __post_init__(self) -> None:
        if not self.chunk_id.startswith("media_chunk_") or not self.media_id:
            raise SchemaValidationError("semantic unit requires stable chunk_id and media_id")
        if not self.retrieval_text:
            raise SchemaValidationError("retrieval_text must not be empty")
        if not self.evidence_refs:
            raise SchemaValidationError("semantic unit requires at least one evidence_ref")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MediaSemanticUnit":
        _strict_keys(data, cls.FIELDS, "MediaSemanticUnit")
        if not isinstance(data["evidence_refs"], list) or not isinstance(data["generation"], dict):
            raise SchemaValidationError("Invalid semantic unit nested fields")
        return cls(
            chunk_id=str(data["chunk_id"]),
            media_id=str(data["media_id"]),
            retrieval_text=str(data["retrieval_text"]),
            graph_text=str(data["graph_text"]),
            evidence_refs=[SourceReference.from_dict(item) for item in data["evidence_refs"]],
            generation=GenerationInfo.from_dict(data["generation"]),
        )


@dataclass(frozen=True)
class CanonicalEntity:
    entity_id: str
    entity_name: str
    entity_type: str
    description: str
    source_refs: list[SourceReference]
    origin_modalities: list[str]
    confidence: float
    aliases: list[str]

    FIELDS: ClassVar[set[str]] = {
        "entity_id", "entity_name", "entity_type", "description", "source_refs",
        "origin_modalities", "confidence", "aliases",
    }

    def __post_init__(self) -> None:
        if not self.entity_id.startswith("ent_") or not self.entity_name or not self.description:
            raise SchemaValidationError("canonical entity requires ID, name, and description")
        if self.entity_type not in {item.value for item in EntityType}:
            raise SchemaValidationError(f"Unsupported canonical entity type: {self.entity_type}")
        if not self.source_refs:
            raise SchemaValidationError("canonical entity requires source_refs")
        allowed_modalities = {"text", "image", "chart", "table"}
        if not self.origin_modalities or not set(self.origin_modalities).issubset(allowed_modalities):
            raise SchemaValidationError("canonical entity has invalid origin_modalities")
        if len(self.origin_modalities) != len(set(self.origin_modalities)):
            raise SchemaValidationError("origin_modalities must be deduplicated")
        if self.entity_name in self.aliases or len(self.aliases) != len(set(self.aliases)):
            raise SchemaValidationError("aliases must be unique and must not repeat entity_name")
        object.__setattr__(self, "confidence", bounded_number(self.confidence, "entity.confidence"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalEntity":
        _strict_keys(data, cls.FIELDS, "CanonicalEntity")
        return cls(
            entity_id=str(data["entity_id"]), entity_name=str(data["entity_name"]),
            entity_type=str(data["entity_type"]), description=str(data["description"]),
            source_refs=[SourceReference.from_dict(item) for item in data["source_refs"]],
            origin_modalities=list(data["origin_modalities"]), confidence=data["confidence"],
            aliases=list(data["aliases"]),
        )


@dataclass(frozen=True)
class CanonicalRelation:
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    description: str
    weight: float
    source_refs: list[SourceReference]
    origin_modalities: list[str]
    confidence: float

    FIELDS: ClassVar[set[str]] = {
        "relation_id", "source_entity_id", "target_entity_id", "relation_type", "description",
        "weight", "source_refs", "origin_modalities", "confidence",
    }

    def __post_init__(self) -> None:
        if not self.relation_id.startswith("rel_"):
            raise SchemaValidationError("canonical relation requires stable relation_id")
        if not self.source_entity_id.startswith("ent_") or not self.target_entity_id.startswith("ent_"):
            raise SchemaValidationError("canonical relation endpoints must be entity IDs")
        if not self.relation_type or not self.description or not self.source_refs:
            raise SchemaValidationError("canonical relation requires type, description, and source_refs")
        if not self.origin_modalities or len(self.origin_modalities) != len(set(self.origin_modalities)):
            raise SchemaValidationError("relation origin_modalities must be non-empty and deduplicated")
        object.__setattr__(self, "weight", bounded_number(self.weight, "relation.weight"))
        object.__setattr__(self, "confidence", bounded_number(self.confidence, "relation.confidence"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalRelation":
        _strict_keys(data, cls.FIELDS, "CanonicalRelation")
        return cls(
            relation_id=str(data["relation_id"]), source_entity_id=str(data["source_entity_id"]),
            target_entity_id=str(data["target_entity_id"]), relation_type=str(data["relation_type"]),
            description=str(data["description"]), weight=data["weight"],
            source_refs=[SourceReference.from_dict(item) for item in data["source_refs"]],
            origin_modalities=list(data["origin_modalities"]), confidence=data["confidence"],
        )


@dataclass(frozen=True)
class ErrorRecord:
    stage: str
    code: str
    message: str
    media_id: str | None = None
    retryable: bool = False
    attempt: int | None = None


@dataclass
class StageRecord:
    status: str
    counts: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
