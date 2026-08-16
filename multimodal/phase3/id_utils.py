from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any


ID_HASH_HEX_LENGTH = 24


def normalize_text(value: Any) -> str:
    """Normalize stored prose without changing its factual content."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: Any) -> str:
    return normalize_text(value)


def normalized_name_key(value: Any) -> str:
    return normalize_name(value).casefold()


def normalize_relation_type(value: Any) -> str:
    text = normalize_text(value).upper()
    text = re.sub(r"[^\w]+", "_", text, flags=re.UNICODE).strip("_")
    if not text:
        raise ValueError("relation_type must not be empty")
    return text


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): canonicalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow non-finite numbers")
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    raise TypeError(f"Unsupported canonical JSON value: {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_id(prefix: str, namespace: str, payload: Any, length: int = ID_HASH_HEX_LENGTH) -> str:
    digest_input = canonical_json({"namespace": namespace, "payload": payload}).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:length]
    return f"{prefix}_{digest}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_unique_ids(records: list[dict[str, Any]], field: str) -> None:
    seen: dict[str, str] = {}
    for record in records:
        value = str(record.get(field) or "")
        if not value:
            raise ValueError(f"Missing stable ID field {field}")
        serialized = canonical_json(record)
        if value in seen and seen[value] != serialized:
            raise ValueError(f"Stable ID collision for {field}={value}")
        if value in seen:
            raise ValueError(f"Duplicate {field}={value}")
        seen[value] = serialized
