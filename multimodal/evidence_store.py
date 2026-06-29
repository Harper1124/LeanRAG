from __future__ import annotations

from pathlib import Path
from typing import Callable

from .io_utils import read_json, write_json
from .schema import MMMedia, dataclass_to_dict


COLLECTION_NAME = "evidence_collection"


def build_evidence_vector_store(
    media_items: list[MMMedia],
    working_dir: str,
    embedding_func: Callable,
    dim: int,
) -> None:
    """Build an optional Milvus Lite index for image/table textual evidence."""
    import numpy as np

    working = Path(working_dir)
    records = [_media_record(item) for item in media_items if _media_text(item)]
    write_json(records, working / "evidence_records.json")
    if not records:
        return
    vectors = np.asarray(embedding_func([record["text"] for record in records]), dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    for record, vector in zip(records, vectors):
        record["dense"] = vector.tolist()
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=str(working / "evidence_milvus.db"))
        if client.has_collection(COLLECTION_NAME):
            client.drop_collection(COLLECTION_NAME)
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_name="dense_index",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        client.create_collection(
            collection_name=COLLECTION_NAME,
            dimension=dim,
            index_params=index_params,
            metric_type="IP",
            consistency_level="Strong",
        )
        client.insert(collection_name=COLLECTION_NAME, data=[_milvus_record(i, record) for i, record in enumerate(records)])
    except Exception:
        write_json(records, working / "evidence_vectors.json")


def search_evidence(
    working_dir: str,
    query_embedding,
    doc_id: str | None = None,
    topk: int = 5,
) -> list[dict]:
    """Search optional media evidence index; falls back to local vector JSON."""
    import numpy as np

    working = Path(working_dir)
    db_path = working / "evidence_milvus.db"
    if db_path.exists():
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=str(db_path))
            query = np.asarray(query_embedding, dtype=float)
            if query.ndim == 2:
                query = query[0]
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query.tolist()],
                limit=topk,
                filter=f'doc_id == "{doc_id}"' if doc_id else "",
                output_fields=["media_id", "doc_id", "modality", "path", "page", "text"],
            )
            return [
                {"score": hit.get("distance"), **hit.get("entity", {})}
                for hit in results[0]
            ]
        except Exception:
            pass
    vector_path = working / "evidence_vectors.json"
    if not vector_path.exists():
        return []
    records = [record for record in read_json(vector_path) if not doc_id or record.get("doc_id") == doc_id]
    query = np.asarray(query_embedding, dtype=float)
    if query.ndim == 2:
        query = query[0]
    scored = []
    for record in records:
        vector = np.asarray(record.get("dense", []), dtype=float)
        if vector.size == 0:
            continue
        score = _cosine(query, vector)
        item = {key: value for key, value in record.items() if key != "dense"}
        item["score"] = score
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:topk]


def _media_record(item: MMMedia) -> dict:
    record = dataclass_to_dict(item)
    record["text"] = _media_text(item)
    return record


def _milvus_record(index: int, record: dict) -> dict:
    keep = ["media_id", "doc_id", "modality", "path", "page", "text", "dense"]
    return {"id": index, **{key: record.get(key) for key in keep}}


def _media_text(item: MMMedia) -> str:
    return "\n".join(
        part for part in [item.caption, item.ocr_text, item.summary, item.table_markdown, item.table_html] if part
    )


def _cosine(a, b) -> float:
    import numpy as np

    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not denom:
        return 0.0
    return float(np.dot(a, b) / denom)
