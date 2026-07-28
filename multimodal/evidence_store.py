from __future__ import annotations

from pathlib import Path
from typing import Callable

from .io_utils import read_json, write_json
from .schema import MMMedia, dataclass_to_dict, is_indexable_media


COLLECTION_NAME = "evidence_collection"


# 可选媒体证据向量库：将图片/表格的文本摘要单独建索引，便于未来做媒体级召回。
def build_evidence_vector_store(
    media_items: list[MMMedia],
    working_dir: str,
    embedding_func: Callable,
    dim: int,
) -> None:
    """Build an optional Milvus Lite index for image/table textual evidence."""
    import numpy as np

    working = Path(working_dir)
    records = media_records_for_index(media_items)
    write_json(records, working / "evidence_records.json")
    if not records:
        return
    # embedding 结果同时写入 Milvus；如果 Milvus 不可用，则退化保存为本地 JSON 向量。
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
        # 优先走 Milvus Lite；查询失败时继续使用本地 JSON 兜底。
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(uri=str(db_path))
            query = np.asarray(query_embedding, dtype=float)
            if query.ndim == 2:
                query = query[0]
            results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query.tolist()],
                limit=max(topk, topk * 2),
                filter=f'doc_id == "{doc_id}"' if doc_id else "",
                output_fields=["media_id", "doc_id", "modality", "mapped_type", "type", "path", "page", "text"],
            )
            return [item for item in [
                {"score": hit.get("distance"), **hit.get("entity", {})}
                for hit in results[0]
            ] if is_indexable_media(item)][:topk]
        except Exception:
            pass
    vector_path = working / "evidence_vectors.json"
    if not vector_path.exists():
        return []
    # 本地 JSON 兜底检索：逐条计算 cosine，适合小规模或调试场景。
    records = [
        record for record in read_json(vector_path)
        if (not doc_id or record.get("doc_id") == doc_id) and is_indexable_media(record)
    ]
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
    # 保留原始媒体字段，并额外拼出可嵌入检索的 text 字段。
    record = dataclass_to_dict(item)
    record["text"] = _media_text(item)
    return record


def media_records_for_index(media_items: list[MMMedia]) -> list[dict]:
    """Build embedding records while retaining noise only in mm_media metadata."""
    return [_media_record(item) for item in media_items if is_indexable_media(item) and _media_text(item)]


def _milvus_record(index: int, record: dict) -> dict:
    keep = ["media_id", "doc_id", "modality", "mapped_type", "type", "path", "page", "text", "dense"]
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
