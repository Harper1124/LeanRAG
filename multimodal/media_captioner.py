from __future__ import annotations

from collections.abc import Callable

from .schema import MMMedia


# 用外部 VLM 为图片生成 caption/summary，提高图片证据的文本可检索性。
def caption_images(
    media_items: list[MMMedia],
    vlm_func: Callable,
    overwrite: bool = False,
) -> list[MMMedia]:
    """Generate captions/OCR summaries for image media with a user-supplied VLM."""
    for item in media_items:
        if item.modality != "image":
            continue
        if item.caption and item.summary and not overwrite:
            continue
        # prompt 强调可见文字、图表含义和标签，便于后续问答命中图中信息。
        prompt = (
            "Describe this document image for retrieval and question answering. Include visible text/OCR, "
            "people or objects and counts, chart title/axes/legend/colors/values, table headers and key "
            "numbers, shapes, layout, and named entities. Be factual and concise."
        )
        response = _call_flexible(vlm_func, prompt=prompt, image_paths=[item.path], image_path=item.path)
        text = str(response or "").strip()
        if text:
            if overwrite or not item.caption:
                item.caption = text
            if overwrite or not item.summary:
                item.summary = text
        elif not item.summary:
            # VLM 失败时保留已有 caption/OCR，至少给检索一个兜底文本。
            item.summary = item.caption or item.ocr_text or "image evidence"
    return media_items


def summarize_tables(
    media_items: list[MMMedia],
    llm_func: Callable,
    overwrite: bool = False,
) -> list[MMMedia]:
    """Generate natural-language summaries for table media with a user-supplied LLM."""
    for item in media_items:
        if item.modality != "table":
            continue
        if item.summary and not overwrite:
            continue
        table_text = item.table_markdown or item.table_html or item.ocr_text or item.caption
        if not table_text:
            item.summary = item.summary or "table evidence"
            continue
        # 表格摘要保留行列名、数字、比较关系和单位，避免只生成笼统描述。
        prompt = (
            "Summarize this document table for retrieval. Keep key row/column names, numbers, "
            f"comparisons, and units.\n\n{table_text}"
        )
        response = _call_flexible(llm_func, prompt=prompt, query=prompt)
        item.summary = str(response or table_text).strip()
    return media_items


def _call_flexible(func: Callable, **kwargs):
    # 兼容不同调用签名的 LLM/VLM 函数，按从结构化到单 prompt 的顺序尝试。
    try:
        return func(**kwargs)
    except TypeError:
        pass
    if "image_paths" in kwargs:
        try:
            return func(kwargs.get("prompt"), kwargs.get("image_paths"))
        except TypeError:
            pass
    if "query" in kwargs:
        try:
            return func(kwargs["query"])
        except TypeError:
            pass
    return func(kwargs.get("prompt", ""))
