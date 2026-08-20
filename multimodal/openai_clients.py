from __future__ import annotations

import base64
import io
import mimetypes
import os
from pathlib import Path
from typing import Any


# 从配置或环境变量中解析 API key，兼容 DASHSCOPE/OpenAI-compatible 服务。
def resolve_api_key(config: dict[str, Any]) -> str:
    env_name = config.get("api_key_env")
    if env_name and os.getenv(env_name):
        return os.getenv(env_name, "")
    value = config.get("api_key", "")
    if value and not str(value).startswith("${"):
        return str(value)
    if os.getenv("DASHSCOPE_API_KEY"):
        return os.getenv("DASHSCOPE_API_KEY", "")
    return str(value)


def make_chat_func(config: dict[str, Any]):
    # 生成一个普通文本聊天函数，供 mm_query 以 use_llm_func 形式调用。
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config["model"]

    def chat(query: str, system_prompt: str = "", **kwargs):
        # system_prompt 放检索证据和规则，user message 放原始问题。
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})
        request = dict(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", config.get("temperature", 0.1)),
        )
        response_format = kwargs.get("response_format")
        if response_format is not None:
            request["response_format"] = response_format
        response = client.chat.completions.create(**request)
        return response.choices[0].message.content

    return chat


def make_async_chat_func(config: dict[str, Any]):
    """Create the async OpenAI-compatible callable expected by GraphExtraction."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config["model"]

    async def chat(query: str, system_prompt: str = "", **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(kwargs.get("history_messages") or [])
        messages.append({"role": "user", "content": query})
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", config.get("temperature", 0.1)),
        )
        return response.choices[0].message.content

    return chat


def make_embedding_func(config: dict[str, Any]):
    """Create a batched OpenAI-compatible embedding callable."""
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config.get("embedding_model") or config["model"]

    def embed(texts: list[str]):
        import numpy as np

        response = client.embeddings.create(input=texts, model=model)
        return np.asarray([item.embedding for item in response.data], dtype=float)

    return embed


def make_vlm_func(config: dict[str, Any]):
    # 生成一个图文模型函数，接口兼容 OpenAI chat.completions 的 image_url 消息格式。
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config["model"]

    def vlm(query: str | None = None, context: str = "", image_paths: list[str] | None = None, **kwargs):
        # 将本地图片转成 data URL，避免调用方额外启动文件服务。
        prompt = kwargs.get("prompt") or query or ""
        if context:
            prompt = f"{prompt}\n\nEvidence:\n{context}"
        content = [{"type": "text", "text": prompt}]
        for image_path in image_paths or []:
            if image_path:
                data_url = _image_to_data_url(image_path, config)
                if data_url:
                    content.append({"type": "image_url", "image_url": {"url": data_url}})
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=kwargs.get("temperature", config.get("temperature", 0.1)),
        )
        return response.choices[0].message.content

    return vlm


def _image_to_data_url(path: str, config: dict[str, Any] | None = None) -> str | None:
    # VLM API 需要可访问的 image_url；本地文件用 base64 data URL 内联传入。
    config = config or {}
    path_obj = Path(path)
    image_bytes, mime_type = _compact_image_bytes(
        path_obj,
        max_side=int(config.get("vlm_image_max_side", 1024)),
        quality=int(config.get("vlm_image_quality", 75)),
    )
    if image_bytes is None:
        mime_type = mimetypes.guess_type(path_obj.name)[0] or "image/png"
        with path_obj.open("rb") as f:
            image_bytes = f.read()
        max_bytes = int(config.get("vlm_raw_image_max_bytes", 50000))
        if max_bytes > 0 and len(image_bytes) > max_bytes:
            return None
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _compact_image_bytes(path: Path, max_side: int, quality: int) -> tuple[bytes | None, str]:
    if max_side <= 0:
        return None, ""
    try:
        from PIL import Image
    except Exception:
        return None, ""
    try:
        with Image.open(path) as image:
            image.load()
            if max(image.size) > max_side:
                image.thumbnail((max_side, max_side))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            return buffer.getvalue(), "image/jpeg"
    except Exception:
        return None, ""
