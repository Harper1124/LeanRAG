from __future__ import annotations

import base64
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
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", config.get("temperature", 0.1)),
        )
        return response.choices[0].message.content

    return chat


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
                content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}})
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=kwargs.get("temperature", config.get("temperature", 0.1)),
        )
        return response.choices[0].message.content

    return vlm


def _image_to_data_url(path: str) -> str:
    # VLM API 需要可访问的 image_url；本地文件用 base64 data URL 内联传入。
    path_obj = Path(path)
    mime_type = mimetypes.guess_type(path_obj.name)[0] or "image/png"
    with path_obj.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
