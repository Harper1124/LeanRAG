from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any


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
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config["model"]

    def chat(query: str, system_prompt: str = "", **kwargs):
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
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config["model"]

    def vlm(query: str | None = None, context: str = "", image_paths: list[str] | None = None, **kwargs):
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
    path_obj = Path(path)
    mime_type = mimetypes.guess_type(path_obj.name)[0] or "image/png"
    with path_obj.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
