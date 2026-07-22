from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .openai_clients import resolve_api_key


ANSWER_EXTRACTION_PROMPT_PATH = Path(__file__).with_name("prompt_for_answer_extraction.md")
ANSWER_EXTRACTION_PROMPT = ANSWER_EXTRACTION_PROMPT_PATH.read_text(encoding="utf-8")


@dataclass
class ExtractedAnswer:
    answer: str
    answer_format: str
    raw_response: str
    error: str | None = None


def make_answer_extractor(config: dict[str, Any]):
    from openai import OpenAI

    client = OpenAI(api_key=resolve_api_key(config), base_url=config["base_url"])
    model = config["model"]
    prompt = str(config.get("prompt") or ANSWER_EXTRACTION_PROMPT)
    temperature = float(config.get("temperature", 0.0))
    max_tokens = int(config.get("max_tokens", 256))

    def extract(question: str, output: str) -> ExtractedAnswer:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"\n\nQuestion:{question}\nAnalysis:{output}\n",
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            raw = str(response.choices[0].message.content or "").strip()
            answer, answer_format = parse_extraction_response(raw)
            return ExtractedAnswer(answer=answer, answer_format=answer_format, raw_response=raw)
        except Exception as exc:
            return ExtractedAnswer(answer="Fail to answer", answer_format="Str", raw_response="", error=str(exc))

    return extract


def parse_extraction_response(text: str) -> tuple[str, str]:
    answer_match = re.search(
        r"Extracted\s+answer\s*:\s*(?P<answer>.*?)(?:\n\s*Answer\s+format\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    format_match = re.search(r"Answer\s+format\s*:\s*\[?\s*(?P<format>[A-Za-z]+)", text, flags=re.IGNORECASE)
    answer = answer_match.group("answer").strip() if answer_match else text.strip()
    answer_format = _normalize_answer_format(format_match.group("format") if format_match else "")
    return answer, answer_format


def _normalize_answer_format(value: str) -> str:
    text = str(value or "").strip().strip("[]").strip().lower()
    if text in {"int", "integer"}:
        return "Int"
    if text in {"float", "number"}:
        return "Float"
    if text in {"str", "string", "none", "unknown"}:
        return "Str"
    if text in {"list", "array"}:
        return "List"
    return "Str"
