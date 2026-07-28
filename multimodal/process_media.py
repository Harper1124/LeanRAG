from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .openai_clients import make_chat_func, make_vlm_func
from .processing.pipeline import process_workspace


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build Phase 2 evidence-aware processed_media.json.")
    parser.add_argument("--working_dir", required=True)
    parser.add_argument("--output_file", default="processed_media.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--offline", action="store_true", help="Skip VLM/LLM calls and emit programmatic fallback output.")
    args = parser.parse_args()

    config = _load_config(args.config)
    vlm_func = None if args.offline else _make_vlm(config)
    llm_func = None if args.offline else _make_llm(config)
    records = process_workspace(args.working_dir, args.output_file, vlm_func=vlm_func, llm_func=llm_func)
    target = Path(args.output_file)
    if not target.is_absolute():
        target = Path(args.working_dir) / target
    print(f"Processed {len(records)} media items into {target}")


def _load_config(path: str) -> dict:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}
    except (FileNotFoundError, ModuleNotFoundError):
        return {}


def _make_vlm(config: dict):
    mm = config.get("multimodal", {}) if isinstance(config, dict) else {}
    if not mm.get("vlm_model") or not mm.get("vlm_base_url"):
        return None
    return make_vlm_func(
        {
            "model": mm["vlm_model"],
            "base_url": mm["vlm_base_url"],
            "api_key": mm.get("vlm_api_key", ""),
            "api_key_env": mm.get("vlm_api_key_env", "DASHSCOPE_API_KEY"),
            "temperature": 0.0,
            "vlm_image_max_side": mm.get("vlm_image_max_side", 1400),
            "vlm_image_quality": mm.get("vlm_image_quality", 85),
        }
    )


def _make_llm(config: dict):
    llm = config.get("deepseek", {}) if isinstance(config, dict) else {}
    if not llm.get("model") or not llm.get("base_url"):
        return None
    return make_chat_func({**llm, "temperature": 0.0})


if __name__ == "__main__":
    main()
