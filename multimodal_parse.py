import argparse
import json
import os
import subprocess
from hashlib import md5
from pathlib import Path
from typing import Any

try:
    import tiktoken
except ImportError:
    tiktoken = None


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    return prefix + md5(content.encode()).hexdigest()


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text if item is not None)
    return str(text).strip()


def _first_value(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return default


def _normalize_page(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        page = int(value)
    except (TypeError, ValueError):
        return None
    return page + 1 if page == 0 else page


def _extract_page(item: dict[str, Any]) -> int | None:
    page_idx = item.get("page_idx")
    if page_idx not in (None, ""):
        try:
            return int(page_idx) + 1
        except (TypeError, ValueError):
            return None
    return _normalize_page(_first_value(item, ["page", "page_no"], None))


def _resolve_asset_path(asset_path: str | None, asset_root: Path) -> str | None:
    if not asset_path:
        return None
    path = Path(asset_path)
    if not path.is_absolute():
        path = asset_root / path
    return str(path)


def _table_to_text(item: dict[str, Any]) -> str:
    table_text = _first_value(
        item,
        ["table_body", "table", "html", "text", "md", "markdown"],
        "",
    )
    caption = _clean_text(_first_value(item, ["table_caption", "caption"], ""))
    footnote = _clean_text(_first_value(item, ["table_footnote", "footnote"], ""))
    parts = []
    if caption:
        parts.append(f"Table caption: {caption}")
    if table_text:
        parts.append(_clean_text(table_text))
    if footnote:
        parts.append(f"Table footnote: {footnote}")
    return "\n".join(parts).strip()


def _image_to_text(item: dict[str, Any]) -> str:
    caption = _clean_text(_first_value(item, ["image_caption", "img_caption", "caption"], ""))
    footnote = _clean_text(_first_value(item, ["image_footnote", "img_footnote", "footnote"], ""))
    path = _clean_text(_first_value(item, ["img_path", "image_path", "asset_path", "path"], ""))
    parts = ["Image evidence converted to text description."]
    if caption:
        parts.append(f"Caption: {caption}")
    if footnote:
        parts.append(f"Footnote: {footnote}")
    if path:
        parts.append(f"Asset path: {path}")
    return " ".join(parts).strip()


def _make_summary(text: str, modality: str, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    if not compact:
        return f"{modality} evidence"
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rstrip() + "..."


def _split_long_text(text: str, max_token_size: int, overlap_token_size: int) -> list[str]:
    if not text:
        return []
    if tiktoken is None:
        if len(text) <= max_token_size:
            return [text]
        step = max(1, max_token_size - overlap_token_size)
        return [text[start : start + max_token_size].strip() for start in range(0, len(text), step)]
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)
    if len(tokens) <= max_token_size:
        return [text]
    step = max(1, max_token_size - overlap_token_size)
    chunks = []
    for start in range(0, len(tokens), step):
        chunk = encoder.decode(tokens[start : start + max_token_size]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _content_item_to_evidence(
    item: dict[str, Any],
    asset_root: Path,
    source_pdf: str,
    max_token_size: int,
    overlap_token_size: int,
) -> list[dict[str, Any]]:
    raw_type = str(_first_value(item, ["type", "category", "modality"], "text")).lower()
    if "table" in raw_type:
        modality = "table"
        text = _table_to_text(item)
    elif "image" in raw_type or "img" in raw_type or "figure" in raw_type:
        modality = "image"
        text = _image_to_text(item)
    else:
        modality = "text"
        text = _clean_text(_first_value(item, ["text", "content", "md", "markdown"], ""))

    asset_path = _resolve_asset_path(
        _first_value(item, ["asset_path", "img_path", "image_path", "path"], None),
        asset_root,
    )
    page = _extract_page(item)
    summary = _clean_text(_first_value(item, ["summary"], "")) or _make_summary(text, modality)
    chunks = _split_long_text(text, max_token_size, overlap_token_size) if modality == "text" else [text]

    evidence = []
    for chunk_index, chunk_text in enumerate(chunks):
        content_for_hash = f"{source_pdf}|{page}|{modality}|{asset_path}|{chunk_index}|{chunk_text}"
        evidence.append(
            {
                "hash_code": compute_mdhash_id(content_for_hash),
                "text": chunk_text,
                "modality": modality,
                "page": page,
                "asset_path": asset_path,
                "summary": summary if len(chunks) == 1 else _make_summary(chunk_text, modality),
                "source_pdf": source_pdf,
                "chunk_order_index": chunk_index,
            }
        )
    return evidence


def _load_content_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    for key in ("content_list", "content", "items", "pages"):
        value = data.get(key)
        if isinstance(value, list):
            if key == "pages":
                flattened = []
                for page in value:
                    if not isinstance(page, dict):
                        continue
                    page_no = _first_value(page, ["page", "page_no", "page_idx"], None)
                    for item in page.get("items", page.get("content", [])):
                        if isinstance(item, dict):
                            item.setdefault("page", page_no)
                            flattened.append(item)
                return flattened
            return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Cannot find MinerU content list in {path}")


def _find_mineru_content_list(output_dir: Path, pdf_path: Path) -> Path:
    stem = pdf_path.stem
    candidates = sorted(output_dir.rglob(f"{stem}*content_list*.json"))
    if not candidates:
        candidates = sorted(output_dir.rglob("*content_list*.json"))
    if not candidates:
        raise FileNotFoundError(f"No MinerU content_list JSON found under {output_dir}")
    return candidates[0]


def run_mineru(pdf_path: Path, output_dir: Path, method: str = "auto") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = ["magic-pdf", "-p", str(pdf_path), "-o", str(output_dir), "-m", method]
    subprocess.run(command, check=True)
    return _find_mineru_content_list(output_dir, pdf_path)


def parse_mineru_content_list(
    content_list_file: str | Path,
    source_pdf: str | Path,
    asset_root: str | Path | None = None,
    max_token_size: int = 1024,
    overlap_token_size: int = 128,
) -> list[dict[str, Any]]:
    content_list_file = Path(content_list_file)
    source_pdf = Path(source_pdf)
    asset_root = Path(asset_root) if asset_root else content_list_file.parent
    records = []
    for item in _load_content_list(content_list_file):
        records.extend(
            _content_item_to_evidence(
                item,
                asset_root=asset_root,
                source_pdf=str(source_pdf),
                max_token_size=max_token_size,
                overlap_token_size=overlap_token_size,
            )
        )
    return [record for record in records if record["text"]]


def parse_pdf_plain_text(
    pdf_path: str | Path,
    max_token_size: int = 1024,
    overlap_token_size: int = 128,
) -> list[dict[str, Any]]:
    pdf_path = Path(pdf_path)
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Plain PDF fallback requires pypdf. Install MinerU and use --backend mineru, "
            "or install pypdf for text-only parsing."
        ) from exc

    records = []
    reader = PdfReader(str(pdf_path))
    for page_index, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text())
        for chunk_index, chunk_text in enumerate(_split_long_text(text, max_token_size, overlap_token_size)):
            records.append(
                {
                    "hash_code": compute_mdhash_id(f"{pdf_path}|{page_index}|text|{chunk_index}|{chunk_text}"),
                    "text": chunk_text,
                    "modality": "text",
                    "page": page_index,
                    "asset_path": None,
                    "summary": _make_summary(chunk_text, "text"),
                    "source_pdf": str(pdf_path),
                    "chunk_order_index": chunk_index,
                }
            )
    return records


def parse_pdf(
    pdf_path: str | Path,
    output_file: str | Path,
    backend: str = "mineru",
    mineru_output_dir: str | Path | None = None,
    mineru_content_list: str | Path | None = None,
    max_token_size: int = 1024,
    overlap_token_size: int = 128,
) -> list[dict[str, Any]]:
    pdf_path = Path(pdf_path)
    output_file = Path(output_file)
    if backend == "mineru":
        if mineru_content_list:
            content_list_file = Path(mineru_content_list)
        else:
            content_list_file = run_mineru(pdf_path, Path(mineru_output_dir or output_file.parent / "mineru"))
        records = parse_mineru_content_list(
            content_list_file,
            source_pdf=pdf_path,
            max_token_size=max_token_size,
            overlap_token_size=overlap_token_size,
        )
    elif backend == "plain":
        records = parse_pdf_plain_text(pdf_path, max_token_size, overlap_token_size)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PDF into LeanRAG enhanced chunk JSON.")
    parser.add_argument("pdf", help="Input PDF path.")
    parser.add_argument("-o", "--output", required=True, help="Output enhanced chunk JSON path.")
    parser.add_argument("--backend", choices=["mineru", "plain"], default="mineru")
    parser.add_argument("--mineru-output-dir", default=None)
    parser.add_argument("--mineru-content-list", default=None, help="Existing MinerU content_list JSON.")
    parser.add_argument("--max-token-size", type=int, default=1024)
    parser.add_argument("--overlap-token-size", type=int, default=128)
    args = parser.parse_args()

    records = parse_pdf(
        args.pdf,
        args.output,
        backend=args.backend,
        mineru_output_dir=args.mineru_output_dir,
        mineru_content_list=args.mineru_content_list,
        max_token_size=args.max_token_size,
        overlap_token_size=args.overlap_token_size,
    )
    print(f"Wrote {len(records)} enhanced chunks to {args.output}")


if __name__ == "__main__":
    main()
