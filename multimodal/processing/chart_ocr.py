from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any


VALUE_RE = re.compile(r"^[<>~≈]?[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?(?:[kKmMbB%])?$")


def extract_chart_ocr(
    path: str,
    ocr_func=None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract positioned OCR items from a chart without using the VLM."""
    config = config or {}
    if ocr_func:
        try:
            return _normalize_custom_result(ocr_func(path))
        except Exception as exc:
            return _empty_result("custom", f"custom OCR failed: {exc}")
    backend = str(config.get("backend") or "pytesseract").lower()
    if backend not in {"pytesseract", "tesseract", "auto"}:
        return _empty_result(backend, f"unsupported chart OCR backend: {backend}")
    return _pytesseract_ocr(path, config)


def parse_chart_layout(ocr_result: dict[str, Any]) -> dict[str, Any]:
    items = [item for item in ocr_result.get("items", []) if str(item.get("text") or "").strip()]
    width = int(ocr_result.get("width") or 0)
    height = int(ocr_result.get("height") or 0)
    lines = _group_lines(items)
    full_lines = [line for line in lines if line.get("region") == "full"]
    rotated_y = [line for line in lines if line.get("region") == "y_axis_rotated"]
    text_lines = [line for line in lines if line.get("region") == "text_only"]

    x_tokens = [item for item in items if _region(item, width, height) == "x_axis"]
    y_tokens = [item for item in items if _region(item, width, height) == "y_axis"]
    x_ticks = [_value_record(item, "x_tick") for item in x_tokens if _is_value(item.get("text"))]
    y_ticks = [_value_record(item, "y_tick") for item in y_tokens if _is_value(item.get("text"))]

    x_labels = _axis_label_candidates(full_lines, width, height, axis="x")
    y_labels = _axis_label_candidates(full_lines, width, height, axis="y")
    for line in text_lines:
        text = str(line.get("text") or "")
        if re.search(r"\bx[- ]?axis\b|horizontal axis", text, re.I):
            x_labels.append(_line_record(line))
        if re.search(r"\by[- ]?axis\b|vertical axis", text, re.I):
            y_labels.append(_line_record(line))
    for line in rotated_y:
        text = str(line.get("text") or "").strip()
        if _has_letters(text) and not _is_value(text):
            y_labels.append(_line_record(line))
    x_labels = _dedupe_records(x_labels)
    y_labels = _dedupe_records(y_labels)

    legend_candidates = _legend_candidates(full_lines, x_labels, y_labels, width, height)
    legend_candidates.extend(
        _line_record(line)
        for line in text_lines
        if re.search(r"\blegend\b|\bseries\b", str(line.get("text") or ""), re.I)
    )
    legend_candidates = _dedupe_records(legend_candidates)
    title = _title_candidate(full_lines, legend_candidates, width, height)
    if not title:
        title = next(
            (
                str(line.get("text") or "")
                for line in text_lines
                if _has_letters(line.get("text"))
                and not re.search(r"\b(?:x|y)[- ]?axis\b|\blegend\b|\bseries\b", str(line.get("text") or ""), re.I)
            ),
            "",
        )

    readable_values = []
    readable_data_points = []
    for item in items:
        if item.get("region") not in {"full", "text_only"} or not _is_value(item.get("text")):
            continue
        role = _region(item, width, height)
        record = _value_record(
            item,
            role if role in {"x_axis", "y_axis"} else "ocr_value" if item.get("region") == "text_only" else "plot_value",
        )
        readable_values.append(record)
        if role == "plot":
            readable_data_points.append(record)
    for line in text_lines:
        values = [item["text"] for item in items if item.get("line_id") == line.get("line_id") and _is_value(item.get("text"))]
        text = str(line.get("text") or "")
        if values and not re.search(r"\b(?:x|y)[- ]?axis\b", text, re.I):
            readable_data_points.append(
                {
                    "text": text,
                    "values": values,
                    "bbox": None,
                    "confidence": float(line.get("confidence", 0.0)),
                    "source": "mineru_ocr",
                }
            )

    x_axis = _axis_record(x_labels, x_ticks)
    y_axis = _axis_record(y_labels, y_ticks)
    parse_confidence = _parse_confidence(ocr_result, x_axis, y_axis, legend_candidates, readable_values)
    return {
        "ocr_text": "\n".join(line["text"] for line in lines if line.get("text")),
        "ocr_backend": ocr_result.get("backend", ""),
        "ocr_status": ocr_result.get("status", "unavailable"),
        "ocr_error": ocr_result.get("error", ""),
        "ocr_items": items,
        "ocr_lines": lines,
        "numeric_tokens": list(dict.fromkeys(record["text"] for record in readable_values)),
        "title": title,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "legends": legend_candidates,
        "readable_values": readable_values,
        "readable_data_points": readable_data_points,
        "parse_confidence": parse_confidence,
    }


def text_only_ocr_result(text: str, source: str = "mineru_ocr") -> dict[str, Any]:
    lines = [" ".join(line.split()) for line in str(text or "").splitlines() if line.strip()]
    items = [
        {"text": token, "confidence": 1.0, "bbox": None, "region": "text_only", "line_id": f"text:{index}"}
        for index, line in enumerate(lines)
        for token in line.split()
    ]
    return {
        "backend": source,
        "status": "ok" if items else "no_text",
        "error": "",
        "width": 0,
        "height": 0,
        "items": items,
    }


def _pytesseract_ocr(path: str, config: dict[str, Any]) -> dict[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        return _empty_result("pytesseract", f"chart image not found: {path}")
    try:
        import pytesseract
        from PIL import Image, ImageOps
        from pytesseract import Output
    except Exception as exc:
        return _empty_result("pytesseract", f"OCR dependency unavailable: {exc}")
    executable = str(config.get("tesseract_cmd") or "").strip()
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable
    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        return _empty_result("pytesseract", f"tesseract executable unavailable: {exc}")

    try:
        with Image.open(path_obj) as source:
            source.load()
            width, height = source.size
            scale = max(1.0, float(config.get("upscale", 2.0)))
            image = ImageOps.autocontrast(source.convert("L"))
            if scale != 1.0:
                image = image.resize((round(width * scale), round(height * scale)))
            items = _tesseract_items(
                pytesseract,
                Output,
                image,
                scale=scale,
                region="full",
                lang=str(config.get("language") or "eng"),
                psm=int(config.get("psm", 11)),
                min_confidence=float(config.get("min_confidence", 25.0)),
            )
            # Vertical y-axis labels are commonly missed by a full-image pass.
            left_fraction = max(0.1, min(0.35, float(config.get("y_axis_crop_fraction", 0.22))))
            left = image.crop((0, 0, max(1, round(image.width * left_fraction)), image.height))
            rotated = left.rotate(90, expand=True)
            items.extend(
                _tesseract_items(
                    pytesseract,
                    Output,
                    rotated,
                    scale=scale,
                    region="y_axis_rotated",
                    lang=str(config.get("language") or "eng"),
                    psm=11,
                    min_confidence=float(config.get("min_confidence", 25.0)),
                    keep_bbox=False,
                )
            )
    except Exception as exc:
        return _empty_result("pytesseract", f"chart OCR execution failed: {exc}")
    return {
        "backend": "pytesseract",
        "status": "ok" if items else "no_text",
        "error": "",
        "width": width,
        "height": height,
        "items": _dedupe_items(items),
    }


def _tesseract_items(
    pytesseract,
    output,
    image,
    scale: float,
    region: str,
    lang: str,
    psm: int,
    min_confidence: float,
    keep_bbox: bool = True,
) -> list[dict[str, Any]]:
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        config=f"--psm {psm}",
        output_type=output.DICT,
    )
    items = []
    for index, raw_text in enumerate(data.get("text", [])):
        text = " ".join(str(raw_text or "").split())
        try:
            confidence_raw = float(data["conf"][index])
        except (KeyError, TypeError, ValueError):
            confidence_raw = -1.0
        if not text or confidence_raw < min_confidence:
            continue
        bbox = None
        if keep_bbox:
            bbox = [
                round(float(data["left"][index]) / scale, 2),
                round(float(data["top"][index]) / scale, 2),
                round(float(data["width"][index]) / scale, 2),
                round(float(data["height"][index]) / scale, 2),
            ]
        line_id = ":".join(
            str(data.get(key, [0] * (index + 1))[index])
            for key in ("block_num", "par_num", "line_num")
        )
        items.append(
            {
                "text": text,
                "confidence": round(max(0.0, min(1.0, confidence_raw / 100.0)), 4),
                "bbox": bbox,
                "region": region,
                "line_id": f"{region}:{line_id}",
            }
        )
    return items


def _normalize_custom_result(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return text_only_ocr_result(value, source="custom")
    if isinstance(value, list):
        value = {"items": value}
    if not isinstance(value, dict):
        return _empty_result("custom", "custom OCR returned an unsupported value")
    items = []
    for index, item in enumerate(value.get("items", [])):
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict) or not str(item.get("text") or "").strip():
            continue
        items.append(
            {
                "text": " ".join(str(item["text"]).split()),
                "confidence": float(item.get("confidence", 1.0)),
                "bbox": item.get("bbox"),
                "region": item.get("region", "full"),
                "line_id": item.get("line_id", f"custom:{index}"),
            }
        )
    return {
        "backend": str(value.get("backend") or "custom"),
        "status": str(value.get("status") or ("ok" if items else "no_text")),
        "error": str(value.get("error") or ""),
        "width": int(value.get("width") or 0),
        "height": int(value.get("height") or 0),
        "items": items,
    }


def _group_lines(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(items):
        grouped[str(item.get("line_id") or f"item:{index}")].append(item)
    lines = []
    for line_id, tokens in grouped.items():
        tokens.sort(key=lambda item: _bbox_value(item, 0))
        boxes = [item.get("bbox") for item in tokens if _valid_bbox(item.get("bbox"))]
        bbox = _union_bbox(boxes) if boxes else None
        lines.append(
            {
                "text": " ".join(str(item["text"]) for item in tokens),
                "confidence": round(sum(float(item.get("confidence", 0.0)) for item in tokens) / len(tokens), 4),
                "bbox": bbox,
                "region": tokens[0].get("region", "full"),
                "line_id": line_id,
            }
        )
    return sorted(lines, key=lambda item: (_bbox_value(item, 1), _bbox_value(item, 0), item["text"]))


def _axis_label_candidates(lines: list[dict[str, Any]], width: int, height: int, axis: str) -> list[dict[str, Any]]:
    result = []
    for line in lines:
        bbox = line.get("bbox")
        if not _valid_bbox(bbox) or not _has_letters(line.get("text")) or _line_is_values(line.get("text")):
            continue
        cx, cy = _center(bbox)
        if axis == "x" and height and cy >= height * 0.76:
            result.append(_line_record(line))
        elif axis == "y" and width and cx <= width * 0.16:
            result.append(_line_record(line))
    return result


def _legend_candidates(
    lines: list[dict[str, Any]],
    x_labels: list[dict[str, Any]],
    y_labels: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    excluded = {item["text"] for item in x_labels + y_labels}
    candidates = []
    for line in lines:
        text, bbox = str(line.get("text") or ""), line.get("bbox")
        if text in excluded or not _valid_bbox(bbox) or not _has_letters(text) or _line_is_values(text):
            continue
        cx, cy = _center(bbox)
        if width and height and width * 0.16 < cx < width * 0.94 and height * 0.08 < cy < height * 0.65:
            candidates.append(_line_record(line))
    # A legend generally contains multiple nearby labels. Keep isolated lines
    # only when they have explicit series-like markers.
    result = []
    for item in candidates:
        bbox = item.get("bbox")
        peers = [other for other in candidates if other is not item and _nearby_vertically(bbox, other.get("bbox"), height)]
        explicit = bool(re.search(r"\bw/?o\b|\bw/\b|baseline|series|model|all|none|two", item["text"], re.I))
        if peers or explicit:
            result.append(item)
    return _dedupe_records(result)


def _title_candidate(
    lines: list[dict[str, Any]], legends: list[dict[str, Any]], width: int, height: int
) -> str:
    legend_text = {item["text"] for item in legends}
    candidates = []
    for line in lines:
        text, bbox = str(line.get("text") or ""), line.get("bbox")
        if text in legend_text or not _valid_bbox(bbox) or not _has_letters(text) or _line_is_values(text):
            continue
        cx, cy = _center(bbox)
        if height and width and cy <= height * 0.2 and width * 0.2 <= cx <= width * 0.8:
            candidates.append(line)
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (_bbox_value(item, 1), -len(str(item.get("text") or ""))))
    return str(candidates[0]["text"])


def _axis_record(labels: list[dict[str, Any]], ticks: list[dict[str, Any]]) -> dict[str, Any]:
    if not labels and not ticks:
        return {}
    label = max((item["text"] for item in labels), key=len, default="")
    return {
        "label": label,
        "label_candidates": labels,
        "tick_labels": _dedupe_records(ticks),
        "source": "program_ocr",
    }


def _value_record(item: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "text": str(item.get("text") or ""),
        "role": "x_tick" if role == "x_axis" else "y_tick" if role == "y_axis" else role,
        "bbox": item.get("bbox"),
        "confidence": float(item.get("confidence", 0.0)),
        "source": "program_ocr",
    }


def _line_record(line: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": str(line.get("text") or ""),
        "bbox": line.get("bbox"),
        "confidence": float(line.get("confidence", 0.0)),
        "source": "program_ocr",
    }


def _parse_confidence(
    ocr_result: dict[str, Any],
    x_axis: dict[str, Any],
    y_axis: dict[str, Any],
    legends: list[dict[str, Any]],
    values: list[dict[str, Any]],
) -> float:
    if ocr_result.get("status") != "ok":
        return 0.0
    score = 0.25
    score += 0.2 if x_axis else 0.0
    score += 0.2 if y_axis else 0.0
    score += 0.15 if legends else 0.0
    score += 0.2 if values else 0.0
    return round(min(1.0, score), 2)


def _region(item: dict[str, Any], width: int, height: int) -> str:
    bbox = item.get("bbox")
    if not _valid_bbox(bbox) or not width or not height:
        return "unknown"
    cx, cy = _center(bbox)
    # Resolve the bottom-left corner in favor of y ticks only when text is
    # clearly left of the plot margin; the first x tick is usually farther in.
    if cx <= width * 0.1:
        return "y_axis"
    if cy >= height * 0.74:
        return "x_axis"
    if cx <= width * 0.18:
        return "y_axis"
    if width * 0.18 < cx < width * 0.96 and height * 0.08 < cy < height * 0.74:
        return "plot"
    return "unknown"


def _is_value(value: Any) -> bool:
    return VALUE_RE.fullmatch(str(value or "").strip().replace(" ", "")) is not None


def _has_letters(value: Any) -> bool:
    return re.search(r"[A-Za-z\u4e00-\u9fff]", str(value or "")) is not None


def _line_is_values(value: Any) -> bool:
    tokens = str(value or "").split()
    return bool(tokens) and all(_is_value(token) for token in tokens)


def _center(bbox: list[float]) -> tuple[float, float]:
    return float(bbox[0]) + float(bbox[2]) / 2, float(bbox[1]) + float(bbox[3]) / 2


def _valid_bbox(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) >= 4


def _bbox_value(item: dict[str, Any], index: int) -> float:
    bbox = item.get("bbox")
    return float(bbox[index]) if _valid_bbox(bbox) else float("inf")


def _union_bbox(boxes: list[list[float]]) -> list[float]:
    left = min(float(box[0]) for box in boxes)
    top = min(float(box[1]) for box in boxes)
    right = max(float(box[0]) + float(box[2]) for box in boxes)
    bottom = max(float(box[1]) + float(box[3]) for box in boxes)
    return [round(left, 2), round(top, 2), round(right - left, 2), round(bottom - top, 2)]


def _nearby_vertically(left: Any, right: Any, height: int) -> bool:
    if not _valid_bbox(left) or not _valid_bbox(right) or not height:
        return False
    return abs(_center(left)[1] - _center(right)[1]) <= height * 0.18


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        bbox = item.get("bbox")
        key = (item.get("region"), str(item.get("text") or "").lower(), tuple(bbox or []))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _dedupe_records(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    result = []
    for item in items:
        key = (str(item.get("text") or "").lower(), str(item.get("role") or ""))
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _empty_result(backend: str, error: str) -> dict[str, Any]:
    return {"backend": backend, "status": "unavailable", "error": error, "width": 0, "height": 0, "items": []}
