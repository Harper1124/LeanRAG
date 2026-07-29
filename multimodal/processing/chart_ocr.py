from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any


# Numeric tokens may carry an inequality/approximation prefix and a compact
# unit suffix (for example 25k, 41.5%, or 7B). Unicode escapes keep this source
# stable across Linux and Windows checkouts.
VALUE_RE = re.compile(r"^[<>~\u2248\u2264\u2265]?[-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?(?:[kKmMbB%])?$")


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

    x_tokens = _remove_value_fragments([item for item in items if _region(item, width, height) == "x_axis"])
    y_tokens = _rightmost_value_per_line(
        _remove_value_fragments([item for item in items if _region(item, width, height) == "y_axis"])
    )
    x_ticks = [_value_record(item, "x_tick") for item in x_tokens if _is_value(item.get("text"))]
    x_ticks.extend(
        _value_record(item, "x_tick")
        for item in items
        if item.get("region") == "x_axis_rotated" and _is_value(item.get("text"))
    )
    y_ticks = [_value_record(item, "y_tick") for item in y_tokens if _is_value(item.get("text"))]

    x_labels = _axis_label_candidates(full_lines, width, height, axis="x")
    # Horizontal text at the left edge is normally a category tick, not a
    # vertical-axis title. Only rotated OCR (or explicit MinerU axis text)
    # is trusted as a y-axis label.
    y_labels: list[dict[str, Any]] = []
    y_category_lines = _categorical_y_lines(full_lines, width, height)
    y_category_text = {item["text"] for item in y_category_lines}
    x_labels = [item for item in x_labels if item["text"] not in y_category_text]
    x_labels = [item for item in x_labels if not _looks_like_legend_label(item["text"])]
    x_category_lines = _categorical_axis_lines(x_labels, axis="x", width=width, height=height)
    if len(x_category_lines) >= 2:
        category_text = {item["text"] for item in x_category_lines}
        x_labels = [item for item in x_labels if item["text"] not in category_text]
        x_ticks.extend(_text_tick_record(item, "x_tick") for item in x_category_lines)
    if len(y_category_lines) >= 2:
        y_ticks.extend(_text_tick_record(item, "y_tick") for item in y_category_lines)
    for line in text_lines:
        text = str(line.get("text") or "")
        if re.search(r"\bx[- ]?axis\b|horizontal axis", text, re.I):
            x_labels.append(_line_record(line))
        if re.search(r"\by[- ]?axis\b|vertical axis", text, re.I):
            y_labels.append(_line_record(line))
    for line in rotated_y:
        text = str(line.get("text") or "").strip()
        if (
            _has_letters(text)
            and not _is_value(text)
            and float(line.get("confidence", 0.0)) >= 0.6
            and len(re.sub(r"[^A-Za-z]", "", text)) >= 3
        ):
            y_labels.append(_line_record(line))
    x_labels = _dedupe_records(x_labels)
    y_labels = _dedupe_records(y_labels)

    legend_candidates = _legend_candidates(
        full_lines,
        x_labels + x_category_lines,
        y_labels + y_category_lines,
        width,
        height,
    )
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

    legend_line_ids = {
        line.get("line_id")
        for line in full_lines
        if any(_clean_label(str(line.get("text") or "")) == legend.get("text") for legend in legend_candidates)
    }
    readable_values = []
    readable_data_points = []
    for item in items:
        if item.get("region") not in {"full", "plot_crop", "x_axis_rotated", "text_only"} or not _is_value(item.get("text")):
            continue
        if item.get("line_id") in legend_line_ids or _inside_any_bbox(
            item.get("bbox"), [legend.get("bbox") for legend in legend_candidates]
        ):
            continue
        role = "x_axis" if item.get("region") == "x_axis_rotated" else _region(item, width, height)
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

    x_axis = _axis_record(x_labels, x_ticks, axis="x")
    y_axis = _axis_record(y_labels, y_ticks, axis="y")
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
            # Sparse-page OCR often misses black labels printed inside coloured
            # bars. A thresholded plot-area pass isolates that text while
            # retaining coordinates in the original image.
            plot_box = (
                round(image.width * float(config.get("plot_left_fraction", 0.10))),
                round(image.height * float(config.get("plot_top_fraction", 0.04))),
                round(image.width * float(config.get("plot_right_fraction", 0.99))),
                round(image.height * float(config.get("plot_bottom_fraction", 0.90))),
            )
            plot = image.crop(plot_box)
            threshold = int(config.get("plot_text_threshold", 125))
            plot_binary = plot.point(lambda value: 255 if value > threshold else 0)
            items.extend(
                _tesseract_items(
                    pytesseract,
                    Output,
                    plot_binary,
                    scale=scale,
                    region="plot_crop",
                    lang=str(config.get("language") or "eng"),
                    psm=int(config.get("plot_psm", 11)),
                    min_confidence=float(config.get("min_confidence", 25.0)),
                    offset=(plot_box[0], plot_box[1]),
                )
            )
            # If the regular pass found no bottom-axis values, retry the
            # bottom strip after deskewing the common 45-degree tick labels.
            bottom_numeric = [
                item
                for item in items
                if item.get("region") == "full"
                and _is_value(item.get("text"))
                and _valid_bbox(item.get("bbox"))
                and _center(item["bbox"])[1] >= height * 0.72
            ]
            if len(bottom_numeric) < 2 and bool(config.get("detect_angled_x_ticks", True)):
                bottom_top = round(image.height * float(config.get("x_tick_crop_top_fraction", 0.70)))
                bottom = image.crop((0, bottom_top, image.width, image.height))
                angled = bottom.rotate(float(config.get("x_tick_rotation_degrees", 45.0)), expand=True)
                items.extend(
                    _tesseract_items(
                        pytesseract,
                        Output,
                        angled,
                        scale=scale,
                        region="x_axis_rotated",
                        lang=str(config.get("language") or "eng"),
                        psm=11,
                        min_confidence=float(config.get("min_confidence", 25.0)),
                        keep_bbox=False,
                    )
                )
            # Vertical y-axis labels are commonly missed by a full-image pass.
            left_fraction = max(0.1, min(0.35, float(config.get("y_axis_crop_fraction", 0.22))))
            left = image.crop((0, 0, max(1, round(image.width * left_fraction)), image.height))
            # Matplotlib-style y labels read bottom-to-top, so a clockwise
            # rotation makes them horizontal for Tesseract.
            rotated = left.rotate(-90, expand=True)
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
    offset: tuple[float, float] = (0.0, 0.0),
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
                round((float(data["left"][index]) + offset[0]) / scale, 2),
                round((float(data["top"][index]) + offset[1]) / scale, 2),
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
    top_candidates = []
    for line in lines:
        text, bbox = str(line.get("text") or ""), line.get("bbox")
        if (
            _clean_label(text) in excluded
            or not _valid_bbox(bbox)
            or not _has_letters(text)
            or (_line_is_values(text) and not _looks_like_series_name(text))
        ):
            continue
        if re.fullmatch(r"\s*(?:legend|agreement)\s*", text, re.I):
            continue
        cx, cy = _center(bbox)
        explicit = bool(re.search(r"\bw/?o\b|\bw/\b|baseline|series|model|all|none|two|wins?|ties?|los(?:e|es)", text, re.I))
        if width and height and width * 0.12 < cx < width * 0.98 and height * 0.08 < cy < height * 0.84:
            candidates.append(_legend_record(line))
        elif width and height and cy <= height * 0.12 and width * 0.12 < cx < width * 0.98:
            top_candidates.append(_legend_record(line))
        elif explicit and width and height and cx > width * 0.55 and cy < height * 0.9:
            candidates.append(_legend_record(line))
    # Multiple labels aligned across the top are normally a legend. A lone
    # top line remains eligible to be the chart title instead.
    if len(top_candidates) >= 2:
        candidates.extend(top_candidates)
    # A legend generally contains multiple nearby labels. Keep isolated lines
    # only when they have explicit series-like markers.
    result = []
    for item in candidates:
        bbox = item.get("bbox")
        peers = [other for other in candidates if other is not item and _nearby_vertically(bbox, other.get("bbox"), height)]
        explicit = bool(re.search(r"\bw/?o\b|\bw/\b|baseline|series|model|all|none|two|wins?|ties?|los(?:e|es)", item["text"], re.I))
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
        if (
            _clean_label(text) in legend_text
            or not _valid_bbox(bbox)
            or not _has_letters(text)
            or _line_is_values(text)
            or len(re.sub(r"[^A-Za-z]", "", text)) < 3
            or float(line.get("confidence", 0.0)) < 0.55
        ):
            continue
        cx, cy = _center(bbox)
        if height and width and cy <= height * 0.2 and width * 0.2 <= cx <= width * 0.8:
            candidates.append(line)
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (_bbox_value(item, 1), -len(str(item.get("text") or ""))))
    return str(candidates[0]["text"])


def _axis_record(
    labels: list[dict[str, Any]], ticks: list[dict[str, Any]], axis: str
) -> dict[str, Any]:
    if not labels and not ticks:
        return {}
    if axis == "x" and labels:
        label = max(labels, key=lambda item: _center(item["bbox"])[1] if _valid_bbox(item.get("bbox")) else 0)["text"]
    else:
        label = max((item["text"] for item in labels), key=len, default="")
    return {
        "label": label,
        "label_candidates": labels,
        "tick_labels": _dedupe_records(ticks),
        "source": "program_ocr",
    }


def _categorical_axis_lines(
    labels: list[dict[str, Any]], axis: str, width: int, height: int
) -> list[dict[str, Any]]:
    del axis, width
    candidates = [
        item
        for item in labels
        if not re.fullmatch(r"[oO0]\s*[kK]", str(item.get("text") or ""))
        and _valid_bbox(item.get("bbox"))
    ]
    if len(candidates) < 2 or not height:
        return []
    groups: list[list[dict[str, Any]]] = []
    for item in candidates:
        cy = _center(item["bbox"])[1]
        group = next(
            (group for group in groups if abs(_center(group[0]["bbox"])[1] - cy) <= height * 0.03),
            None,
        )
        if group is None:
            groups.append([item])
        else:
            group.append(item)
    return max(groups, key=len) if groups and len(max(groups, key=len)) >= 2 else []


def _categorical_y_lines(
    lines: list[dict[str, Any]], width: int, height: int
) -> list[dict[str, Any]]:
    if not width or not height:
        return []
    candidates = []
    for line in lines:
        bbox = line.get("bbox")
        text = str(line.get("text") or "")
        if not _valid_bbox(bbox) or not _has_letters(text) or _line_is_values(text):
            continue
        cx, cy = _center(bbox)
        if cx <= width * 0.2 and height * 0.06 < cy < height * 0.88:
            candidates.append(_line_record(line))
    return candidates


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
        "text": _clean_label(str(line.get("text") or "")),
        "bbox": line.get("bbox"),
        "confidence": float(line.get("confidence", 0.0)),
        "source": "program_ocr",
    }


def _legend_record(line: dict[str, Any]) -> dict[str, Any]:
    record = _line_record(line)
    match = re.search(r"\b(All|Two|None|Wins?|Ties?|Loses?)\b\s*$", record["text"], re.I)
    if match:
        record["text"] = match.group(1)
    return record


def _text_tick_record(item: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "text": str(item.get("text") or ""),
        "role": role,
        "bbox": item.get("bbox"),
        "confidence": float(item.get("confidence", 0.0)),
        "source": "program_ocr",
    }


def _clean_label(text: str) -> str:
    return re.sub(r"^[^A-Za-z0-9\u4e00-\u9fff]+\s*", "", text).strip()


def _looks_like_legend_label(text: str) -> bool:
    return re.search(r"\b(?:All|Two|None|Wins?|Ties?|Loses?)\b\s*$", str(text or ""), re.I) is not None


def _looks_like_series_name(text: str) -> bool:
    return re.fullmatch(r"\d+(?:\.\d+)?[bBmM]", str(text or "").strip()) is not None


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
    if cx <= width * 0.13:
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


def _remove_value_fragments(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        text = str(item.get("text") or "")
        is_fragment = any(
            other is not item
            and item.get("line_id") == other.get("line_id")
            and text != str(other.get("text") or "")
            and text in str(other.get("text") or "")
            and _is_value(other.get("text"))
            and _boxes_touch(item.get("bbox"), other.get("bbox"))
            for other in items
        )
        if not is_fragment:
            result.append(item)
    return result


def _boxes_touch(left: Any, right: Any) -> bool:
    if not _valid_bbox(left) or not _valid_bbox(right):
        return False
    l_left, l_top, l_width, l_height = map(float, left[:4])
    r_left, r_top, r_width, r_height = map(float, right[:4])
    horizontal_gap = max(r_left - (l_left + l_width), l_left - (r_left + r_width), 0.0)
    vertical_overlap = min(l_top + l_height, r_top + r_height) - max(l_top, r_top)
    return horizontal_gap <= 2.0 and vertical_overlap >= 0


def _rightmost_value_per_line(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("line_id") or id(item))].append(item)
    result = []
    for group in grouped.values():
        values = [item for item in group if _is_value(item.get("text"))]
        non_values = [item for item in group if not _is_value(item.get("text"))]
        if values:
            result.append(max(values, key=lambda item: _center(item["bbox"])[0] if _valid_bbox(item.get("bbox")) else 0))
        result.extend(non_values)
    return result


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


def _inside_any_bbox(bbox: Any, containers: list[Any]) -> bool:
    if not _valid_bbox(bbox):
        return False
    cx, cy = _center(bbox)
    for container in containers:
        if not _valid_bbox(container):
            continue
        left, top, width, height = map(float, container[:4])
        if left <= cx <= left + width and top <= cy <= top + height:
            return True
    return False


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        bbox = item.get("bbox")
        text = str(item.get("text") or "").lower()
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(result)
                if str(existing.get("text") or "").lower() == text
                and _same_position(existing.get("bbox"), bbox)
            ),
            None,
        )
        if duplicate_index is None:
            result.append(item)
        elif float(item.get("confidence", 0.0)) > float(result[duplicate_index].get("confidence", 0.0)):
            result[duplicate_index] = item
    return result


def _same_position(left: Any, right: Any) -> bool:
    if not _valid_bbox(left) or not _valid_bbox(right):
        return False
    lcx, lcy = _center(left)
    rcx, rcy = _center(right)
    tolerance = max(3.0, min(float(left[2]), float(left[3]), float(right[2]), float(right[3])) * 0.35)
    return abs(lcx - rcx) <= tolerance and abs(lcy - rcy) <= tolerance


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
