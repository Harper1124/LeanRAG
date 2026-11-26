from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .schema import MMChunk, MMMedia


logger = logging.getLogger(__name__)
FIGURE_GROUPING_VERSION = 2
FIGURE_CAPTION = re.compile(r"^\s*(?:figure|fig\.?|chart)\s+([A-Z]?\d+(?:\.\d+)*)\s*[:.．：]", re.I)
CAPTION_START = re.compile(r"^\s*(?:figure|fig\.?|chart|table)\s+[A-Z]?\d+(?:\.\d+)*\s*[:.．：]", re.I)
PANEL_LABEL = re.compile(r"^\s*\(([a-z])\)(?=\s|$)", re.I)
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class FigureCaption:
    number: str
    text: str
    box: Box
    from_text: bool


def group_figures(content_file: Path, chunks: list[MMChunk], media_items: list[MMMedia]) -> None:
    """Resolve figure membership from MinerU caption and body coordinates."""
    media_by_page: dict[int, list[MMMedia]] = defaultdict(list)
    for media in media_items:
        if media.modality in {"chart", "image"}:
            media.original_caption = media.caption
            media.panel_label = _panel_label(media.caption)
            media.figure_group_status = "unresolved"
        if media.page is not None:
            media_by_page[media.page].append(media)

    middle_file = content_file.with_name(content_file.stem.split("_content_list", 1)[0] + "_middle.json")
    pages = _read_layout(middle_file)
    chunks_by_page: dict[int, list[MMChunk]] = defaultdict(list)
    for chunk in chunks:
        if chunk.page_start is not None:
            chunks_by_page[chunk.page_start].append(chunk)

    for page, media in media_by_page.items():
        visuals = [item for item in media if item.modality in {"chart", "image"}]
        if not visuals:
            continue
        layout = pages.get(page, {})
        blocks = list(_leaf_blocks(layout.get("para_blocks") or []))
        bodies, captions, labels, barriers = _page_geometry(blocks)
        panels = {
            item.media_id: box for item in visuals
            if len(boxes := bodies.get(_asset_name(item.path), [])) == 1
            for box in boxes
        }
        page_width, page_height = layout.get("page_size") or (0, 0)
        max_caption_gap = max(12.0, 0.055 * page_height)

        matches: dict[str, list[tuple[int, float, FigureCaption]]] = defaultdict(list)
        for caption in captions:
            for media_id, rank in _caption_members(caption, panels, barriers, max_caption_gap).items():
                matches[media_id].append((rank, _coverage(caption.box, panels[media_id]), caption))

        # content_list preserves the formatted text (including inline math) when
        # it contains exactly one version of the caption on this page.
        raw_captions: dict[str, set[str]] = defaultdict(set)
        for item in media:
            for number, text in _numbered_captions(item.caption):
                raw_captions[number].add(text)
        layout_counts = Counter(caption.number for caption in captions)
        positioned_labels = {label for label, _ in labels}
        panel_labels = _match_panel_labels(labels, panels)

        for item in visuals:
            label_matches = panel_labels.get(item.media_id, set())
            if len(label_matches) == 1:
                item.panel_label = next(iter(label_matches))
            elif item.media_id in panels and item.panel_label in positioned_labels:
                item.panel_label = None

            choices = matches[item.media_id]
            choices.sort(key=lambda choice: (choice[0], -choice[1]))
            if choices:
                rank, coverage, caption = choices[0]
                tied = len(choices) > 1 and choices[1][0] == rank and coverage - choices[1][1] < 0.15
                if not tied and layout_counts[caption.number] == 1:
                    _assign(item, caption.number, _caption_text(caption, raw_captions))
            else:
                own = list(_numbered_captions(item.caption))
                if len(own) == 1 and own[0][0] not in layout_counts:
                    # With no positioned caption, the original explicit number
                    # can identify this crop, but gives no evidence for its peers.
                    _assign(item, *own[0])
                    item.figure_group_status = "caption_only"

            if not item.figure_id and captions:
                body = panels.get(item.media_id)
                if not body and item.bbox and page_width and page_height:
                    scale = (page_width / 1000, page_height / 1000) * 2
                    body = _box([value * factor for value, factor in zip(item.bbox, scale)])
                if body:
                    _attach_candidate(item, body, captions, raw_captions)

        text_captions = {
            caption.number: caption for caption in captions
            if caption.from_text and any(
                item.figure_id == _figure_id(item, caption.number) for item in visuals
            )
        }
        caption_boxes = set()
        for chunk in chunks_by_page[page]:
            match = FIGURE_CAPTION.match(chunk.text)
            caption = text_captions.get(match[1].lower()) if match else None
            if caption and _same_caption_block(chunk, caption, page_width, page_height):
                chunk.is_figure_caption = True
                if chunk.bbox:
                    caption_boxes.add(tuple(chunk.bbox))
        # Split pieces of the same content_list block retain its box.
        for chunk in chunks_by_page[page]:
            if chunk.bbox and tuple(chunk.bbox) in caption_boxes:
                chunk.is_figure_caption = True


def _read_layout(path: Path) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream)
        return {
            int(page.get("page_idx", idx)) + 1: page
            for idx, page in enumerate(data.get("pdf_info", []))
        }
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        logger.warning("Skipping Figure layout from %s: %s", path, exc)
        return {}


def _leaf_blocks(blocks: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for block in blocks:
        if block.get("blocks"):
            yield from _leaf_blocks(block["blocks"])
        else:
            yield block


def _page_geometry(
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, list[Box]], list[FigureCaption], list[tuple[str, Box]], list[Box]]:
    bodies: dict[str, list[Box]] = defaultdict(list)
    captions: list[FigureCaption] = []
    labels: list[tuple[str, Box]] = []
    barriers: list[Box] = []
    for block in blocks:
        box = _box(block.get("bbox"))
        if box is None:
            continue
        kind = block.get("type") or ""
        lines = block.get("lines") or []
        if kind in {"chart_body", "image_body"}:
            for line in lines:
                for span in line.get("spans", []):
                    if span.get("image_path"):
                        crop_boxes = bodies[_asset_name(span["image_path"])]
                        crop_box = _box(span.get("bbox")) or box
                        if crop_box not in crop_boxes:
                            crop_boxes.append(crop_box)
        elif kind.endswith("_caption") or kind in {"text", "title"}:
            records = [
                (" ".join(span.get("content") or "" for span in line.get("spans", [])), _box(line.get("bbox")) or box)
                for line in lines
            ]
            starts = [idx for idx, (text, _) in enumerate(records) if CAPTION_START.match(text)]
            for start, end in zip(starts, starts[1:] + [len(records)]):
                match = FIGURE_CAPTION.match(records[start][0])
                if match and (kind.endswith("_caption") or start == 0):
                    text = " ".join(line.strip() for line, _ in records[start:end]).strip()
                    bounds = box if len(starts) == 1 and start == 0 else _union([b for _, b in records[start:end]])
                    caption = FigureCaption(match[1].lower(), text, bounds, kind in {"text", "title"})
                    if caption not in captions:
                        captions.append(caption)
                    barriers.append(bounds)
            first_line = records[0][0] if records else ""
            if kind in {"chart_caption", "image_caption"} and (label := _panel_label(first_line)):
                labels.append((label, records[0][1]))
            if kind in {"text", "title"} or kind == "table_caption":
                barriers.append(box)
            elif starts and not FIGURE_CAPTION.match(records[starts[0]][0]):
                barriers.append(box)
        elif kind == "table_body":
            barriers.append(box)
    return bodies, captions, labels, barriers


def _match_panel_labels(labels: list[tuple[str, Box]], panels: dict[str, Box]) -> dict[str, set[str]]:
    matches: dict[str, set[str]] = defaultdict(set)
    for label, box in labels:
        distances = sorted(
            (abs(box[1] - body[3]), media_id)
            for media_id, body in panels.items()
            if -4 <= box[1] - body[3] <= max(12, _height(body) * 0.2) and _coverage(body, box) >= 0.6
        )
        if distances and (len(distances) == 1 or distances[1][0] - distances[0][0] > 2):
            matches[distances[0][1]].add(label)
    return matches


def _same_caption_block(chunk: MMChunk, caption: FigureCaption, width: float, height: float) -> bool:
    if not chunk.bbox or not width or not height:
        return caption.text.startswith(" ".join(chunk.text.split()))
    # content_list coordinates are normalized to 0–1000. middle.json boxes
    # use the page's own units.
    scale = (width / 1000, height / 1000) * 2
    return all(abs(value * factor - raw) <= 2 for value, factor, raw in zip(chunk.bbox, scale, caption.box))


def _caption_members(
    caption: FigureCaption, panels: dict[str, Box], barriers: list[Box], max_gap: float
) -> dict[str, int]:
    box = caption.box
    remaining = {
        key: body for key, body in panels.items()
        if body[3] <= box[1] + 2 and _under_caption(box, body)
    }
    members: dict[str, int] = {}
    lower_row: list[Box] = []
    rank = 0
    while remaining:
        if lower_row:
            near = {
                key: body for key, body in remaining.items()
                if any(
                    -2 <= lower[1] - body[3] <= min(_height(body), _height(lower)) * 0.45
                    and _coverage(lower, body) >= 0.5
                    and not _blocked(body, lower[1], barriers)
                    for lower in lower_row
                )
            }
        else:
            near = {
                key: body for key, body in remaining.items()
                if box[1] - body[3] <= max_gap and not _blocked(body, box[1], barriers)
            }
        if not near:
            break
        closest = max(near.values(), key=lambda body: body[3])
        row = {key: body for key, body in near.items() if _same_row(closest, body)}
        members.update(dict.fromkeys(row, rank))
        lower_row = list(row.values())
        for key in row:
            remaining.pop(key)
        rank += 1
    # MinerU can retain the enclosing illustration and separate crops cut out
    # of that image. A contained crop can inherit the enclosing Figure identity.
    for key, body in panels.items():
        if key not in members and any(_contains(panels[member], body) for member in members):
            members[key] = rank
    return members


def _blocked(body: Box, bottom: float, barriers: list[Box]) -> bool:
    return any(
        obstacle[1] >= body[3] - 1 and obstacle[3] <= bottom
        and _coverage(obstacle, body) >= 0.4
        for obstacle in barriers
    )


def _box(value: Any) -> Box | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = map(float, value)
    except (TypeError, ValueError):
        return None
    return (left, top, right, bottom) if left < right and top < bottom else None


def _union(boxes: list[Box]) -> Box:
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def _height(box: Box) -> float:
    return box[3] - box[1]


def _coverage(container: Box, body: Box) -> float:
    overlap = min(container[2], body[2]) - max(container[0], body[0])
    return max(0.0, overlap) / (body[2] - body[0])


def _under_caption(caption: Box, body: Box) -> bool:
    if _coverage(caption, body) >= 0.55:
        return True
    # A centered caption can be noticeably narrower than its single panel.
    return (
        _coverage(body, caption) >= 0.95
        and abs((caption[0] + caption[2]) - (body[0] + body[2])) <= (body[2] - body[0]) * 0.3
    )


def _contains(outer: Box, inner: Box) -> bool:
    return (
        outer != inner and outer[0] <= inner[0] and outer[1] <= inner[1]
        and outer[2] >= inner[2] and outer[3] >= inner[3]
    )


def _same_row(first: Box, second: Box) -> bool:
    overlap = min(first[3], second[3]) - max(first[1], second[1])
    return overlap >= min(_height(first), _height(second)) * 0.5


def _asset_name(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _numbered_captions(text: str) -> Iterator[tuple[str, str]]:
    parts = re.split(r"\n(?=\s*(?:figure|fig\.?|chart|table)\s+)", text, flags=re.I)
    for part in parts:
        if match := FIGURE_CAPTION.match(part):
            yield match[1].lower(), part.strip()


def _panel_label(text: str) -> str | None:
    match = PANEL_LABEL.match(text)
    return match[1].lower() if match else None


def _figure_id(media: MMMedia, number: str) -> str:
    return f"{media.doc_id}::figure::{number}"


def _assign(media: MMMedia, number: str, text: str) -> None:
    media.figure_id = _figure_id(media, number)
    media.caption = text
    media.figure_group_status = "resolved"


def _caption_text(caption: FigureCaption, raw_captions: dict[str, set[str]]) -> str:
    versions = raw_captions.get(caption.number, set())
    return next(iter(versions)) if len(versions) == 1 and not caption.from_text else caption.text


def _attach_candidate(
    media: MMMedia, body: Box, captions: list[FigureCaption], raw_captions: dict[str, set[str]]
) -> None:
    ranked = sorted(
        ((_candidate_distance(body, caption.box), caption) for caption in captions),
        key=lambda item: (*item[0], item[1].box, item[1].number),
    )
    (priority, distance), first = ranked[0]
    media.candidate_figure_id = _figure_id(media, first.number)
    media.candidate_caption = _caption_text(first, raw_captions)
    media.candidate_alternatives = [
        {"figure_id": _figure_id(media, caption.number), "caption": _caption_text(caption, raw_captions)}
        for (other_priority, other_distance), caption in ranked[1:]
        if other_priority == priority and other_distance - distance <= max(3, _height(body) * 0.05)
    ]


def _candidate_distance(body: Box, caption: Box) -> tuple[int, float]:
    below = caption[1] >= body[3] - 2
    overlap = _coverage(caption, body) >= 0.35 or _coverage(body, caption) >= 0.35
    horizontal_gap = max(0.0, body[0] - caption[2], caption[0] - body[2])
    vertical_gap = max(0.0, body[1] - caption[3], caption[1] - body[3])
    # Keep a same-column caption ahead of one across a column gutter.
    priority = (0 if overlap else 2) + (0 if below else 1)
    return priority, horizontal_gap + vertical_gap
