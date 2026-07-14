"""Small, dependency-free view of Marker JSON layout provenance."""
from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Iterable


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_DOCUMENT_HEADING_RE = re.compile(
    r"^(?:(?:soal|problem\s+set)\s+)?(?:osk|osp|osn)\b.*(?:fisika|physics).*\b20\d{2}\b|"
    r"^(?:soal|problem\s+set)\b.*\b(?:osk|osp|osn|fisika|physics)\b.*\b20\d{2}\b",
    re.IGNORECASE,
)
_PROMO_HEADING_RE = re.compile(
    r"\b(?:f z t i|dimensi\s+sains|program\s+persiapan|unduh\s+buku|pendaftaran)\b",
    re.IGNORECASE,
)
_MARKDOWN_HEADING_RE = re.compile(
    r"^#{2,6}\s+\*{0,2}(.+?)\*{0,2}\s*$", re.MULTILINE
)
_IMAGE_REF_RE = re.compile(r"!\[\]\(([^)]+)\)")


@dataclass(frozen=True)
class LayoutSection:
    """A Marker heading and the blocks belonging to it."""

    heading_id: str
    title: str
    page_start: int | None
    page_end: int | None
    block_ids: tuple[str, ...]
    image_block_ids: tuple[str, ...]
    image_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Block:
    block_id: str
    block_type: str
    page: int | None
    bbox: tuple[float, float, float, float] | None
    title: str
    section_ids: tuple[str, ...]


def _text_from_html(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = html.unescape(_TAG_RE.sub(" ", value))
    return _SPACE_RE.sub(" ", text).strip()


def _page_number(block_id: str) -> int | None:
    match = re.search(r"/page/(\d+)(?:/|$)", block_id)
    return int(match.group(1)) if match else None


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        value = value.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return tuple(float(part) for part in value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _page_nodes(document: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for node in document.get("children") or []:
        if not isinstance(node, dict):
            continue
        if node.get("block_type") == "Page":
            yield node
        else:
            yield from _page_nodes(node)


def _blocks_for_page(page: dict[str, Any]) -> list[_Block]:
    blocks: list[_Block] = []
    for raw in page.get("children") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        hierarchy = raw.get("section_hierarchy") or {}
        section_ids = tuple(str(item) for item in hierarchy.values() if item)
        blocks.append(
            _Block(
                block_id=str(raw["id"]),
                block_type=str(raw.get("block_type") or ""),
                page=_page_number(str(raw["id"])),
                bbox=_bbox(raw.get("bbox")),
                title=_text_from_html(raw.get("html")),
                section_ids=section_ids,
            )
        )
    return blocks


def _is_problem_heading(block: _Block) -> bool:
    if block.block_type.lower() not in {"sectionheader", "heading"}:
        return False
    title = block.title
    if not title or _DOCUMENT_HEADING_RE.search(title) or _PROMO_HEADING_RE.search(title):
        return False
    return True


def extract_problem_sections(document: dict[str, Any]) -> list[LayoutSection]:
    """Return ordered problem headings with their Marker-owned image blocks.

    Marker normally carries the owning heading in ``section_hierarchy`` even
    when a problem continues onto another page. If that metadata is absent,
    the ordered heading fallback keeps the block assigned to the most recent
    accepted heading.
    """
    ordered: list[_Block] = []
    for page in _page_nodes(document):
        ordered.extend(_blocks_for_page(page))

    headings = [block for block in ordered if _is_problem_heading(block)]
    if not headings:
        return []
    accepted_ids = {heading.block_id for heading in headings}
    grouped: dict[str, list[_Block]] = {heading.block_id: [] for heading in headings}
    current: str | None = None
    for block in ordered:
        matching = [sid for sid in block.section_ids if sid in accepted_ids]
        if matching:
            current = matching[-1]
        elif block.block_id in accepted_ids:
            current = block.block_id
        if current is not None:
            grouped[current].append(block)

    sections: list[LayoutSection] = []
    for heading in headings:
        blocks = grouped[heading.block_id]
        pages = [block.page for block in blocks if block.page is not None]
        images = tuple(
            block.block_id
            for block in blocks
            if block.block_type.lower() in {"picture", "figure"}
        )
        sections.append(
            LayoutSection(
                heading_id=heading.block_id,
                title=heading.title,
                page_start=min(pages) if pages else heading.page,
                page_end=max(pages) if pages else heading.page,
                block_ids=tuple(block.block_id for block in blocks),
                image_block_ids=images,
            )
        )
    return sections


def _markdown_sections(markdown: str) -> list[tuple[str, str, tuple[str, ...]]]:
    matches = list(_MARKDOWN_HEADING_RE.finditer(markdown))
    out: list[tuple[str, str, tuple[str, ...]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[match.end() : end].strip()
        title = _SPACE_RE.sub(" ", html.unescape(match.group(1))).strip()
        out.append((title, body, tuple(_IMAGE_REF_RE.findall(body))))
    return out


def _debug_blocks(debug_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for page in debug_data:
        blocks.extend(page.get("children") or page.get("current_children") or [])
    return [block for block in blocks if isinstance(block, dict)]


def _debug_is_heading(block: dict[str, Any]) -> bool:
    return str(block.get("block_type") or "").lower() in {"21", "sectionheader", "heading"}


def _debug_is_image(block: dict[str, Any]) -> bool:
    return str(block.get("block_type") or "").lower() in {
        "11",
        "20",
        "picture",
        "figure",
    }


def extract_debug_sections(
    debug_data: list[dict[str, Any]], markdown: str
) -> list[LayoutSection]:
    """Recover image ownership from Marker ``--debug_json`` blocks.

    Debug JSON does not repeat rendered heading text, so its reading-order
    headings are paired with Markdown headings. This is deliberately a
    conservative fallback: if the two streams have different heading counts,
    no layout sections are returned rather than guessing image ownership.
    """
    md_sections = _markdown_sections(markdown)
    raw_blocks = _debug_blocks(debug_data)
    heading_positions = [i for i, block in enumerate(raw_blocks) if _debug_is_heading(block)]
    if len(heading_positions) != len(md_sections):
        return []

    sections: list[LayoutSection] = []
    for index, (title, _body, refs) in enumerate(md_sections):
        if _DOCUMENT_HEADING_RE.search(title) or _PROMO_HEADING_RE.search(title):
            continue
        start = heading_positions[index]
        end = heading_positions[index + 1] if index + 1 < len(heading_positions) else len(raw_blocks)
        owned = raw_blocks[start:end]
        pages = [int(block["page_id"]) for block in owned if str(block.get("page_id", "")).isdigit()]
        image_blocks = [block for block in owned if _debug_is_image(block)]
        sections.append(
            LayoutSection(
                heading_id=str(raw_blocks[start].get("block_id", index)),
                title=title,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                block_ids=tuple(str(block.get("block_id")) for block in owned),
                image_block_ids=tuple(str(block.get("block_id")) for block in image_blocks),
                image_refs=refs,
            )
        )
    return sections
