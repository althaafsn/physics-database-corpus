"""Deterministic figure binding: Marker per-problem refs, not LLM caption matching."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.bronze.hybrid_bronze import _attach_marker_images, _infer_year
from src.bronze.marker_layout import extract_debug_sections
from src.text.attach_images import extract_image_refs
from src.text.split_problems import split_markdown_auto_with_variants

_IMAGE_LINE_RE = re.compile(r"^[ \t]*!\[[^\]]*\]\([^)]+\)[ \t]*$", re.MULTILINE)


def strip_image_refs_from_body(body_md: str) -> str:
    """Remove all image references before deterministic Marker reattachment."""
    body = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", body_md)
    kept = [line for line in body.splitlines() if not _IMAGE_LINE_RE.match(line)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def marker_images_by_problem(marker_md: str, *, slug: str) -> dict[int, list[str]]:
    year = _infer_year(slug)
    out: dict[int, list[str]] = {}
    for number, _title, body, _variant in split_markdown_auto_with_variants(marker_md, year=year):
        refs = extract_image_refs(body)
        if refs:
            out[number] = refs
    return out


def rebind_problem_images(
    problems: list[dict[str, Any]],
    marker_md: str,
    *,
    slug: str,
    layout_debug_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Replace LLM-assigned ``image_refs`` with Marker refs from the same problem number."""
    year = _infer_year(slug)
    layout_refs: list[tuple[str, ...]] | None = None
    if layout_debug_path is not None and layout_debug_path.is_file():
        try:
            debug_data = json.loads(layout_debug_path.read_text(encoding="utf-8"))
            sections = extract_debug_sections(debug_data, marker_md)
            if len(sections) == len(problems):
                layout_refs = [section.image_refs for section in sections]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            layout_refs = None
    marker_by_key = {
        (variant, number): body
        for number, _title, body, variant in split_markdown_auto_with_variants(marker_md, year=year)
    }
    marker_by_num = {}
    for (variant, number), body in marker_by_key.items():
        marker_by_num.setdefault(number, []).append((variant, body))

    rebound: list[dict[str, Any]] = []
    for index, item in enumerate(problems):
        number = int(item["number"])
        variant = item.get("variant")
        marker_body = marker_by_key.get((variant, number), "")
        if not marker_body:
            candidates = marker_by_num.get(number, [])
            if len(candidates) == 1:
                marker_body = candidates[0][1]
        refs = list(layout_refs[index]) if layout_refs is not None else extract_image_refs(marker_body)
        body = strip_image_refs_from_body(str(item.get("body_md") or ""))
        if layout_refs is not None:
            for ref in refs:
                if ref not in extract_image_refs(body):
                    body = f"{body}\n\n![]({ref})" if body else f"![]({ref})"
        elif marker_body.strip():
            body = _attach_marker_images(body, marker_body)
        rebound.append(
            {
                **item,
                "body_md": body,
                "image_refs": refs,
            }
        )
    return rebound


def rebind_structured_markdown(
    structured_md: str,
    marker_md: str,
    *,
    slug: str,
    layout_debug_path: Path | None = None,
) -> str:
    """Rebind images on an existing ``## **N. …**`` bronze document."""
    from src.bronze.doc_structure import problems_to_markdown

    year = _infer_year(slug)
    problems: list[dict[str, Any]] = []
    for number, title, body, variant in split_markdown_auto_with_variants(structured_md, year=year):
        problems.append(
            {
                "number": number,
                "title": title,
                "body_md": body,
                "image_refs": extract_image_refs(body),
                "variant": variant,
            }
        )
    if not problems:
        return structured_md

    rebound = rebind_problem_images(
        problems,
        marker_md,
        slug=slug,
        layout_debug_path=layout_debug_path,
    )
    md = problems_to_markdown(rebound)
    return md if md.endswith("\n") else md + "\n"
