"""Merge pdftotext bodies with Marker markdown (images + headings)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.text.attach_images import extract_image_refs
from src.bronze.pdf_text import extract_pdf_text, has_usable_text_layer, text_layer_stats
from src.text.split_problems import (
    split_markdown_auto_with_variants,
    split_pdf_text_problem_variants,
)

_HEADING_RE = re.compile(
    r"^#{1,6}\s+\*{0,2}(\d+\.\s+.+?)\*{0,2}\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class HybridBronzeInfo:
    text_source: str
    pdf_problem_coverage: float
    pdf_stats: dict[str, int]
    problem_count: int


def hybrid_bronze_enabled() -> bool:
    import os

    return os.environ.get("PHYSICS_HYBRID_BRONZE", "1").lower() in {"1", "true", "yes"}


def _infer_year(slug: str) -> int | None:
    match = re.search(r"(20\d{2})", slug)
    return int(match.group(1)) if match else None


def _marker_preamble(marker_md: str) -> str:
    match = re.search(
        r"^#{1,6}\s+\*{0,2}(?:Versi|Version)\s+\d+\*{0,2}\s*$",
        marker_md,
        re.MULTILINE | re.IGNORECASE,
    )
    if match:
        return marker_md[: match.start()].strip()
    match = _HEADING_RE.search(marker_md)
    if not match:
        return ""
    return marker_md[: match.start()].strip()


def _attach_marker_images(pdf_body: str, marker_body: str) -> str:
    refs = extract_image_refs(marker_body)
    if not refs:
        return pdf_body
    body = pdf_body.rstrip()
    existing = set(extract_image_refs(body))
    for ref in refs:
        if ref not in existing:
            body = f"{body}\n\n![]({ref})"
    return body


def merge_hybrid_markdown(
    pdf_path: Path,
    marker_md: str,
    *,
    slug: str | None = None,
) -> tuple[str, HybridBronzeInfo | None]:
    """Replace Marker problem bodies with pdftotext when the PDF text layer is usable."""
    slug = slug or pdf_path.stem
    year = _infer_year(slug)

    if not hybrid_bronze_enabled() or not has_usable_text_layer(pdf_path):
        return marker_md, None

    pdf_text = extract_pdf_text(pdf_path)
    pdf_problems = split_pdf_text_problem_variants(pdf_text, year=year, slug=slug)
    marker_problems = split_markdown_auto_with_variants(marker_md, year=year)
    if not pdf_problems:
        return marker_md, None

    marker_by_key = {
        (variant, number): (title, body)
        for number, title, body, variant in marker_problems
    }
    pdf_by_key = {
        (variant, number): (title, body)
        for number, title, body, variant in pdf_problems
    }
    pdf_by_number: dict[int, list[tuple[int | None, str, str]]] = {}
    for number, title, body, variant in pdf_problems:
        pdf_by_number.setdefault(number, []).append((variant, title, body))
    ordered_keys: list[tuple[int | None, int]] = []
    for number, _title, _body, variant in marker_problems + pdf_problems:
        key = (variant, number)
        if key not in ordered_keys:
            ordered_keys.append(key)
    if not ordered_keys:
        return marker_md, None

    def pdf_for(variant: int | None, number: int) -> tuple[int | None, str, str] | None:
        exact = pdf_by_key.get((variant, number))
        if exact is not None:
            return variant, exact[0], exact[1]
        candidates = pdf_by_number.get(number, [])
        return candidates[0] if len(candidates) == 1 else None

    matched = sum(1 for variant, number in ordered_keys if pdf_for(variant, number) is not None)
    coverage = matched / len(ordered_keys)

    merged_blocks: list[str] = []
    current_variant: int | None = None
    for variant, number in ordered_keys:
        marker_title, marker_body = marker_by_key.get(
            (variant, number), (f"Soal {number}", "")
        )
        pdf_segment = pdf_for(variant, number)
        if pdf_segment is not None:
            _pdf_variant, pdf_title, pdf_body = pdf_segment
            title = marker_title if marker_title != f"Soal {number}" else pdf_title
            body = _attach_marker_images(pdf_body, marker_body)
        else:
            title, body = marker_title, marker_body
        if not body.strip():
            continue
        if variant is not None and variant != current_variant:
            merged_blocks.append(f"## **Versi {variant}**")
            current_variant = variant
        merged_blocks.append(f"## **{number}. {title}**\n\n{body.strip()}")

    if coverage < 0.25:
        return marker_md, None

    preamble = _marker_preamble(marker_md)
    parts = [preamble] if preamble else []
    parts.append("\n\n".join(merged_blocks))
    merged_md = "\n\n".join(part for part in parts if part).strip() + "\n"

    info = HybridBronzeInfo(
        text_source="hybrid_pdftotext",
        pdf_problem_coverage=round(coverage, 4),
        pdf_stats=text_layer_stats(pdf_text),
        problem_count=len(merged_blocks),
    )
    return merged_md, info


def write_hybrid_bronze_metadata(folder: Path, slug: str, info: HybridBronzeInfo) -> None:
    path = folder / f"{slug}_bronze_source.json"
    path.write_text(
        json.dumps(
            {
                "text_source": info.text_source,
                "pdf_problem_coverage": info.pdf_problem_coverage,
                "pdf_stats": info.pdf_stats,
                "problem_count": info.problem_count,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def apply_hybrid_bronze_to_folder(pdf_path: Path, bronze_folder: Path) -> HybridBronzeInfo | None:
    """Post-process an existing Marker bronze folder with pdftotext bodies."""
    slug = bronze_folder.name
    md_path = bronze_folder / f"{slug}.md"
    if not md_path.is_file():
        return None

    marker_md = md_path.read_text(encoding="utf-8")
    merged, info = merge_hybrid_markdown(pdf_path, marker_md, slug=slug)
    if info is None:
        return None

    md_path.write_text(merged, encoding="utf-8")
    write_hybrid_bronze_metadata(bronze_folder, slug, info)
    return info
