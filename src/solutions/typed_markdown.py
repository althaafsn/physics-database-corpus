"""pdftotext-first markdown resolution for typed solution PDFs.

Typed exam solution PDFs (Dimensi Sains style) carry faithful math in the
embedded text layer. Marker is only needed for diagram refs — not for body text.
"""
from __future__ import annotations

import os
from pathlib import Path

from src.text.attach_images import extract_image_refs
from src.bronze.bronze_convert import convert_pdf_to_bronze, marker_extra_args_from_env
from src.bronze.pdf_text import extract_pdf_text, has_usable_text_layer, strip_pdf_footers


def merge_pdftotext_with_marker_images(pdf_text: str, marker_md: str) -> str:
    """Keep pdftotext body; append Marker image refs missing from the text layer."""
    refs = extract_image_refs(marker_md)
    if not refs:
        return pdf_text
    body = pdf_text.rstrip()
    existing = set(extract_image_refs(body))
    extras = [f"![]({ref})" for ref in refs if ref not in existing]
    if not extras:
        return body
    return body + "\n\n" + "\n\n".join(extras)


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def resolve_typed_solution_markdown(
    pdf_path: Path,
    bronze_dir: Path,
    *,
    force: bool,
    force_marker: bool = False,
) -> tuple[str, str, str | None]:
    """Return ``(markdown, method, error)``.

    method is one of: ``pdftotext``, ``hybrid_pdftotext``, ``marker``, ``failed``.
    """
    slug = pdf_path.stem
    md_path = bronze_dir / slug / f"{slug}.md"
    min_words = int(os.environ.get("PHYSICS_SOLUTION_TEXT_MIN_WORDS", "40"))

    pdf_text = ""
    if has_usable_text_layer(pdf_path, min_words=min_words, min_math_italic=0):
        pdf_text = strip_pdf_footers(extract_pdf_text(pdf_path))

    skip_marker = (
        pdf_text.strip()
        and not force_marker
        and not _env_truthy("PHYSICS_SOLUTIONS_MARKER_FOR_IMAGES", "0")
    )

    if skip_marker:
        if md_path.is_file():
            marker_md = md_path.read_text(encoding="utf-8")
            merged = merge_pdftotext_with_marker_images(pdf_text, marker_md)
            method = "hybrid_pdftotext" if merged != pdf_text else "pdftotext"
            return merged, method, None
        return pdf_text, "pdftotext", None

    if force or not md_path.is_file():
        result = convert_pdf_to_bronze(
            pdf_path,
            bronze_dir=bronze_dir,
            marker_extra_args=marker_extra_args_from_env(),
            timeout_s=float(os.environ.get("MARKER_TIMEOUT_S", "5400")),
        )
        if not result.ok:
            if pdf_text.strip():
                return pdf_text, "pdftotext", None
            return "", "failed", result.detail

    if md_path.is_file():
        marker_md = md_path.read_text(encoding="utf-8")
        if pdf_text.strip():
            merged = merge_pdftotext_with_marker_images(pdf_text, marker_md)
            return merged, "hybrid_pdftotext", None
        return marker_md, "marker", None

    if pdf_text.strip():
        return pdf_text, "pdftotext", None
    return "", "failed", "no markdown available"
