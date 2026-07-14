"""Deterministic prep before document-level LLM fuse/structure.

Strips promo, WhatsApp, watermarks, and junk image refs from Marker markdown
and pdftotext separately. Does **not** fuse the two parsers.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.repair.repair_images import is_definite_watermark
from src.text.attach_images import extract_image_refs
from src.text.clean import clean_text

_WHATSAPP_BLOCK_RE = re.compile(
    r"(?m)^[ \t]*(?:Join Grup WA[^\n]*|Grup WA Komunitas[^\n]*|(?:https?://)?chat\.whatsapp\.com/\S+)[ \t]*\n?",
    re.IGNORECASE,
)
_IMAGE_LINE_RE = re.compile(r"^[ \t]*!\[[^\]]*\]\(([^)]+)\)[ \t]*$", re.MULTILINE)


def strip_promo_and_ads(text: str) -> str:
    """Remove publisher ads / WhatsApp / footers without merging parsers."""
    if not text or not text.strip():
        return text
    cleaned = clean_text(text)
    cleaned = _WHATSAPP_BLOCK_RE.sub("", cleaned)
    return cleaned.strip()


def drop_junk_image_refs(text: str, bronze_folder: Path | None = None) -> str:
    """Drop watermark / tiny junk ``![](...)`` lines; keep real diagram refs."""

    def keep(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        name = Path(ref).name
        size = 0
        if bronze_folder is not None:
            candidate = bronze_folder / name
            if candidate.is_file():
                size = candidate.stat().st_size
        if is_definite_watermark(name, size):
            return ""
        return match.group(0)

    return _IMAGE_LINE_RE.sub(keep, text)


def prep_marker_markdown(marker_md: str, bronze_folder: Path | None = None) -> str:
    text = strip_promo_and_ads(marker_md)
    return drop_junk_image_refs(text, bronze_folder)


def prep_pdftotext(pdf_text: str) -> str:
    return strip_promo_and_ads(pdf_text)


def list_caption_candidate_images(bronze_folder: Path, fused_or_marker_md: str) -> list[Path]:
    """Image files referenced in markdown that are not definite watermarks."""
    refs = extract_image_refs(fused_or_marker_md)
    out: list[Path] = []
    seen: set[str] = set()
    for ref in refs:
        name = Path(ref).name
        if name in seen:
            continue
        seen.add(name)
        path = bronze_folder / name
        if not path.is_file():
            continue
        if is_definite_watermark(name, path.stat().st_size):
            continue
        out.append(path)
    return out
