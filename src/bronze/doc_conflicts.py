"""Resolve Marker ↔ pdftotext aligner conflict blocks in student-facing text."""
from __future__ import annotations

import re
from pathlib import Path

from src.text.attach_images import extract_image_refs

_JUNK_MARKER_RE = re.compile(r"^[\*#_\s]+$")
_CONFLICT_BLOCK_RE = re.compile(
    r"<<<MARKER\s*(.*?)\s*===\s*(.*?)\s*PDFTEXT>>>",
    re.DOTALL,
)


def is_junk_marker_span(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    without_images = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", stripped).strip()
    if not without_images:
        return False
    return bool(_JUNK_MARKER_RE.fullmatch(without_images))


def resolve_align_conflicts(text: str) -> str:
    """Strip ``<<<MARKER … === … PDFTEXT>>>`` blocks from student-facing text."""
    def repl(match: re.Match[str]) -> str:
        marker_part = match.group(1).strip()
        pdf_part = match.group(2).strip()
        marker_images = extract_image_refs(marker_part)

        if pdf_part:
            body = pdf_part
        elif marker_part and not is_junk_marker_span(marker_part):
            body = marker_part
        else:
            body = ""

        existing = set(extract_image_refs(body))
        for ref in marker_images:
            name = Path(ref).name
            if name not in existing:
                body = f"{body}\n\n![]({name})" if body else f"![]({name})"
                existing.add(name)
        return body

    cleaned = _CONFLICT_BLOCK_RE.sub(repl, text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
