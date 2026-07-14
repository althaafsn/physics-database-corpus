"""Deterministic Marker ↔ pdftotext aligner (no LLM).

Where the two parsers agree, emit text once (prefer pdftotext wording).
Where they diverge, emit a tagged conflict block until they re-sync:

    <<<MARKER
    ...marker-only / divergent span...
    ===
    ...pdftotext-only / divergent span...
    PDFTEXT>>>

Marker ``![](...)`` image refs are always preserved.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from src.bronze.doc_conflicts import is_junk_marker_span
from src.bronze.doc_prep import prep_marker_markdown, prep_pdftotext
from src.bronze.pdf_text import extract_pdf_text, has_usable_text_layer
from src.text.attach_images import extract_image_refs

_TOKEN_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)]+)\)"
    r"|\$\$[\s\S]*?\$\$"
    r"|\$[^$\n]+\$"
    r"|[A-Za-z0-9]+(?:'[A-Za-z]+)?"
    r"|[^\s]",
    re.MULTILINE,
)

_WS_RE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]


def _norm_token(tok: str) -> str:
    if tok.startswith("!["):
        return tok
    if tok.startswith("$"):
        return _WS_RE.sub("", tok)
    return tok.casefold()


def _join_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    out: list[str] = []
    for tok in tokens:
        if not out:
            out.append(tok)
            continue
        prev = out[-1]
        if tok in {",", ".", ";", ":", "!", "?", ")", "]", "}", "%"}:
            out.append(tok)
        elif prev in {"(", "[", "{"}:
            out.append(tok)
        elif tok.startswith("!["):
            out.append("\n\n" + tok)
        else:
            out.append(" " + tok)
    return "".join(out).strip()


def _is_image_token(tok: str) -> bool:
    return tok.startswith("![") and "](" in tok


def align_marker_and_pdftotext(marker_md: str, pdf_text: str) -> str:
    """Produce an aligned document with conflict blocks where parsers disagree."""
    marker_tokens = _tokenize(marker_md)
    pdf_tokens = _tokenize(pdf_text)
    marker_norm = [_norm_token(t) for t in marker_tokens]
    pdf_norm = [_norm_token(t) for t in pdf_tokens]

    matcher = SequenceMatcher(a=marker_norm, b=pdf_norm, autojunk=False)
    parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        m_span = marker_tokens[i1:i2]
        p_span = pdf_tokens[j1:j2]

        if tag == "equal":
            rebuilt: list[str] = []
            for mt, pt in zip(m_span, p_span, strict=False):
                rebuilt.append(mt if _is_image_token(mt) else pt)
            if len(m_span) > len(p_span):
                for mt in m_span[len(p_span) :]:
                    if _is_image_token(mt):
                        rebuilt.append(mt)
            text = _join_tokens(rebuilt)
            if text:
                parts.append(text)
            continue

        if tag == "delete":
            images = [t for t in m_span if _is_image_token(t)]
            other = [t for t in m_span if not _is_image_token(t)]
            other_text = _join_tokens(other)
            if other_text and not is_junk_marker_span(other_text):
                parts.append("<<<MARKER\n" + other_text + "\n===\n\nPDFTEXT>>>")
            parts.extend(images)
            continue

        if tag == "insert":
            text = _join_tokens(p_span)
            if text:
                parts.append("<<<MARKER\n\n===\n" + text + "\nPDFTEXT>>>")
            continue

        # replace
        if m_span and all(_is_image_token(t) for t in m_span):
            p_text = _join_tokens(p_span)
            if p_text:
                parts.append(p_text)
            parts.extend(m_span)
            continue
        m_text = _join_tokens(m_span)
        p_text = _join_tokens(p_span)
        if is_junk_marker_span(m_text) and p_text:
            parts.append(p_text)
        elif is_junk_marker_span(p_text) and m_text and not is_junk_marker_span(m_text):
            parts.append(m_text)
        elif is_junk_marker_span(m_text) and is_junk_marker_span(p_text):
            pass
        else:
            parts.append(
                "<<<MARKER\n"
                + m_text
                + "\n===\n"
                + p_text
                + "\nPDFTEXT>>>"
            )

    aligned = "\n\n".join(p for p in parts if p and str(p).strip()).strip() + "\n"

    for ref in extract_image_refs(marker_md):
        name = Path(ref).name
        if f"![]({name})" not in aligned and f"]({ref})" not in aligned:
            aligned = aligned.rstrip() + f"\n\n![]({name})\n"
    return aligned


def align_document(
    pdf_path: Path,
    marker_md: str,
    *,
    bronze_folder: Path | None = None,
) -> str | None:
    """Prep both parsers and align. Returns aligned markdown or None if no text layer."""
    if not has_usable_text_layer(pdf_path):
        return None
    prepared_marker = prep_marker_markdown(marker_md, bronze_folder)
    prepared_pdf = prep_pdftotext(extract_pdf_text(pdf_path))
    if not prepared_pdf.strip():
        return None
    return align_marker_and_pdftotext(prepared_marker, prepared_pdf)
