"""Extract text from PDFs via Poppler pdftotext (typed exam documents)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

# Mathematical Alphanumeric Symbols (italic Latin + Greek in exam PDFs).
MATH_ITALIC_RE = re.compile(r"[\U0001D400-\U0001D7FF]")
_WORD_RE = re.compile(r"[a-zA-Zà-ÿ]{3,}")

# Page-header blocks after form-feed (Dimensi Sains letterhead on each page).
_PDF_FOOTER_BLOCK_RE = re.compile(
    r"(?:"
    r"\f\s*Dimensi Sains[\s\-–—]*(?:Ahmad Basyir Najwan\s*)?"
    r"(?:www\.basyiralbanjari\.wordpress\.com[^\n]*\n)?"
    r"(?:[^\n]*(?:Youtube|Tiktok|Instagram|WA|0852|0896)[^\n]*\n)?"
    r"|Dimensi Sains\s*-\s*Bersama Sainskan Indonesia\s*"
    r")",
    re.IGNORECASE,
)


def strip_pdf_footers(text: str) -> str:
    """Remove publisher footers/watermarks from pdftotext output."""
    if not text:
        return text
    from src.text.clean import FOOTER_INLINE_RE, FOOTER_LINE_RE, INLINE_FOOTER_RE

    text = _PDF_FOOTER_BLOCK_RE.sub("", text)
    text = FOOTER_LINE_RE.sub("", text)
    text = INLINE_FOOTER_RE.sub("", text)
    text = FOOTER_INLINE_RE.sub("", text)
    text = re.sub(r"\f+", "\n", text)
    return text


def pdftotext_available() -> bool:
    return shutil.which("pdftotext") is not None


def extract_pdf_text(pdf_path: Path, *, layout: bool = False) -> str:
    """Return the PDF embedded text layer, or \"\" when unavailable."""
    if not pdf_path.is_file():
        return ""
    if not pdftotext_available():
        return ""
    cmd = ["pdftotext"]
    if layout:
        cmd.append("-layout")
    cmd.extend([str(pdf_path.resolve()), "-"])
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def text_layer_stats(text: str) -> dict[str, int]:
    words = _WORD_RE.findall(text)
    return {
        "words": len(words),
        "math_italic": len(MATH_ITALIC_RE.findall(text)),
        "chars": len(text.strip()),
    }


def has_usable_text_layer(
    pdf_path: Path,
    *,
    min_words: int | None = None,
    min_math_italic: int | None = None,
) -> bool:
    """Heuristic: typed physics PDFs have prose + math-italic variables in the text layer."""
    import os

    if min_words is None:
        min_words = int(os.environ.get("PHYSICS_TEXT_MIN_WORDS", "80"))
    if min_math_italic is None:
        min_math_italic = int(os.environ.get("PHYSICS_TEXT_MIN_MATH_ITALIC", "3"))

    text = extract_pdf_text(pdf_path)
    if not text.strip():
        return False
    stats = text_layer_stats(text)
    return stats["words"] >= min_words and stats["math_italic"] >= min_math_italic


def has_extractable_text_layer(
    pdf_path: Path,
    *,
    min_words: int | None = None,
    min_chars: int | None = None,
) -> bool:
    """Looser gate for uploads: English prose or unicode math counts as extractable."""
    import os

    if min_words is None:
        min_words = int(os.environ.get("PHYSICS_EXTRACT_MIN_WORDS", "50"))
    if min_chars is None:
        min_chars = int(os.environ.get("PHYSICS_EXTRACT_MIN_CHARS", "120"))

    text = strip_pdf_footers(extract_pdf_text(pdf_path))
    if len(text.strip()) < min_chars:
        return False
    stats = text_layer_stats(text)
    if stats["words"] >= min_words:
        return True
    return stats["math_italic"] >= int(os.environ.get("PHYSICS_TEXT_MIN_MATH_ITALIC", "3")) and stats[
        "words"
    ] >= int(os.environ.get("PHYSICS_TEXT_MIN_WORDS", "80"))
