"""Heuristic document language detection for physics exam PDFs."""
from __future__ import annotations

import re
from typing import Literal

ContentLocale = Literal["id", "en"]

_ID_MARKERS = (
    "soal",
    "jawaban",
    "benar",
    "pilihan",
    "berikut",
    "tentukan",
    "dengan",
    "adalah",
    "gerak",
    "fisika",
    "pernyataan",
    "semua",
    "kecuali",
    "yang",
    "dan",
    "dari",
    "pada",
    "oleh",
    "sebuah",
    "hukum",
)

_EN_MARKERS = (
    "problem",
    "following",
    "determine",
    "which",
    "correct",
    "answer",
    "physics",
    "calculate",
    "given",
    "choose",
    "statement",
    "statements",
    "except",
    "particle",
    "velocity",
    "acceleration",
    "force",
    "energy",
    "momentum",
    "estimate",
    "suppose",
    "consider",
    "neglecting",
    "write",
    "expression",
    "magnitude",
    "direction",
)

# Indonesian-only; "massa" removed (false match risk). Use function words instead.
_ID_FUNCTION_WORDS = ("yang", "dan", "dari", "pada", "adalah", "dengan", "untuk", "atau", "jika")

_EN_FILENAME_RE = re.compile(
    r"(?:usapho|ipho|apho|bpho|usaco|f=ma|physics\s*olympiad|"
    r"international\s*physics|traveling\s*team|aapt|british\s*physics)",
    re.IGNORECASE,
)

_ID_FILENAME_RE = re.compile(
    r"(?:soal|osp|osk|osn|pembahasan|semifinal|dimensi\s*sains)",
    re.IGNORECASE,
)

_EN_HEADING_RE = re.compile(
    r"(?m)^\s*(?:problem|question|part)\s+[a-z0-9]+",
    re.IGNORECASE,
)

_ID_HEADING_RE = re.compile(
    r"(?m)^\s*(?:soal|nomor)\s+(?:nomor\s+)?\d+",
    re.IGNORECASE,
)


def _word_hits(text: str, words: tuple[str, ...]) -> int:
    return sum(1 for word in words if re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE))


def _score_text(sample: str) -> tuple[int, int]:
    lowered = sample.lower()
    id_score = _word_hits(lowered, _ID_MARKERS)
    id_score += len(re.findall(rf"\b({'|'.join(_ID_FUNCTION_WORDS)})\b", lowered))
    en_score = _word_hits(lowered, _EN_MARKERS)
    if _EN_HEADING_RE.search(sample):
        en_score += 3
    if _ID_HEADING_RE.search(sample):
        id_score += 3
    return id_score, en_score


def _sample_spread(text: str, *, max_chunk: int = 6000) -> str:
    """Cover pages mislead; sample start, middle, and end."""
    text = (text or "").strip()
    if len(text) <= max_chunk * 2:
        return text[: max_chunk * 3]
    third = len(text) // 3
    parts = [
        text[:max_chunk],
        text[third : third + max_chunk],
        text[-max_chunk:],
    ]
    return "\n\n".join(parts)


def detect_content_locale(
    text: str,
    *,
    hint: str | None = None,
    slug: str | None = None,
    filename: str | None = None,
) -> ContentLocale:
    """Return ``id`` or ``en`` from exam-like text and optional filename hints."""
    if hint in ("id", "en"):
        return hint  # type: ignore[return-value]

    name_hint = f"{slug or ''} {filename or ''}"
    if _EN_FILENAME_RE.search(name_hint):
        return "en"
    if _ID_FILENAME_RE.search(name_hint):
        return "id"

    sample = _sample_spread(text)
    if not sample.strip():
        return "id"

    id_score, en_score = _score_text(sample)

    if en_score > id_score + 1:
        return "en"
    if id_score > en_score + 1:
        return "id"

    if re.search(r"\bproblem\s+\d+\b", sample, re.IGNORECASE):
        return "en"
    if re.search(r"\bsoal\s+(nomor\s+)?\d+\b", sample, re.IGNORECASE):
        return "id"

    # Latin-heavy exam with no Indonesian function words → English.
    if id_score == 0 and en_score >= 2:
        return "en"

    return "id"
