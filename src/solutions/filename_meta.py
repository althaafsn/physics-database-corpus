"""Parse level/year/variant hints from all_pdf/solutions/*.pdf filenames.

Solution filenames use a different, looser convention than the main corpus
(e.g. ``osk-fisika-2012-tipe-a-solusi.pdf``, ``pembahasan-osn-fisika-2020.pdf``,
``solusi-osk-fisika-sma-2024 (1).pdf``) so this is a dedicated, more permissive
parser rather than reusing src/parse_filename.py's strict "Soal OSK Fisika SMA
<year>.pdf" patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LEVEL_YEAR_RE = re.compile(r"(?P<level>osk|osp|osn)[^\d]{0,40}(?P<year>20\d{2})", re.IGNORECASE)
YEAR_ONLY_RE = re.compile(r"(20\d{2})")
VARIANT_TIPE_RE = re.compile(r"tipe[\s_-]*([a-c])", re.IGNORECASE)
ROUND_HINTS: dict[str, list[str]] = {
    "final": ["final"],
    "semifinal": ["semifinal"],
    "kabupaten": ["kabupaten", "kota"],
    "provinsi": ["provinsi", "propinsi"],
    "eksperimen": ["eksperimen"],
    "teori": ["teori"],
}


@dataclass(frozen=True)
class SolutionDocMeta:
    slug: str
    level: str | None
    year: int | None
    variant_hint: str | None  # e.g. "a" / "b" / "c" from "tipe-a"
    round_hint: str | None
    is_handwriting: bool


def parse_solution_filename(name: str) -> SolutionDocMeta:
    stem = name.rsplit(".", 1)[0]
    lower = stem.lower()

    level: str | None = None
    year: int | None = None
    match = LEVEL_YEAR_RE.search(lower)
    if match:
        level = match.group("level").upper()
        year = int(match.group("year"))
    else:
        year_match = YEAR_ONLY_RE.search(lower)
        if year_match:
            year = int(year_match.group(1))
        for lvl in ("osk", "osp", "osn"):
            if lvl in lower:
                level = lvl.upper()
                break

    variant_match = VARIANT_TIPE_RE.search(lower)
    variant_hint = variant_match.group(1).lower() if variant_match else None

    round_hint: str | None = None
    for name_hint, keywords in ROUND_HINTS.items():
        if any(kw in lower for kw in keywords):
            round_hint = name_hint
            break

    is_handwriting = "handwriting" in lower

    slug = re.sub(r"\s+", "-", stem.strip())
    slug = re.sub(r"[^a-zA-Z0-9\-]", "", slug)

    return SolutionDocMeta(
        slug=slug,
        level=level,
        year=year,
        variant_hint=variant_hint,
        round_hint=round_hint,
        is_handwriting=is_handwriting,
    )
