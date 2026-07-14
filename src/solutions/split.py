"""Split a worked-solution markdown document into per-problem segments.

Solution documents number problems more loosely than the main corpus (e.g.
``1- (nilai 10) ...``, ``## **1- Jawab:**``, ``OSK Fisika 2014 Number 1``,
``2. (10 poin) ...``), so this is a dedicated, more permissive splitter rather
than reusing src/split_problems.py's problem-statement-oriented regexes.
"""
from __future__ import annotations

import re

# Matches a new-problem marker at the start of a line: "1- ", "1. ", "1) ",
# "No 1", "No. 1", "Soal 1", "Soal nomor 1" - all commonly seen across the 71
# solution PDFs from different authors/years.
_NUMBERED_LINE_RE = re.compile(
    r"^(?:no\.?\s*|soal\s*(?:nomor\s*)?)?(?P<num>\d{1,2})[.\-]\s+(?!\))",
    re.IGNORECASE | re.MULTILINE,
)

# "OSK Fisika 2014 Number 1", "OSP Fisika 2015 Number 7", etc.
_NUMBER_HEADING_RE = re.compile(
    r"^.*?\bNumber\s+(?P<num>\d{1,2})\b",
    re.IGNORECASE | re.MULTILINE,
)

# "Soal Nomor 1: Title", "Soal nomor 2 - ..."
_SOAL_NOMOR_RE = re.compile(
    r"^Soal\s+Nomor\s+(?P<num>\d{1,2})\s*[:\-]",
    re.IGNORECASE | re.MULTILINE,
)

# Standalone "Nomor 1" / "Nomor 2" section headers (OSN teori 2024 style).
_NOMOR_LINE_RE = re.compile(
    r"^Nomor\s+(?P<num>\d{1,2})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# "1. Title" at line start (OSP 2019 Dimensi Sains style) — title must follow.
_PLAIN_TITLE_RE = re.compile(
    r"^(?P<num>\d{1,2})\.\s+(?P<title>[A-Za-zÀ-ÿ].{8,})$",
    re.MULTILINE,
)


def _normalize_solution_markdown(md_text: str) -> str:
    """Strip Marker heading/bold/list noise so problem numbers sit at line starts."""
    lines: list[str] = []
    for raw in md_text.splitlines():
        line = raw.strip()
        line = re.sub(r"^#+\s*", "", line)
        line = line.replace("**", "")
        # List item before a problem number: "- 3- Jawab:"
        line = re.sub(r"^[-*+]\s+", "", line)
        lines.append(line)
    return "\n".join(lines)


def _collect_markers(normalized: str) -> list[tuple[int, int, int]]:
    """Return [(start_pos, end_pos, problem_number), ...] sorted by position."""
    seen_starts: set[int] = set()
    markers: list[tuple[int, int, int]] = []

    for pattern in (
        _NUMBERED_LINE_RE,
        _NUMBER_HEADING_RE,
        _SOAL_NOMOR_RE,
        _NOMOR_LINE_RE,
        _PLAIN_TITLE_RE,
    ):
        for match in pattern.finditer(normalized):
            start = match.start()
            if start in seen_starts:
                continue
            line = normalized[start : normalized.find("\n", start)]
            if re.match(r"^\d{1,2}-\d{1,2}\s", line):
                continue
            seen_starts.add(start)
            markers.append((start, match.end(), int(match.group("num"))))

    markers.sort(key=lambda item: item[0])
    return markers


def _accept_markers(markers: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Keep a monotonically increasing number sequence (gaps allowed)."""
    if len(markers) < 2:
        return []

    # Drop obvious false positives: same number repeated at nearby positions.
    deduped: list[tuple[int, int, int]] = []
    for marker in markers:
        if deduped and marker[2] == deduped[-1][2]:
            continue
        deduped.append(marker)

    accepted: list[tuple[int, int, int]] = [deduped[0]]
    for marker in deduped[1:]:
        if marker[2] > accepted[-1][2]:
            accepted.append(marker)
    return accepted if len(accepted) >= 2 else []


def split_solution_markdown(md_text: str) -> list[tuple[int, str]]:
    """Return [(problem_number, body_md), ...] in document order."""
    normalized = _normalize_solution_markdown(md_text)
    accepted = _accept_markers(_collect_markers(normalized))
    if not accepted:
        return []

    segments: list[tuple[int, str]] = []

    first_num = accepted[0][2]
    if first_num > 1:
        body_end = accepted[0][0]
        body = normalized[:body_end].strip()
        if body:
            segments.append((1, body))

    for i, (_start, body_start, num) in enumerate(accepted):
        body_end = accepted[i + 1][0] if i + 1 < len(accepted) else len(normalized)
        body = normalized[body_start:body_end].strip()
        if body:
            segments.append((num, body))

    return segments
