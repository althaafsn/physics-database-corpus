"""Conservative title validation shared by Markdown and AI parsing paths."""
from __future__ import annotations

import re


_DOC_TITLE_RE = re.compile(
    r"^(?:(?:soal|problem\s+set)\s+)?(?:osk|osp|osn)\b.*(?:fisika|physics).*\b20\d{2}\b|"
    r"^(?:soal|problem\s+set)\b.*\b(?:osk|osp|osn|fisika|physics)\b.*\b20\d{2}\b",
    re.IGNORECASE,
)
_CREDIT_RE = re.compile(
    r"\b(?:dimensi\s+sains|ahmad\s+basyir|hak\s+cipta|copyright|instagram|youtube|whatsapp)\b",
    re.IGNORECASE,
)
_GENERIC_RE = re.compile(
    r"^(?:problem|question|soal)\s*(?:no\.?\s*)?\d*$|^(?:part\s+)?[a-z](?:\s*\.\s*\d+)?$",
    re.IGNORECASE,
)


def deterministic_problem_title(
    *, level: str | None, year: int | None, number: int, document_slug: str = ""
) -> str:
    if level and year is not None:
        return f"{level} {year} — Soal {number}"
    if document_slug:
        return f"{document_slug} — Problem {number}"
    return f"Problem {number}"


def choose_problem_title(
    title: str,
    body_md: str,
    *,
    level: str | None,
    year: int | None,
    number: int,
    document_slug: str = "",
) -> str:
    """Keep a plausible source/LLM title, otherwise use a safe fallback."""
    candidate = re.sub(r"\*{1,2}", "", title or "")
    candidate = re.sub(r"\s+", " ", candidate).strip(" :-")
    body = re.sub(r"\s+", " ", body_md or "").strip()
    if (
        not candidate
        or len(candidate) > 160
        or _DOC_TITLE_RE.search(candidate)
        or _CREDIT_RE.search(candidate)
        or _GENERIC_RE.fullmatch(candidate)
        or (len(body) < 80 and _CREDIT_RE.search(body))
    ):
        return deterministic_problem_title(
            level=level,
            year=year,
            number=number,
            document_slug=document_slug,
        )
    return candidate
