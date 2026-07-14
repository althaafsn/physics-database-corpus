"""Decide whether a PDF in all_pdf/solutions/ is an actual worked solution or a
mislabeled problem-statement duplicate (confirmed: e.g. osk-fisika-2011.pdf is
just the problem set, no solution content, despite living in the solutions
folder)."""
from __future__ import annotations

import re
from dataclasses import dataclass

FILENAME_SOLUTION_HINTS = ("solusi", "pembahasan", "kunci-jawaban")

# Strong markers: only ever appear when a problem is actually being *worked*,
# not when it's merely being posed. Kept deliberately narrow - generic
# derivation connectives like "sehingga"/"diperoleh" also show up inside
# problem *statements* (describing a scenario) and are not reliable signals.
STRONG_SOLUTION_MARKERS = [
    r"\bpembahasan\b",
    r"\bpenyelesaian\s*:",
    r"\bsolusi\b",
    r"\bkunci jawaban\b",
    r"\(nilai\s*\d+\)",
    r"\bjawab(?:an)?\s*:",
]
# Administrative/exam-paper boilerplate that only shows up on a raw problem
# set, never inside a worked-solution transcript.
EXAM_PAPER_MARKERS = [
    r"\bpetunjuk\s+tes\b",
    r"\bnomor peserta\b",
    r"\blembar jawaban\b",
    r"waktu\s*:\s*\d",
    r"\bseleksi tim\b",
]

_STRONG_RE = re.compile("|".join(STRONG_SOLUTION_MARKERS), re.IGNORECASE)
_EXAM_RE = re.compile("|".join(EXAM_PAPER_MARKERS), re.IGNORECASE)


@dataclass(frozen=True)
class DocTypeResult:
    is_solution: bool
    solution_score: float
    problem_score: float
    reason: str


def classify_doc_type(filename: str, text: str) -> DocTypeResult:
    """Heuristic classification; no LLM call needed for ~70 documents given
    how distinct worked-solution prose is from problem-statement prose.

    Decision rule: count explicit "this is being solved" markers
    (pembahasan/penyelesaian/jawab:/nilai-points) against explicit exam-paper
    boilerplate markers (petunjuk tes/nomor peserta/waktu: N jam). A real
    worked-solution transcript repeats the former once per problem; a raw
    problem set has none of the former and usually has at least one of the
    latter on its cover page.
    """
    if not text or not text.strip():
        return DocTypeResult(False, 0.0, 0.0, "no extractable text (likely scanned/handwritten)")

    lower_name = filename.lower()
    filename_says_solution = any(hint in lower_name for hint in FILENAME_SOLUTION_HINTS)

    strong_hits = len(_STRONG_RE.findall(text))
    exam_hits = len(_EXAM_RE.findall(text))

    if strong_hits >= 2:
        return DocTypeResult(True, float(strong_hits), float(exam_hits), "repeated solution markers found")

    if filename_says_solution and strong_hits >= 1:
        return DocTypeResult(True, float(strong_hits), float(exam_hits), "filename + at least one solution marker")

    if filename_says_solution and exam_hits == 0:
        # Filename claims solution, no exam boilerplate contradicts it, but
        # markers are sparse (e.g. Marker OCR missed "Pembahasan:" labels) -
        # still accept, split_solution_markdown() will independently gate on
        # whether numbered segments are actually found.
        return DocTypeResult(True, float(strong_hits), float(exam_hits), "filename indicates solution, no contradiction")

    return DocTypeResult(
        False,
        float(strong_hits),
        float(exam_hits),
        "no repeated solution markers - looks like a bare problem set",
    )
