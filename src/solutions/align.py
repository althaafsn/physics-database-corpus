"""Align a solved problem_number from a solution document to a gold
problem_id, via level + year (+ variant/round hints), matching the
make_problem_id() scheme in src/pipeline.py. Ambiguous/low-confidence
alignments are flagged rather than silently guessed."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from src.schema import ProblemRecord

_VARIANT_LETTER_TO_INT = {"a": 1, "b": 2, "c": 3}
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
_STOPWORDS = frozenset(
    {
        "dan",
        "atau",
        "yang",
        "dengan",
        "untuk",
        "dari",
        "pada",
        "adalah",
        "dalam",
        "jika",
        "agar",
        "serta",
        "oleh",
        "akan",
        "dapat",
        "jawab",
        "soal",
        "nilai",
        "poin",
        "tentukan",
        "hitung",
        "nyatakan",
    }
)


@dataclass(frozen=True)
class AlignResult:
    problem_id: str | None
    method: str  # "exact" | "ambiguous" | "no_match"
    confidence: float
    flags: tuple[str, ...] = ()


class GoldIndex:
    """Groups gold ProblemRecords by (level, year, problem_number) for fast
    alignment lookups."""

    def __init__(self, records: list[ProblemRecord]) -> None:
        self._by_key: dict[tuple[str | None, int | None, int], list[ProblemRecord]] = defaultdict(list)
        for rec in records:
            self._by_key[(rec.level, rec.year, rec.problem_number)].append(rec)

    def candidates(self, level: str | None, year: int | None, problem_number: int) -> list[ProblemRecord]:
        return self._by_key.get((level, year, problem_number), [])


def _content_tokens(text: str) -> set[str]:
    tokens = {t.lower() for t in _TOKEN_RE.findall(text)}
    return {t for t in tokens if t not in _STOPWORDS and not t.isdigit()}


def text_overlap_score(solution_body: str, problem_body: str) -> float:
    """Jaccard overlap on content tokens (numbers + physics nouns)."""
    left = _content_tokens(solution_body)
    right = _content_tokens(problem_body)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _pick_by_overlap(
    candidates: list[ProblemRecord],
    solution_body: str,
) -> tuple[ProblemRecord | None, float]:
    if not solution_body.strip():
        return None, 0.0
    scored = [
        (text_overlap_score(solution_body, cand.body_md), cand)
        for cand in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    if best_score <= 0.0:
        return None, 0.0
    if len(scored) > 1 and scored[1][0] > 0 and (best_score - scored[1][0]) < 0.05:
        return None, best_score
    return best, best_score


def align_solution(
    index: GoldIndex,
    *,
    level: str | None,
    year: int | None,
    problem_number: int,
    variant_hint: str | None,
    round_hint: str | None,
    solution_body: str = "",
) -> AlignResult:
    candidates = index.candidates(level, year, problem_number)

    if not candidates:
        return AlignResult(None, "no_match", 0.0, ("no_gold_match",))

    if len(candidates) == 1:
        overlap = text_overlap_score(solution_body, candidates[0].body_md) if solution_body else 1.0
        flags: list[str] = []
        confidence = 1.0
        if solution_body and overlap < 0.04:
            flags.append(f"low_text_overlap:{overlap:.2f}")
            confidence = 0.65
        return AlignResult(
            candidates[0].id,
            "exact",
            confidence,
            tuple(flags),
        )

    # Multiple gold records share (level, year, problem_number) - usually
    # distinct variants (tipe A/B/C) or rounds (final/semifinal). Try to
    # disambiguate with hints parsed from the solution filename.
    if variant_hint:
        wanted_variant = _VARIANT_LETTER_TO_INT.get(variant_hint.lower())
        if wanted_variant is not None:
            matches = [c for c in candidates if c.variant == wanted_variant]
            if len(matches) == 1:
                return AlignResult(matches[0].id, "exact", 0.9)

    if round_hint:
        matches = [c for c in candidates if c.round == round_hint]
        if len(matches) == 1:
            return AlignResult(matches[0].id, "exact", 0.9)

    if solution_body.strip():
        picked, overlap = _pick_by_overlap(candidates, solution_body)
        if picked is not None and overlap >= 0.08:
            return AlignResult(picked.id, "exact", min(0.95, 0.7 + overlap), ())

    # Still ambiguous: pick the lowest-id candidate as a best-effort guess,
    # but flag it loudly for human review instead of guessing silently.
    best_guess = sorted(candidates, key=lambda c: c.id)[0]
    return AlignResult(
        best_guess.id,
        "ambiguous",
        0.4,
        ("alignment_review_required", f"ambiguous_among:{','.join(sorted(c.id for c in candidates))}"),
    )
