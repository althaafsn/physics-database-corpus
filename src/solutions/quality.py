"""Formatting and progressive-step quality checks for worked solutions."""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.solutions.hints import split_solution_into_hints
from src.solutions.safety_gate import solution_passes_safety_gate
from src.solutions.schema import SolutionStep


_MERGE_MARKER_RE = re.compile(r"<<<(?:MARKER|PDFTEXT)|PDFTEXT>>>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class SolutionQuality:
    steps: list[SolutionStep]
    errors: list[str]
    formatting_confidence: float


def parse_solution_quality(body_md: str) -> SolutionQuality:
    text = (body_md or "").strip()
    errors: list[str] = []
    safe, reason = solution_passes_safety_gate(text)
    if not safe and reason:
        errors.append("safety_gate_rejected")
    if _MERGE_MARKER_RE.search(text):
        errors.append("unresolved_merge_marker")
    if _CONTROL_RE.search(text):
        errors.append("control_character")
    if text.count("$") % 2:
        errors.append("unbalanced_math_delimiter")

    hints = split_solution_into_hints(text)
    steps = [
        SolutionStep(index=index, body_md=step, kind="derivation")
        for index, step in enumerate(hints, start=1)
        if step.strip()
    ]
    if text and not steps:
        errors.append("no_solution_steps")
    if len(errors) > 1:
        errors = list(dict.fromkeys(errors))
    confidence = max(0.0, 1.0 - 0.35 * len(errors))
    return SolutionQuality(
        steps=steps,
        errors=errors,
        formatting_confidence=round(confidence, 3),
    )
