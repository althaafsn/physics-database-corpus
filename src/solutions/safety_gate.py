"""Safety checks for ingested solution text, modeled on
src/symbol_restore.py's repair_passes_safety_gate: reject placeholder
leftovers, garbled symbols, and mangled LaTeX escapes; flag (not reject)
low-confidence content for human review instead of silently publishing it."""
from __future__ import annotations

import re

from src.validate import GARBLED_SYMBOL_RE, MANGLED_ESCAPE_RE, PLACEHOLDER_LEFTOVER_RE

MIN_BODY_CHARS = 20
_WORD_RE = re.compile(r"[a-zA-Zà-ÿ]{3,}")


def solution_passes_safety_gate(body_md: str) -> tuple[bool, str | None]:
    """Hard rejects: content that should never be stored, regardless of
    alignment confidence (these indicate OCR/LLM corruption, not a real
    solution)."""
    if not body_md or len(body_md.strip()) < MIN_BODY_CHARS:
        return False, "body too short to be a real worked solution"
    if PLACEHOLDER_LEFTOVER_RE.search(body_md):
        return False, "placeholder leftover in output"
    if GARBLED_SYMBOL_RE.search(body_md):
        return False, "garbled concatenated symbol in output"
    if MANGLED_ESCAPE_RE.search(body_md):
        return False, "mangled LaTeX escape (control char) in output"
    word_count = len(_WORD_RE.findall(body_md))
    if word_count < 5:
        return False, "too few words to be a genuine derivation"
    return True, None
