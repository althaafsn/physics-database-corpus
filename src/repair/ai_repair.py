"""Tiered AI repair: deterministic fixes first, cheap symbol LLM, full repair last."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.math.math_normalize import normalize_problem_symbols
from src.schema import ProblemRecord, ValidationIssue
from src.math.symbol_restore import restore_symbols_llm
from src.validate import (
    apply_validation,
    filter_symbol_restore_issues,
    needs_full_llm_repair,
    needs_symbol_restore,
)

DEFAULT_SYMBOL_MODEL = "qwen2.5:3b"


@dataclass
class DeterministicRepairResult:
    record: ProblemRecord
    changed: bool
    errors_before: int
    errors_after: int


@dataclass
class TieredRepairOutcome:
    record: ProblemRecord
    deterministic_changed: bool
    symbol_restore_attempted: bool
    symbol_restore_succeeded: bool
    full_llm_repair_needed: bool
    remaining_errors: list[ValidationIssue]


def symbol_restore_model() -> str:
    return (
        os.environ.get("SYMBOL_RESTORE_MODEL")
        or os.environ.get("LLM_REPAIR_MODEL")
        or DEFAULT_SYMBOL_MODEL
    )


def apply_deterministic_symbol_repair(record: ProblemRecord) -> DeterministicRepairResult:
    """Normalize math-italic Unicode and re-validate without any LLM."""
    errors_before = len(record.errors)
    original_body = record.body_md
    record.body_md = normalize_problem_symbols(record.body_md)
    for subpart in record.subparts:
        subpart.text = normalize_problem_symbols(subpart.text)
    changed = record.body_md != original_body or any(
        sp.text != normalize_problem_symbols(sp.text) for sp in record.subparts
    )
    apply_validation(record)
    return DeterministicRepairResult(
        record=record,
        changed=changed,
        errors_before=errors_before,
        errors_after=len(record.errors),
    )


def apply_symbol_restore_if_needed(
    record: ProblemRecord,
    *,
    model: str | None = None,
    timeout_s: float | None = None,
) -> TieredRepairOutcome:
    """Run focused symbol-restore LLM only when symbol-specific errors remain."""
    det = apply_deterministic_symbol_repair(record)
    issues = filter_symbol_restore_issues(record.errors)
    if not issues:
        return TieredRepairOutcome(
            record=record,
            deterministic_changed=det.changed,
            symbol_restore_attempted=False,
            symbol_restore_succeeded=False,
            full_llm_repair_needed=needs_full_llm_repair(record.errors),
            remaining_errors=list(record.errors),
        )

    outcome = restore_symbols_llm(
        record,
        issues,
        model=model or symbol_restore_model(),
        timeout_s=timeout_s,
    )
    apply_validation(record)
    return TieredRepairOutcome(
        record=record,
        deterministic_changed=det.changed,
        symbol_restore_attempted=True,
        symbol_restore_succeeded=outcome.succeeded,
        full_llm_repair_needed=needs_full_llm_repair(record.errors),
        remaining_errors=list(record.errors),
    )


def record_needs_ai(record: ProblemRecord) -> bool:
    return bool(record.errors) and (
        needs_symbol_restore(record.errors) or needs_full_llm_repair(record.errors)
    )
