"""Restore OCR-dropped physics symbols via focused local LLM prompts."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from src.llm.llm_client import ChatCompletionFailure, DEFAULT_TIMEOUT_S, chat_completion_json
from src.schema import ProblemRecord, SubPart, ValidationIssue
from src.text.split_problems import extract_subparts
from src.validate import (
    GARBLED_SYMBOL_RE,
    MANGLED_ESCAPE_RE,
    PLACEHOLDER_LEFTOVER_RE,
    SYMBOL_REPAIR_CODES,
    validate_record,
    sync_flags_from_errors,
)

_WORD_RE = re.compile(r"[a-zA-Zà-ÿ]{3,}")
MIN_WORD_RETENTION = 0.85


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def repair_passes_safety_gate(original_body: str, new_body: str) -> tuple[bool, str | None]:
    """Reject LLM repairs that hallucinate, truncate, or corrupt content instead
    of doing a minimal, targeted symbol restoration. This is what caught (after
    the fact) real corruption like a whole body replaced with the literal
    string ``<cleaned markdown>``, invented unrelated sentences, or nonsense
    concatenated LaTeX tokens like ``$m_1gT$``."""
    if PLACEHOLDER_LEFTOVER_RE.search(new_body):
        return False, "placeholder leftover in output"
    if GARBLED_SYMBOL_RE.search(new_body):
        return False, "garbled concatenated symbol in output"
    if MANGLED_ESCAPE_RE.search(new_body):
        return False, "mangled LaTeX escape (control char) in output"

    original_words = _word_set(original_body)
    if original_words:
        new_words = _word_set(new_body)
        retention = len(original_words & new_words) / len(original_words)
        if retention < MIN_WORD_RETENTION:
            return False, f"only {retention:.0%} of original wording retained"
    return True, None


def _default_model() -> str:
    return (
        os.environ.get("LLM_REPAIR_MODEL")
        or os.environ.get("HALLIDAY_TAG_MODEL")
        or (
            "qwen2.5:3b"
            if os.environ.get("LLM_PROVIDER", "").strip().lower() in {"local", "ollama"}
            or os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
            else "qwen3.6-35b"
        )
    )


SYSTEM_PROMPT = """You fix Indonesian physics olympiad problems where OCR dropped LaTeX variable symbols.

The text has blank slots where physics symbols should appear (e.g. "periode rotasinya ," should become "periode rotasinya $T$,").

Return strict JSON only:
{"body_md": "<fixed markdown>", "subparts": [{"label": "a", "text": "..."}]}

Rules:
- Restore ONLY missing $...$ symbols for physics quantities, and ONLY when the
  surrounding Indonesian words justify that exact symbol:
  massa/bermassa -> m (subscript m_1, m_2, ... if multiple masses are introduced),
  panjang -> L or l, jari-jari -> r or R, sudut/kemiringan -> \\theta,
  periode/rotasi/putaran/berosilasi -> T, kecepatan -> v (v_0 for "kecepatan awal"),
  percepatan gravitasi -> g, konstanta pegas -> k, waktu -> t, gaya -> F,
  tegangan tali -> T, energi -> E, massa jenis -> \\rho.
- NEVER insert $T$ or $\\theta$ (or any symbol) unless the specific sentence you
  are editing is actually about that quantity (rotation/period for T, an angle
  for theta). Do not default to generic "$T$, $\\theta$" filler when you are
  unsure what the missing symbols are - if you cannot determine the correct
  symbol from context, leave that slot unchanged instead of guessing.
- When a "dalam X, Y, dan Z" or "(X, Y)" summary list refers back to variables
  already introduced earlier in the SAME body, reuse those exact same symbols
  in the same order - never invent new, unrelated symbols for these lists.
- Fix patterns like "adalah ," -> "adalah $g$," and "dalam , , dan" ->
  "dalam $m_1$, $m_2$, dan $v_0$" (using the record's OWN previously
  introduced variables, not generic placeholders).
- Convert HTML <sup>/<sub> to LaTeX inside $...$
- Keep every word, sentence, and ![](...) image reference from the input
  unchanged and in the same order - you are only inserting missing $...$
  symbols into blank slots, never rewriting, summarizing, shortening, or
  removing existing sentences. The output must be the same length as the
  input plus only the inserted symbols.
- Do not invent new problem content, new clauses, or sentences that are not
  a direct restoration of a blank slot already present in the input.
- subparts: return [] unless you changed subpart text; never duplicate the full body in subparts"""


@dataclass
class SymbolRestoreOutcome:
    record: ProblemRecord
    succeeded: bool
    failure: str | None = None
    model: str | None = None


def _build_messages(rec: ProblemRecord, issues: list[ValidationIssue]) -> list[dict[str, str]]:
    user = {
        "id": rec.id,
        "title": rec.title,
        "body_md": rec.body_md,
        "body_md_original": rec.body_md_raw,
        "subparts": [sp.model_dump() for sp in rec.subparts],
        "validation_errors": [issue.model_dump() for issue in issues],
        "hint": (
            "Use error snippets to locate blank slots. Cross-check body_md_original "
            "(raw OCR from the source PDF) to restore the correct physics symbols."
        ),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def restore_symbols_llm(
    rec: ProblemRecord,
    issues: list[ValidationIssue],
    *,
    model: str | None = None,
    timeout_s: float | None = None,
) -> SymbolRestoreOutcome:
    if model is None:
        model = _default_model()
    if timeout_s is None:
        timeout_s = float(os.environ.get("NETRA_TIMEOUT_S", DEFAULT_TIMEOUT_S))

    completion = chat_completion_json(
        messages=_build_messages(rec, issues),
        model=model,
        max_tokens=min(4096, len(rec.body_md) + 1024),
        timeout_s=timeout_s,
    )
    if isinstance(completion, ChatCompletionFailure):
        return SymbolRestoreOutcome(record=rec, succeeded=False, failure=completion.detail)

    try:
        data = json.loads(completion.content)
    except json.JSONDecodeError as exc:
        return SymbolRestoreOutcome(record=rec, succeeded=False, failure=str(exc))

    body_md = data.get("body_md")
    if not isinstance(body_md, str) or not body_md.strip():
        return SymbolRestoreOutcome(record=rec, succeeded=False, failure="empty body_md")
    body_md = body_md.strip()

    original_body = rec.body_md
    ok, gate_failure = repair_passes_safety_gate(original_body, body_md)
    if not ok:
        return SymbolRestoreOutcome(
            record=rec, succeeded=False, failure=f"rejected by safety gate: {gate_failure}", model=model
        )

    rec.body_md = body_md
    raw_subparts = data.get("subparts", [])
    if isinstance(raw_subparts, list) and raw_subparts:
        subparts: list[SubPart] = []
        for item in raw_subparts:
            if isinstance(item, dict) and item.get("label") and item.get("text"):
                subparts.append(SubPart(label=str(item["label"]), text=str(item["text"])))
        if subparts:
            rec.subparts = subparts
    else:
        rec.subparts = [SubPart(**sp) for sp in extract_subparts(rec.body_md)]

    rec.errors = validate_record(rec)
    attach_flags = [
        f
        for f in rec.flags
        if f.startswith("missing_image:") or f == "expected_image_missing"
    ]
    rec.flags = sync_flags_from_errors(rec.errors, attach_flags)
    rec.errors = validate_record(rec)

    original_codes = {issue.code for issue in issues}
    remaining_codes = {issue.code for issue in rec.errors}
    unfixed_symbol = original_codes & SYMBOL_REPAIR_CODES & remaining_codes

    if unfixed_symbol:
        return SymbolRestoreOutcome(
            record=rec,
            succeeded=False,
            failure=f"still has: {', '.join(sorted(unfixed_symbol))}",
            model=model,
        )

    if rec.llm_repaired is False:
        rec.llm_repaired = True
        rec.llm_model = model
    return SymbolRestoreOutcome(record=rec, succeeded=True, model=model)
