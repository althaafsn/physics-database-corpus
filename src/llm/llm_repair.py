from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.llm.llm_client import (
    DEFAULT_MODEL,
    DEFAULT_REPAIR_MAX_TOKENS,
    ChatCompletionFailure,
    LLMCallMetrics,
    chat_completion_json,
)
from src.llm.llm_progress import RepairProgressStore
from src.repair.repair_log import LogFn, log_repair_event
from src.schema import ProblemRecord, SubPart, ValidationIssue
from src.text.split_problems import extract_subparts
from src.validate import validate_record


class FixApplied(BaseModel):
    code: str
    before: str = ""
    after: str = ""


class RepairResult(BaseModel):
    body_md: str
    subparts: list[SubPart] = Field(default_factory=list)
    fixes_applied: list[FixApplied] = Field(default_factory=list)


@dataclass
class RepairAttempt:
    result: RepairResult | None
    failure: str | None = None
    failure_detail: str | None = None
    from_cache: bool = False
    metrics: LLMCallMetrics | None = None


@dataclass
class RepairOutcome:
    record: ProblemRecord
    succeeded: bool
    remaining_errors: list[ValidationIssue] | None = None
    failure_reason: str | None = None
    failure_detail: str | None = None
    from_cache: bool = False
    metrics: LLMCallMetrics | None = None
    skipped_duplicate: bool = False


SYSTEM_PROMPT = """You are an editor for Indonesian physics olympiad (OSK/OSP/OSN) problems.
You receive markdown that was already stripped of ads/footers, plus a list of parse errors that MUST all be fixed.

Rules:
- Fix every listed error code exactly.
- Restore OCR-dropped physics symbols only when implied by context.
- Strip promotional footer/ad content (Dimensi Sains, social links, phone numbers).
- Normalize math to $...$ or $$...$$; replace HTML <sup>/<sub> with LaTeX.
- Preserve problem meaning, numeric values, image markdown ![](...) refs, and subpart labels.
- Do not invent values unsupported by the text.
- Return strict JSON with keys: body_md, subparts, fixes_applied.
- subparts is a list of {"label": "a", "text": "..."} extracted from the cleaned body.
- fixes_applied lists each fix: {"code": "...", "before": "...", "after": "..."}.
- before and after must always be strings (use "" if not applicable).
- Keep fixes_applied short: at most 5 entries, before/after under 80 characters each.
- Do NOT echo the full body_md inside fixes_applied."""

COMPACT_SYSTEM_PROMPT = """You are an editor for Indonesian physics olympiad problems.
Fix ONLY the listed parse errors in body_md. Return minimal JSON:
{"body_md": "<cleaned markdown>", "subparts": [], "fixes_applied": []}

Rules:
- Fix every listed error code in body_md.
- Strip footer/ad lines (Dimensi Sains, social links, phone numbers).
- Normalize math to $...$; replace HTML <sup>/<sub> with LaTeX.
- Preserve meaning, numbers, image refs ![](...), and subpart labels (a), (b), ...
- subparts: return [] unless you changed subpart text; never duplicate the full body.
- fixes_applied: always return [] (do not enumerate fixes).
- Output ONLY the JSON object, nothing else."""

LARGE_RECORD_CHAR_THRESHOLD = 4000
LARGE_SUBPART_COUNT = 8


def cache_key(record: ProblemRecord, issues: list[ValidationIssue]) -> str:
    payload = {
        "body_md_raw": record.body_md_raw or record.body_md,
        "body_md": record.body_md,
        "errors": [issue.code for issue in issues],
        "images": [img.path for img in record.images],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def load_cached_repair(cache_dir: Path, record_id: str, key: str) -> RepairResult | None:
    path = cache_dir / f"{record_id}_{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RepairResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


def save_cached_repair(cache_dir: Path, record_id: str, key: str, result: RepairResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{record_id}_{key}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def build_repair_messages(
    record: ProblemRecord,
    issues: list[ValidationIssue],
    *,
    compact: bool = False,
) -> list[dict[str, str]]:
    payload = {
        "id": record.id,
        "title": record.title,
        "level": record.level,
        "year": record.year,
        "body_md": record.body_md,
        "body_md_original": record.body_md_raw,
        "subparts": [sp.model_dump() for sp in record.subparts],
        "images": [img.path for img in record.images],
        "errors": [issue.model_dump() for issue in issues],
    }
    if compact:
        payload["note"] = (
            "Large record: return subparts as [] unless a subpart text changed. "
            "fixes_applied must be []."
        )
    return [
        {"role": "system", "content": COMPACT_SYSTEM_PROMPT if compact else SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def should_use_compact_repair(record: ProblemRecord) -> bool:
    body = record.body_md
    return len(body) >= LARGE_RECORD_CHAR_THRESHOLD or len(record.subparts) >= LARGE_SUBPART_COUNT


def estimate_repair_max_tokens(record: ProblemRecord, *, base: int = DEFAULT_REPAIR_MAX_TOKENS) -> int:
    body = record.body_md
    # Output should stay proportional to input to avoid runaway generations.
    proportional = len(body) + 1024
    return min(base, max(1024, proportional))


def _extract_json_object(content: str) -> dict:
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def _normalize_fix_entry(raw: object) -> FixApplied | None:
    if not isinstance(raw, dict):
        return None
    code = raw.get("code")
    if not isinstance(code, str) or not code.strip():
        return None
    before = raw.get("before")
    after = raw.get("after")
    return FixApplied(
        code=code.strip(),
        before=before if isinstance(before, str) else "",
        after=after if isinstance(after, str) else "",
    )


def parse_repair_response(content: str) -> RepairResult:
    data = _extract_json_object(content)
    body_md = data.get("body_md")
    if not isinstance(body_md, str) or not body_md.strip():
        raise ValueError("LLM response missing non-empty body_md")

    subparts: list[SubPart] = []
    for raw in data.get("subparts", []):
        if not isinstance(raw, dict):
            continue
        label = raw.get("label")
        text = raw.get("text")
        if isinstance(label, str) and isinstance(text, str):
            subparts.append(SubPart(label=label, text=text))

    fixes: list[FixApplied] = []
    for raw in data.get("fixes_applied", []):
        fix = _normalize_fix_entry(raw)
        if fix is not None:
            fixes.append(fix)

    return RepairResult(
        body_md=body_md,
        subparts=subparts,
        fixes_applied=fixes,
    )


def accept_repair(
    record: ProblemRecord,
    repaired: RepairResult,
    original_issues: list[ValidationIssue],
) -> tuple[bool, list[ValidationIssue]]:
    original_codes = {issue.code for issue in original_issues}
    trial = record.model_copy(deep=True)
    trial.body_md = repaired.body_md
    trial.subparts = repaired.subparts
    remaining = validate_record(trial)
    remaining_codes = {issue.code for issue in remaining}
    if original_codes & remaining_codes:
        return False, remaining
    return True, remaining


def _start_heartbeat(record_id: str, log: LogFn | None) -> threading.Event:
    stop = threading.Event()

    def _run() -> None:
        elapsed = 0
        while not stop.wait(10.0):
            elapsed += 10
            if log:
                log(f"  … still waiting on Netra for {record_id} ({elapsed}s elapsed, no hang)")

    threading.Thread(target=_run, daemon=True).start()
    return stop


def repair_record(
    record: ProblemRecord,
    issues: list[ValidationIssue],
    *,
    model: str = DEFAULT_MODEL,
    cache_dir: Path | None = None,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    log: LogFn | None = None,
    index: int | None = None,
    total: int | None = None,
    compact: bool | None = None,
) -> RepairAttempt:
    if not issues:
        return RepairAttempt(result=None, failure="no_issues")

    key = cache_key(record, issues)
    body = record.body_md_raw or record.body_md
    error_codes = [issue.code for issue in issues]
    use_compact = should_use_compact_repair(record) if compact is None else compact
    if max_tokens is None:
        max_tokens = estimate_repair_max_tokens(record)

    log_repair_event(
        log,
        phase="inspect",
        record_id=record.id,
        index=index,
        total=total,
        errors=error_codes,
        body_chars=len(body),
        cache_key=key,
    )

    if cache_dir is not None:
        cached = load_cached_repair(cache_dir, record.id, key)
        if cached is not None:
            log_repair_event(log, phase="cache_hit", record_id=record.id, index=index, total=total)
            return RepairAttempt(result=cached, from_cache=True)

    attempts: list[tuple[bool, int]] = [(use_compact, max_tokens)]
    if not use_compact:
        attempts.append((True, max_tokens))
    elif max_tokens >= DEFAULT_REPAIR_MAX_TOKENS:
        attempts.append((True, DEFAULT_REPAIR_MAX_TOKENS // 2))

    last_failure: RepairAttempt | None = None
    for attempt_idx, (compact_mode, token_budget) in enumerate(attempts):
        if attempt_idx > 0:
            log_repair_event(
                log,
                phase="retry_compact",
                record_id=record.id,
                index=index,
                total=total,
                detail=f"max_tokens={token_budget}",
            )

        log_repair_event(log, phase="api_start", record_id=record.id, index=index, total=total)
        stop_heartbeat = _start_heartbeat(record.id, log)
        try:
            completion = chat_completion_json(
                messages=build_repair_messages(record, issues, compact=compact_mode),
                model=model,
                temperature=0,
                timeout_s=timeout_s,
                max_tokens=token_budget,
                log=log,
            )
        finally:
            stop_heartbeat.set()

        if isinstance(completion, ChatCompletionFailure):
            last_failure = RepairAttempt(
                result=None,
                failure=completion.reason,
                failure_detail=completion.detail,
                metrics=completion.metrics,
            )
            continue

        if completion.truncated:
            last_failure = RepairAttempt(
                result=None,
                failure="truncated",
                failure_detail=(
                    f"Response hit max_tokens={token_budget} "
                    f"({completion.metrics.completion_tokens} completion tokens)"
                ),
                metrics=completion.metrics,
            )
            continue

        try:
            result = parse_repair_response(completion.content)
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as exc:
            detail = str(exc)
            if completion.truncated or completion.finish_reason == "length":
                detail = f"truncated JSON: {exc}"
            log_repair_event(
                log,
                phase="parse_error",
                record_id=record.id,
                index=index,
                total=total,
                detail=detail,
            )
            last_failure = RepairAttempt(
                result=None,
                failure="truncated" if completion.finish_reason == "length" else f"parse_error: {exc}",
                failure_detail=detail,
                metrics=completion.metrics,
            )
            continue

        if cache_dir is not None:
            save_cached_repair(cache_dir, record.id, key, result)
        return RepairAttempt(
            result=result,
            from_cache=False,
            metrics=completion.metrics,
        )

    return last_failure or RepairAttempt(result=None, failure="unknown_error")


def apply_repair_to_record(
    record: ProblemRecord,
    issues: list[ValidationIssue],
    *,
    model: str = DEFAULT_MODEL,
    cache_dir: Path | None = None,
    progress: RepairProgressStore | None = None,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    log: LogFn | None = None,
    index: int | None = None,
    total: int | None = None,
) -> RepairOutcome:
    key = cache_key(record, issues)
    attempt = repair_record(
        record,
        issues,
        model=model,
        cache_dir=cache_dir,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        log=log,
        index=index,
        total=total,
    )

    if attempt.result is None:
        failure = attempt.failure or "unknown_error"
        if progress is not None:
            if failure == "api_error":
                status = "api_error"
            elif failure == "truncated" or failure.startswith("truncated"):
                status = "truncated"
            else:
                status = "parse_error"
            progress.mark(
                record.id,
                cache_key=key,
                status=status,
                error=attempt.failure_detail or failure,
                from_cache=False,
                usage=attempt.metrics.as_dict() if attempt.metrics else None,
            )
        log_repair_event(
            log,
            phase=failure,
            record_id=record.id,
            index=index,
            total=total,
            detail=attempt.failure_detail,
        )
        return RepairOutcome(
            record=record,
            succeeded=False,
            remaining_errors=issues,
            failure_reason=failure,
            failure_detail=attempt.failure_detail,
            metrics=attempt.metrics,
        )

    accepted, remaining = accept_repair(record, attempt.result, issues)
    if not accepted:
        record.body_md = record.body_md_raw or record.body_md
        remaining_codes = [issue.code for issue in remaining]
        if progress is not None:
            progress.mark(
                record.id,
                cache_key=key,
                status="rejected",
                error="validation_failed",
                from_cache=attempt.from_cache,
                usage=attempt.metrics.as_dict() if attempt.metrics else None,
            )
        log_repair_event(
            log,
            phase="validation_rejected",
            record_id=record.id,
            index=index,
            total=total,
            remaining=remaining_codes,
        )
        return RepairOutcome(
            record=record,
            succeeded=False,
            remaining_errors=remaining,
            failure_reason="validation_failed",
            failure_detail=f"still has: {', '.join(remaining_codes)}",
            from_cache=attempt.from_cache,
            metrics=attempt.metrics,
        )

    record.body_md = attempt.result.body_md
    if attempt.result.subparts:
        record.subparts = attempt.result.subparts
    else:
        record.subparts = [SubPart(**sp) for sp in extract_subparts(attempt.result.body_md)]
    record.llm_repaired = True
    record.llm_model = model

    from src.text.clean import finalize_record_after_repair
    from src.paths import PipelinePaths

    paths = PipelinePaths.resolve()
    output_folder = Path(record.source.md).parent
    finalize_record_after_repair(record, output_folder, paths.assets_dir)

    record.errors = validate_record(record)
    from src.validate import sync_flags_from_errors

    attach_flags = [
        f for f in record.flags if f.startswith("missing_image:") or f == "expected_image_missing"
    ]
    record.flags = sync_flags_from_errors(record.errors, attach_flags)

    if progress is not None:
        progress.mark(
            record.id,
            cache_key=key,
            status="cached" if attempt.from_cache else "succeeded",
            from_cache=attempt.from_cache,
            usage=attempt.metrics.as_dict() if attempt.metrics else None,
        )

    log_repair_event(
        log,
        phase="cached_ok" if attempt.from_cache else "succeeded",
        record_id=record.id,
        index=index,
        total=total,
    )

    return RepairOutcome(
        record=record,
        succeeded=True,
        remaining_errors=record.errors,
        from_cache=attempt.from_cache,
        metrics=attempt.metrics,
    )
