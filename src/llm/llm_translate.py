from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from src.text.attach_images import extract_image_refs
from src.llm.llm_client import (
    DEFAULT_MODEL,
    ChatCompletionFailure,
    LLMCallMetrics,
    chat_completion_json,
)
from src.llm.llm_progress import RepairProgressStore
from src.repair.repair_log import LogFn
from src.schema import ProblemRecord, SubPart

SYSTEM_PROMPT = """You translate Indonesian physics olympiad (OSK/OSP/OSN) problems into clear, natural English.

Rules:
- Translate title, body_md, and each subpart text into English.
- Preserve ALL LaTeX/math exactly: $...$, $$...$$, \\boldsymbol{}, \\text{}, subscripts, Greek letters, numeric values.
- Preserve every image markdown reference ![](...) exactly (same path/filename).
- Keep subpart labels unchanged (a, b, c, ...) — translate only the prompt text.
- Do not translate variable names, units inside math, or competition metadata.
- Do not add solutions, hints, or commentary.
- Return strict JSON only:
  {"title": "...", "body_md": "...", "subparts": [{"label": "a", "text": "..."}]}
- subparts must use the same labels as the input; return [] only when input subparts is empty."""


class TranslationResult(BaseModel):
    title: str
    body_md: str
    subparts: list[SubPart] = Field(default_factory=list)


@dataclass
class TranslationAttempt:
    result: TranslationResult | None
    failure: str | None = None
    failure_detail: str | None = None
    from_cache: bool = False
    metrics: LLMCallMetrics | None = None


@dataclass
class TranslationOutcome:
    record: ProblemRecord
    succeeded: bool
    failure_reason: str | None = None
    failure_detail: str | None = None
    from_cache: bool = False
    metrics: LLMCallMetrics | None = None
    skipped: bool = False


def cache_key(record: ProblemRecord) -> str:
    payload = {
        "title": record.title,
        "body_md": record.body_md,
        "subparts": [sp.model_dump() for sp in record.subparts],
        "images": [img.path for img in record.images],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def translate_cache_dir(cache_root: Path) -> Path:
    return cache_root / "translate"


def load_cached_translation(
    cache_root: Path, record_id: str, key: str
) -> TranslationResult | None:
    path = translate_cache_dir(cache_root) / f"{record_id}_{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return TranslationResult.model_validate(data)
    except (json.JSONDecodeError, ValidationError):
        return None


def save_cached_translation(
    cache_root: Path, record_id: str, key: str, result: TranslationResult
) -> None:
    dest = translate_cache_dir(cache_root)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{record_id}_{key}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def build_translate_messages(record: ProblemRecord) -> list[dict[str, str]]:
    payload = {
        "id": record.id,
        "title": record.title,
        "body_md": record.body_md,
        "subparts": [sp.model_dump() for sp in record.subparts],
        "images": [img.path for img in record.images],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _extract_json_object(content: str) -> dict:
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def parse_translation_response(content: str) -> TranslationResult:
    data = _extract_json_object(content)
    title = data.get("title")
    body_md = data.get("body_md")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("LLM response missing non-empty title")
    if not isinstance(body_md, str) or not body_md.strip():
        raise ValueError("LLM response missing non-empty body_md")

    subparts: list[SubPart] = []
    for raw in data.get("subparts", []):
        if not isinstance(raw, dict):
            continue
        label = raw.get("label")
        text = raw.get("text")
        if isinstance(label, str) and isinstance(text, str) and text.strip():
            subparts.append(SubPart(label=label, text=text))

    return TranslationResult(title=title.strip(), body_md=body_md.strip(), subparts=subparts)


def _math_delimiter_issues(text: str) -> list[str]:
    issues: list[str] = []
    if text.count("$") % 2 != 0:
        issues.append("unbalanced $ delimiters")
    return issues


def accept_translation(
    record: ProblemRecord, translated: TranslationResult
) -> tuple[bool, list[str]]:
    issues: list[str] = []

    if not translated.title.strip():
        issues.append("empty title")
    if not translated.body_md.strip():
        issues.append("empty body_md")

    for msg in _math_delimiter_issues(translated.body_md):
        issues.append(f"body_md: {msg}")
    for sp in translated.subparts:
        for msg in _math_delimiter_issues(sp.text):
            issues.append(f"subpart {sp.label}: {msg}")

    source_labels = [sp.label for sp in record.subparts]
    translated_labels = [sp.label for sp in translated.subparts]
    if source_labels:
        if translated_labels != source_labels:
            issues.append(
                "subpart labels mismatch: "
                f"expected {source_labels}, got {translated_labels}"
            )
    elif translated.subparts:
        issues.append("unexpected subparts in translation")

    source_refs = set(extract_image_refs(record.body_md))
    for sp in record.subparts:
        source_refs.update(extract_image_refs(sp.text))
    translated_text = translated.body_md + "".join(sp.text for sp in translated.subparts)
    translated_refs = set(extract_image_refs(translated_text))
    missing_refs = source_refs - translated_refs
    if missing_refs:
        issues.append(f"missing image refs: {sorted(missing_refs)}")

    extra_refs = translated_refs - source_refs
    if extra_refs:
        issues.append(f"unexpected image refs: {sorted(extra_refs)}")

    return len(issues) == 0, issues


def repair_translation_structure(
    record: ProblemRecord, translated: TranslationResult
) -> TranslationResult:
    repaired = translated.model_copy(deep=True)
    source_refs = extract_image_refs(record.body_md)
    translated_refs = set(extract_image_refs(repaired.body_md))
    for ref in source_refs:
        marker = f"![]({ref})"
        if record.body_md.rstrip().endswith(marker) and ref not in translated_refs:
            repaired.body_md = f"{repaired.body_md.rstrip()}\n\n{marker}"

    if record.body_md.count("$") % 2 == 0 and repaired.body_md.count("$") % 2:
        unmatched: int | None = None
        for index, char in enumerate(repaired.body_md):
            if char == "$":
                unmatched = index if unmatched is None else None
        if unmatched is not None:
            line_end = repaired.body_md.find("\n", unmatched)
            if line_end < 0:
                line_end = len(repaired.body_md)
            insert_at = line_end
            while insert_at > unmatched and repaired.body_md[insert_at - 1].isspace():
                insert_at -= 1
            if insert_at > unmatched and repaired.body_md[insert_at - 1] in ".,;:!?":
                insert_at -= 1
            repaired.body_md = repaired.body_md[:insert_at] + "$" + repaired.body_md[insert_at:]
    return repaired


def apply_translation_to_record(
    record: ProblemRecord, translated: TranslationResult, *, model: str
) -> ProblemRecord:
    updated = record.model_copy(deep=True)
    updated.title_en = translated.title
    updated.body_md_en = translated.body_md
    updated.subparts_en = translated.subparts
    updated.llm_translated = True
    updated.llm_translate_model = model
    return updated


def translate_record(
    record: ProblemRecord,
    *,
    cache_root: Path,
    model: str = DEFAULT_MODEL,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    force: bool = False,
    log: LogFn | None = None,
) -> TranslationAttempt:
    key = cache_key(record)
    if not force:
        cached = load_cached_translation(cache_root, record.id, key)
        if cached is not None:
            return TranslationAttempt(result=cached, from_cache=True)

    messages = build_translate_messages(record)
    completion = chat_completion_json(
        messages=messages,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        reasoning_effort="none",
        log=log,
    )
    if isinstance(completion, ChatCompletionFailure):
        return TranslationAttempt(
            result=None,
            failure="api_error",
            failure_detail=completion.detail,
            metrics=completion.metrics,
        )
    if completion.truncated:
        return TranslationAttempt(
            result=None,
            failure="truncated",
            failure_detail="LLM response truncated at max_tokens",
            metrics=completion.metrics,
        )

    try:
        parsed = parse_translation_response(completion.content)
    except (json.JSONDecodeError, ValueError) as exc:
        return TranslationAttempt(
            result=None,
            failure="parse_error",
            failure_detail=str(exc),
            metrics=completion.metrics,
        )

    save_cached_translation(cache_root, record.id, key, parsed)
    return TranslationAttempt(result=parsed, metrics=completion.metrics)


def translate_record_with_progress(
    record: ProblemRecord,
    *,
    cache_root: Path,
    progress: RepairProgressStore | None,
    model: str = DEFAULT_MODEL,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    force: bool = False,
    log: LogFn | None = None,
) -> TranslationOutcome:
    if record.llm_translated and record.body_md_en and not force:
        return TranslationOutcome(record=record, succeeded=True, skipped=True)

    key = cache_key(record)
    attempt = translate_record(
        record,
        cache_root=cache_root,
        model=model,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        force=force,
        log=log,
    )

    if attempt.result is None:
        if progress is not None:
            progress.mark(
                record.id,
                cache_key=key,
                status=attempt.failure or "api_error",
                error=attempt.failure_detail,
                from_cache=False,
                usage=attempt.metrics.as_dict() if attempt.metrics else None,
            )
        return TranslationOutcome(
            record=record,
            succeeded=False,
            failure_reason=attempt.failure,
            failure_detail=attempt.failure_detail,
            metrics=attempt.metrics,
        )

    repaired = repair_translation_structure(record, attempt.result)
    if repaired != attempt.result:
        attempt.result = repaired
        save_cached_translation(cache_root, record.id, key, repaired)

    ok, issues = accept_translation(record, attempt.result)
    if not ok:
        detail = "; ".join(issues)
        if progress is not None:
            progress.mark(
                record.id,
                cache_key=key,
                status="rejected",
                error=detail,
                from_cache=attempt.from_cache,
                usage=attempt.metrics.as_dict() if attempt.metrics else None,
            )
        return TranslationOutcome(
            record=record,
            succeeded=False,
            failure_reason="rejected",
            failure_detail=detail,
            from_cache=attempt.from_cache,
            metrics=attempt.metrics,
        )

    updated = apply_translation_to_record(record, attempt.result, model=model)
    if progress is not None:
        progress.mark(
            record.id,
            cache_key=key,
            status="cached" if attempt.from_cache else "succeeded",
            from_cache=attempt.from_cache,
            usage=attempt.metrics.as_dict() if attempt.metrics else None,
        )
    return TranslationOutcome(
        record=updated,
        succeeded=True,
        from_cache=attempt.from_cache,
        metrics=attempt.metrics,
    )
