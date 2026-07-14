from __future__ import annotations

from datetime import UTC, datetime

from src.catalog import is_catalog_eligible, record_content_locale, sync_catalog
from src.llm.llm_client import DEFAULT_MODEL
from src.llm.llm_progress import RepairProgressStore, format_usage_summary
from src.llm.llm_translate import translate_record_with_progress
from src.paths import PipelinePaths
from src.record_store import load_jsonl, save_jsonl
from src.schema import ProblemRecord


def select_records_for_translation(
    records: list[ProblemRecord],
    *,
    ids: set[str] | None = None,
    slugs: set[str] | None = None,
    catalog_only: bool = True,
    skip_translated: bool = True,
) -> list[ProblemRecord]:
    selected: list[ProblemRecord] = []
    for record in records:
        if record_content_locale(record) == "en":
            continue
        if ids and record.id not in ids:
            continue
        if slugs and record.document_slug not in slugs:
            continue
        if catalog_only and not is_catalog_eligible(record):
            continue
        if skip_translated and record.llm_translated and record.body_md_en:
            continue
        selected.append(record)
    selected.sort(key=lambda r: (r.level or "", r.year or 0, r.document_slug, r.problem_number, r.id))
    return selected


def run_translate_pipeline(
    paths: PipelinePaths,
    *,
    ids: set[str] | None = None,
    slugs: set[str] | None = None,
    catalog_only: bool = True,
    force: bool = False,
    limit: int | None = None,
    model: str = DEFAULT_MODEL,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    reset_progress: bool = False,
    dry_run: bool = False,
    sync_catalog_after: bool = True,
    log=print,
) -> dict[str, int | str | None]:
    paths.ensure_dirs()
    gold_path = paths.gold_problems_path
    if not gold_path.is_file():
        raise FileNotFoundError(
            f"No gold corpus at {gold_path}. Run extract with --llm-repair first."
        )

    gold = load_jsonl(gold_path, lenient=True)
    if not gold:
        raise FileNotFoundError("Gold corpus is empty.")

    skip_translated = not force
    targets = select_records_for_translation(
        gold,
        ids=ids,
        slugs=slugs,
        catalog_only=catalog_only,
        skip_translated=skip_translated,
    )
    if limit is not None:
        targets = targets[: max(0, limit)]

    summary: dict[str, int | str | None] = {
        "started_at": datetime.now(UTC).isoformat(),
        "model": model,
        "gold_total": len(gold),
        "targets": len(targets),
        "succeeded": 0,
        "skipped": 0,
        "rejected": 0,
        "api_error": 0,
        "parse_error": 0,
        "truncated": 0,
        "cached": 0,
    }

    if dry_run:
        summary["dry_run"] = 1
        if log:
            log(f"Dry run: would translate {len(targets)} problem(s)")
            for record in targets[:20]:
                log(f"  - {record.id}: {record.title}")
            if len(targets) > 20:
                log(f"  ... and {len(targets) - 20} more")
        return summary

    progress_path = paths.llm_cache_dir / "translate_progress.json"
    progress = RepairProgressStore(progress_path, model=model)
    if reset_progress:
        progress.reset()

    by_id = {rec.id: rec for rec in gold}
    for index, record in enumerate(targets, start=1):
        if log:
            log(f"[{index}/{len(targets)}] Translating {record.id} …")

        outcome = translate_record_with_progress(
            record,
            cache_root=paths.llm_cache_dir,
            progress=progress,
            model=model,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            force=force,
            log=log,
        )

        if outcome.skipped:
            summary["skipped"] = int(summary["skipped"]) + 1
            continue

        if outcome.succeeded:
            by_id[outcome.record.id] = outcome.record
            if outcome.from_cache:
                summary["cached"] = int(summary["cached"]) + 1
            else:
                summary["succeeded"] = int(summary["succeeded"]) + 1
            # Persist incrementally so the UI catalog can pick up partial batches.
            partial = list(by_id.values())
            partial.sort(
                key=lambda r: (r.level or "", r.year or 0, r.document_slug, r.problem_number, r.id)
            )
            save_jsonl(gold_path, partial)
            if sync_catalog_after:
                sync_catalog(paths)
            continue

        reason = outcome.failure_reason or "api_error"
        if reason in summary:
            summary[reason] = int(summary[reason]) + 1
        else:
            summary["api_error"] = int(summary["api_error"]) + 1
        if log and outcome.failure_detail:
            log(f"  ✗ {record.id}: {outcome.failure_detail}")

    updated = list(by_id.values())
    updated.sort(key=lambda r: (r.level or "", r.year or 0, r.document_slug, r.problem_number, r.id))
    save_jsonl(gold_path, updated)

    if sync_catalog_after:
        catalog_meta = sync_catalog(paths)
        summary["catalog_total"] = catalog_meta.get("catalog_total")
        summary["catalog_updated_at"] = catalog_meta.get("updated_at")

    summary["finished_at"] = datetime.now(UTC).isoformat()
    summary["usage_summary"] = format_usage_summary(progress.usage_totals, model=model)
    return summary
