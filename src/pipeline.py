from __future__ import annotations

import json
from pathlib import Path

from src.text.attach_images import attach_images
from src.catalog import sync_catalog
from src.text.classify_topic import classify_topic, llm_classify_topic
from src.text.clean import clean_record
from src.repair.ai_repair import (
    apply_deterministic_symbol_repair,
    apply_symbol_restore_if_needed,
    needs_full_llm_repair,
    needs_symbol_restore,
    record_needs_ai,
    symbol_restore_model,
)
from src.repair.repair_images import repair_record_images
from src.ingest.ingest_registry import IngestRegistryStore
from src.llm.llm_client import DEFAULT_MAX_TOKENS, DEFAULT_TIMEOUT_S, format_metrics_line
from src.llm.llm_progress import RepairProgressStore, format_usage_summary
from src.llm.llm_repair import apply_repair_to_record, cache_key
from src.text.parse_filename import parse_document
from src.paths import PipelinePaths
from src.record_store import (
    append_run_history,
    merge_records,
    migrate_legacy_problems,
    new_run_id,
    save_jsonl,
    source_pdf_key,
)
from src.repair.repair_log import default_log
from src.schema import Manifest, MetadataOverrides, ParseError, ProblemRecord, ProblemSource, SubPart
from src.text.split_problems import extract_subparts, split_problems_from_folder
from src.validate import apply_validation
from src.text.title_quality import choose_problem_title

LOW_CONFIDENCE_THRESHOLD = 0.6


def make_problem_id(
    doc_slug: str,
    problem_number: int,
    *,
    level: str | None,
    year: int | None,
    round_name: str | None,
    variant: int | None,
) -> str:
    if level and year is not None:
        parts = [level, str(year)]
        if round_name:
            parts.append(round_name)
        if variant is not None:
            parts.append(f"v{variant}")
        parts.append(f"{problem_number:02d}")
        return "-".join(parts)

    safe_slug = doc_slug.replace(" ", "-")
    parts = [safe_slug]
    if variant is not None:
        parts.append(f"v{variant}")
    parts.append(f"{problem_number:02d}")
    return "-".join(parts)


def stable_problem_id(
    doc_slug: str,
    problem_number: int,
    level: str | None,
    year: int | None,
    round_name: str | None,
    variant: int | None,
) -> str:
    """Keep IDs published before variant-aware parsing was introduced."""
    if level == "OSK" and year == 2011 and round_name is None and variant == 2:
        return f"OSK-2011-{problem_number:02d}"
    if level == "OSK" and year == 2012 and round_name is None:
        if variant == 1:
            return f"OSK-2012-{problem_number:02d}"
        if variant == 3:
            return f"OSK-2012-{problem_number + 8:02d}"
    return make_problem_id(
        doc_slug,
        problem_number,
        level=level,
        year=year,
        round_name=round_name,
        variant=variant,
    )


def process_output_folder(
    output_folder: Path,
    *,
    paths: PipelinePaths,
    metadata_overrides: MetadataOverrides | None = None,
    use_llm_topics: bool = False,
) -> tuple[list[ProblemRecord], ParseError | None]:
    folder_name = output_folder.name
    md_path = output_folder / f"{folder_name}.md"
    meta_path = output_folder / f"{folder_name}_meta.json"

    if not md_path.is_file():
        return [], ParseError(folder=str(output_folder), error=f"Missing markdown: {md_path}")

    md_text = md_path.read_text(encoding="utf-8")
    pdf_path = Path(paths.pdf_dir / f"{folder_name}.pdf")
    if not pdf_path.is_file():
        registry = IngestRegistryStore(paths.registry_path).get(folder_name)
        if registry and Path(registry.pdf_path).is_file():
            pdf_path = Path(registry.pdf_path)

    try:
        doc_meta = parse_document(
            folder_name,
            pdf_dir=paths.pdf_dir,
            md_text=md_text,
            pdf_path=pdf_path if pdf_path.is_file() else None,
            overrides=metadata_overrides,
        )
        raw_problems = split_problems_from_folder(output_folder)
    except Exception as exc:
        return [], ParseError(folder=str(output_folder), error=str(exc))

    records: list[ProblemRecord] = []
    for raw in raw_problems:
        if not raw.body_md.strip():
            continue

        topic, confidence, scores = classify_topic(raw.title, raw.body_md)
        if use_llm_topics and (topic == "mixed" or confidence < LOW_CONFIDENCE_THRESHOLD):
            llm_topic = llm_classify_topic(raw.title, raw.body_md)
            if llm_topic:
                topic = llm_topic
                confidence = max(confidence, 0.75)

        problem_variant = raw.variant if raw.variant is not None else doc_meta.variant
        images, flags = attach_images(
            raw.body_md,
            output_folder,
            paths.assets_dir,
            doc_meta.level,
            doc_meta.year,
            raw.problem_number,
            document_slug=doc_meta.slug,
        )
        subparts = [SubPart(**sp) for sp in extract_subparts(raw.body_md)]

        record = ProblemRecord(
            id=stable_problem_id(
                doc_meta.slug,
                raw.problem_number,
                level=doc_meta.level,
                year=doc_meta.year,
                round_name=doc_meta.round,
                variant=problem_variant,
            ),
            document_slug=doc_meta.slug,
            level=doc_meta.level,
            year=doc_meta.year,
            round=doc_meta.round,
            variant=problem_variant,
            problem_number=raw.problem_number,
            title=choose_problem_title(
                raw.title,
                raw.body_md,
                level=doc_meta.level,
                year=doc_meta.year,
                number=raw.problem_number,
                document_slug=doc_meta.slug,
            ),
            topic=topic,
            topic_confidence=round(confidence, 4),
            topic_scores={k: round(v, 4) for k, v in scores.items()},
            subparts=subparts,
            body_md=raw.body_md,
            images=images,
            source=ProblemSource(
                pdf=str(pdf_path.resolve()) if pdf_path.is_file() else doc_meta.source_pdf,
                md=str(md_path.resolve()),
                meta_json=str(meta_path.resolve()) if meta_path.is_file() else "",
            ),
            flags=flags,
        )
        clean_record(record)
        repair_record_images(record, output_folder, paths.assets_dir)
        if not record.body_md.strip():
            continue
        apply_validation(record)
        records.append(record)

    return records, None


def _silver_index(records: list) -> dict[str, list]:
    index: dict[str, list] = {}
    for rec in records:
        slug = getattr(rec, "document_slug", None) or Path(rec.source.pdf).stem
        index.setdefault(slug, []).append(rec)
    return index


def _resolve_target_slugs(
    paths: PipelinePaths,
    registry: IngestRegistryStore,
    *,
    only_slugs: set[str] | None,
    incremental: bool,
    full_rebuild: bool,
    silver_records: list,
) -> tuple[list[str], list[str]]:
    """Return (slugs_to_process, pdf_only_slugs)."""
    registry.scan_paths(paths, silver_by_slug=_silver_index(silver_records))

    if only_slugs:
        return sorted(only_slugs), []

    if full_rebuild:
        slugs: list[str] = []
        for folder in sorted(paths.bronze_dir.iterdir()):
            if not folder.is_dir():
                continue
            if (folder / f"{folder.name}.md").is_file():
                slugs.append(folder.name)
        return slugs, []

    if incremental:
        pending = registry.pending_silver_slugs()
        return pending, []

    slugs = []
    pdf_only = []
    for slug, entry in registry.documents.items():
        if entry.bronze_hash:
            slugs.append(slug)
        else:
            pdf_only.append(slug)
    return sorted(slugs), sorted(pdf_only)


def run_pipeline(
    paths: PipelinePaths | None = None,
    *,
    input_dir: Path | None = None,
    pdf_dir: Path | None = None,
    out_dir: Path | None = None,
    only_slugs: set[str] | None = None,
    incremental: bool = True,
    full_rebuild: bool = False,
    metadata_overrides: MetadataOverrides | None = None,
    use_llm_topics: bool = False,
    llm_repair: bool = False,
    llm_model: str = "qwen3.6-35b",
    llm_cache_dir: Path | None = None,
    llm_reset_progress: bool = False,
    llm_timeout_s: float = DEFAULT_TIMEOUT_S,
    llm_max_tokens: int = DEFAULT_MAX_TOKENS,
    llm_verbose: bool = True,
    repair_only_processed: bool = True,
    repair_all: bool = False,
) -> Manifest:
    if paths is None:
        paths = PipelinePaths.resolve()
    if input_dir is not None or pdf_dir is not None or out_dir is not None:
        paths = PipelinePaths(
            root=paths.root,
            pdf_dir=(pdf_dir or paths.pdf_dir).resolve(),
            bronze_dir=(input_dir or paths.bronze_dir).resolve(),
            parsed_dir=(out_dir or paths.parsed_dir).resolve(),
        )

    paths.ensure_dirs()
    run_id = new_run_id()
    registry = IngestRegistryStore(paths.registry_path)

    existing_silver = migrate_legacy_problems(
        paths.silver_problems_path,
        paths.legacy_problems_path,
    )

    target_slugs, pdf_only_slugs = _resolve_target_slugs(
        paths,
        registry,
        only_slugs=only_slugs,
        incremental=incremental and not full_rebuild,
        full_rebuild=full_rebuild,
        silver_records=existing_silver,
    )

    if llm_cache_dir is None:
        llm_cache_dir = paths.llm_cache_dir

    progress_store: RepairProgressStore | None = None
    if llm_repair:
        progress_path = llm_cache_dir / "repair_progress.json"
        progress_store = RepairProgressStore(progress_path, model=llm_model)
        if llm_reset_progress:
            progress_store.reset()

    batch_records: list[ProblemRecord] = []
    low_confidence: list[ProblemRecord] = []
    unfixed_errors: list[ProblemRecord] = []
    parse_errors: list[ParseError] = []
    skipped: list[str] = []
    processed_slugs: list[str] = []

    records_with_errors = 0
    llm_repair_attempted = 0
    llm_repair_succeeded = 0
    llm_repair_failed = 0
    llm_repair_cached = 0
    llm_repair_api_errors = 0
    llm_repair_parse_errors = 0
    llm_repair_truncated = 0
    llm_repair_rejected = 0
    llm_repair_skipped_duplicates = 0
    symbol_restore_attempted = 0
    symbol_restore_succeeded = 0
    deterministic_symbol_fixed = 0
    needs_ai_records: list[ProblemRecord] = []

    for slug in target_slugs:
        folder = paths.bronze_folder(slug)
        md_path = folder / f"{slug}.md"
        if not md_path.is_file():
            skipped.append(str(folder))
            continue

        records, err = process_output_folder(
            folder,
            paths=paths,
            metadata_overrides=metadata_overrides,
            use_llm_topics=use_llm_topics,
        )
        if err:
            parse_errors.append(err)
            continue

        batch_records.extend(records)
        processed_slugs.append(slug)
        err_count = sum(1 for r in records if r.errors)
        registry.mark_silver(
            slug,
            problems_count=len(records),
            errors_count=err_count,
            run_id=run_id,
        )

    if repair_all and not target_slugs and llm_repair:
        all_records = list(existing_silver)
    else:
        replace_pdfs = {source_pdf_key(r) for r in batch_records}
        replace_slugs = set(processed_slugs)
        all_records = merge_records(
            existing_silver,
            batch_records,
            replace_source_pdfs=replace_pdfs,
            replace_slugs=replace_slugs,
        )

    for rec in all_records:
        if rec.errors:
            records_with_errors += 1

    # Deterministic symbol normalization on every record in the batch (cheap).
    for rec in batch_records:
        det = apply_deterministic_symbol_repair(rec)
        if det.changed and det.errors_after < det.errors_before:
            deterministic_symbol_fixed += 1

    repair_scope = (
        all_records
        if repair_all or not (repair_only_processed and processed_slugs)
        else batch_records
    )

    tiered_ai = llm_repair
    symbol_model = symbol_restore_model() if tiered_ai else llm_model

    if tiered_ai:
        repair_targets = [rec for rec in repair_scope if needs_symbol_restore(rec.errors)]
        symbol_total = len(repair_targets)
        if symbol_total and llm_verbose:
            print(
                f"Symbol restore: {symbol_total} records "
                f"(model={symbol_model}, deterministic pass already applied)",
                flush=True,
            )
        for repair_index, rec in enumerate(repair_targets, start=1):
            symbol_restore_attempted += 1
            outcome = apply_symbol_restore_if_needed(
                rec,
                model=symbol_model,
                timeout_s=llm_timeout_s,
            )
            if outcome.symbol_restore_succeeded:
                symbol_restore_succeeded += 1
            elif llm_verbose and outcome.symbol_restore_attempted:
                print(
                    f"[Symbol {repair_index}/{symbol_total}] {rec.id} failed | "
                    f"errors={len(rec.errors)}",
                    flush=True,
                )

    if llm_repair:
        repair_targets = [
            rec
            for rec in repair_scope
            if needs_full_llm_repair(rec.errors)
            or (not tiered_ai and rec.errors)
        ]
        repair_total = len(repair_targets)
        log_fn = default_log if llm_verbose else None
        seen_jobs: set[tuple[str, str]] = set()

        print(
            f"LLM repair: {repair_total} records with errors "
            f"(timeout={llm_timeout_s:.0f}s, max_tokens={llm_max_tokens})",
            flush=True,
        )

        for repair_index, rec in enumerate(repair_targets, start=1):
            job_key = (rec.id, cache_key(rec, rec.errors))
            if job_key in seen_jobs:
                llm_repair_skipped_duplicates += 1
                continue
            seen_jobs.add(job_key)

            llm_repair_attempted += 1
            original_errors = list(rec.errors)
            outcome = apply_repair_to_record(
                rec,
                original_errors,
                model=llm_model,
                cache_dir=llm_cache_dir,
                progress=progress_store,
                timeout_s=llm_timeout_s,
                max_tokens=llm_max_tokens,
                log=log_fn,
                index=repair_index,
                total=repair_total,
            )

            if outcome.succeeded:
                for other in repair_targets:
                    if other is rec:
                        continue
                    if (other.id, cache_key(other, other.errors)) == job_key:
                        other.body_md = rec.body_md
                        other.subparts = rec.subparts
                        other.llm_repaired = rec.llm_repaired
                        other.llm_model = rec.llm_model
                        other.errors = list(rec.errors)
                        other.flags = list(rec.flags)

            if outcome.succeeded:
                llm_repair_succeeded += 1
                if outcome.from_cache:
                    llm_repair_cached += 1
            else:
                llm_repair_failed += 1
                rec.errors = outcome.remaining_errors or original_errors
                unfixed_errors.append(rec)
                reason = outcome.failure_reason or "unknown"
                if reason == "api_error":
                    llm_repair_api_errors += 1
                elif reason == "truncated" or reason.startswith("truncated"):
                    llm_repair_truncated += 1
                elif reason.startswith("parse_error"):
                    llm_repair_parse_errors += 1
                else:
                    llm_repair_rejected += 1

            if llm_verbose:
                status = "succeeded" if outcome.succeeded else (outcome.failure_reason or "failed")
                print(
                    f"[LLM {repair_index}/{repair_total}] {rec.id} {status} | "
                    f"{format_metrics_line(outcome.metrics, cached=outcome.from_cache)}",
                    flush=True,
                )

        if progress_store is not None and llm_verbose:
            print("", flush=True)
            print(format_usage_summary(progress_store.usage_totals, model=llm_model), flush=True)

        for slug in processed_slugs:
            registry.mark_gold(slug, run_id=run_id)

    unfixed_errors = [rec for rec in all_records if rec.errors]
    needs_ai_records = [rec for rec in all_records if record_needs_ai(rec)]

    for rec in all_records:
        if rec.topic_confidence < LOW_CONFIDENCE_THRESHOLD or rec.topic == "mixed":
            low_confidence.append(rec)

    save_jsonl(paths.silver_problems_path, all_records)
    save_jsonl(paths.legacy_problems_path, all_records)
    save_jsonl(paths.gold_problems_path, all_records)

    review_dir = paths.review_dir
    with (review_dir / "low_confidence.jsonl").open("w", encoding="utf-8") as f:
        for rec in low_confidence:
            f.write(rec.model_dump_json() + "\n")

    with (review_dir / "unfixed_errors.jsonl").open("w", encoding="utf-8") as f:
        for rec in unfixed_errors:
            f.write(rec.model_dump_json() + "\n")

    with (review_dir / "needs_ai_repair.jsonl").open("w", encoding="utf-8") as f:
        for rec in needs_ai_records:
            f.write(rec.model_dump_json() + "\n")

    with (review_dir / "parse_errors.jsonl").open("w", encoding="utf-8") as f:
        for err in parse_errors:
            f.write(err.model_dump_json() + "\n")

    run_entry = {
        "run_id": run_id,
        "processed_slugs": processed_slugs,
        "incremental": incremental and not full_rebuild,
        "problems_in_batch": len(batch_records),
        "total_problems": len(all_records),
        "records_with_errors": records_with_errors,
        "llm_repair": llm_repair,
    }
    append_run_history(paths.run_history_path, run_entry)

    manifest = Manifest(
        documents_processed=len(processed_slugs),
        problems_extracted=len(all_records),
        skipped_folders=skipped + [f"pdf_only:{s}" for s in pdf_only_slugs],
        parse_errors=parse_errors,
        low_confidence_count=len(low_confidence),
        extra={
            "run_id": run_id,
            "paths": {
                "root": str(paths.root),
                "pdf_dir": str(paths.pdf_dir),
                "bronze_dir": str(paths.bronze_dir),
                "parsed_dir": str(paths.parsed_dir),
                "silver_problems": str(paths.silver_problems_path),
                "gold_problems": str(paths.gold_problems_path),
                "registry": str(paths.registry_path),
            },
            "processed_slugs": processed_slugs,
            "incremental": incremental and not full_rebuild,
            "full_rebuild": full_rebuild,
            "batch_problems": len(batch_records),
            "records_with_errors": records_with_errors,
            "registry_summary": registry.summary(),
            "llm_repair_attempted": llm_repair_attempted,
            "llm_repair_succeeded": llm_repair_succeeded,
            "llm_repair_failed": llm_repair_failed,
            "llm_repair_cached": llm_repair_cached,
            "llm_repair_api_errors": llm_repair_api_errors,
            "llm_repair_parse_errors": llm_repair_parse_errors,
            "llm_repair_truncated": llm_repair_truncated,
            "llm_repair_rejected": llm_repair_rejected,
            "llm_repair_skipped_duplicates": llm_repair_skipped_duplicates,
            "symbol_restore_attempted": symbol_restore_attempted,
            "symbol_restore_succeeded": symbol_restore_succeeded,
            "deterministic_symbol_fixed": deterministic_symbol_fixed,
            "needs_ai_repair_count": len(needs_ai_records),
            "llm_timeout_s": llm_timeout_s if llm_repair else None,
            "llm_max_tokens": llm_max_tokens if llm_repair else None,
            "llm_progress_file": str(llm_cache_dir / "repair_progress.json")
            if llm_repair
            else None,
            "llm_usage_totals": progress_store.usage_totals if progress_store else None,
        },
    )
    paths.manifest_path.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sync_catalog(paths)
    return manifest
