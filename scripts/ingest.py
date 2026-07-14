#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bronze.bronze_convert import (
    convert_pending_pdfs,
    list_pending_marker_pdfs,
)
from src.env_loader import load_local_env
from src.ingest.ingest_registry import IngestRegistryStore, IngestStage
from src.paths import PipelinePaths
from src.pipeline import run_pipeline
from src.record_store import migrate_legacy_problems
from src.schema import MetadataOverrides
from src.translate.translate_pipeline import run_translate_pipeline


def _silver_index(records) -> dict[str, list]:
    index: dict[str, list] = {}
    for rec in records:
        index.setdefault(rec.document_slug, []).append(rec)
    return index


def cmd_status(paths: PipelinePaths) -> int:
    paths.ensure_dirs()
    store = IngestRegistryStore(paths.registry_path)
    silver = migrate_legacy_problems(paths.silver_problems_path, paths.legacy_problems_path)
    store.scan_paths(paths, silver_by_slug=_silver_index(silver))
    summary = store.summary()
    print(json.dumps(summary, indent=2))
    print("")
    pending = store.pending_silver_slugs()
    if pending:
        print(f"Pending silver ({len(pending)}):")
        for slug in pending[:20]:
            entry = store.get(slug)
            print(f"  - {slug} [{entry.stage if entry else '?'}]")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
    else:
        print("No documents pending silver processing.")
    pdf_only = [
        slug
        for slug, entry in store.documents.items()
        if entry.stage == IngestStage.PDF_ONLY
    ]
    pending_marker = list_pending_marker_pdfs(paths, registry=store)
    if pending_marker:
        print("")
        print(f"Need Marker bronze ({len(pending_marker)}):")
        for slug, _ in pending_marker[:10]:
            print(f"  - {slug}")
        if len(pending_marker) > 10:
            print(f"  ... and {len(pending_marker) - 10} more")
        print("")
        print("Run: python scripts/ingest.py convert --pending")
    elif pdf_only:
        print("")
        print(f"PDF only in registry ({len(pdf_only)}) — rescan after Marker:")
        for slug in pdf_only[:10]:
            print(f"  - {slug}")
    return 0


def cmd_scan(paths: PipelinePaths) -> int:
    paths.ensure_dirs()
    store = IngestRegistryStore(paths.registry_path)
    silver = migrate_legacy_problems(paths.silver_problems_path, paths.legacy_problems_path)
    changed = store.scan_paths(paths, silver_by_slug=_silver_index(silver))
    print(f"Registry updated: {paths.registry_path}")
    print(json.dumps(store.summary(), indent=2))
    if changed:
        print(f"\n{len(changed)} document(s) flagged for processing.")
    return 0


def cmd_convert(args: argparse.Namespace, paths: PipelinePaths) -> int:
    paths.ensure_dirs()
    store = IngestRegistryStore(paths.registry_path)
    silver = migrate_legacy_problems(paths.silver_problems_path, paths.legacy_problems_path)
    store.scan_paths(paths, silver_by_slug=_silver_index(silver))

    only_slugs: set[str] | None = None
    if args.pdf:
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.is_file():
            print(f"PDF not found: {pdf_path}", file=sys.stderr)
            return 1
        only_slugs = {pdf_path.stem}
    elif args.slug:
        only_slugs = {args.slug}

    pending = list_pending_marker_pdfs(paths, registry=store, only_slugs=only_slugs)
    if args.max is not None:
        pending = pending[: args.max]

    if not pending:
        print("No PDFs pending Marker conversion.")
        return 0

    print(f"Marker conversion queue: {len(pending)} PDF(s)")
    for slug, pdf_path in pending:
        print(f"  - {slug} ({pdf_path.name})")

    if args.dry_run:
        print("")
        print("Dry run — no conversion performed.")
        if args.then_process:
            print("After real convert, run:")
            print("  python scripts/ingest.py scan")
            print("  python scripts/ingest.py process")
        return 0

    results = convert_pending_pdfs(
        paths,
        pending,
        timeout_s=args.timeout,
        log=print if not args.quiet else None,
    )
    ok = sum(1 for r in results if r.ok)
    failed = [r for r in results if not r.ok]
    print("")
    print(f"Marker done: {ok}/{len(results)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for r in failed:
            print(f"  - {r.slug}: {r.detail}")

    store.scan_paths(paths, silver_by_slug=_silver_index(silver))
    pending_silver = store.pending_silver_slugs()
    print(f"Bronze ready for silver: {len(pending_silver)}")

    if args.then_process and pending_silver:
        print("")
        print("Running silver extraction for bronze-ready documents...")
        process_args = argparse.Namespace(
            pdf=None,
            slug=None,
            full=False,
            llm_repair=False,
            llm_reset_progress=False,
            level=None,
            year=None,
            round=None,
            variant=None,
            title=None,
        )
        return cmd_process(process_args, paths)

    if ok and not args.then_process:
        print("")
        print("Next:")
        print("  python scripts/ingest.py scan")
        print("  python scripts/ingest.py process")

    return 1 if failed else 0


def cmd_process(args: argparse.Namespace, paths: PipelinePaths) -> int:
    only_slugs: set[str] | None = None
    if args.pdf:
        pdf_path = Path(args.pdf).resolve()
        if not pdf_path.is_file():
            print(f"PDF not found: {pdf_path}", file=sys.stderr)
            return 1
        only_slugs = {pdf_path.stem}
    elif args.slug:
        only_slugs = {args.slug}

    overrides = None
    if args.level or args.year or args.round or args.variant or args.title:
        overrides = MetadataOverrides(
            level=args.level.upper() if args.level else None,
            year=args.year,
            round=args.round,
            variant=args.variant,
            title=args.title,
        )

    incremental = not args.full and not args.pdf and not args.slug and not args.repair_all
    use_llm_topics = args.use_llm_topics or os.environ.get("PHYSICS_USE_LLM_TOPICS", "").lower() in {
        "1",
        "true",
        "yes",
    }
    llm_model = (
        os.environ.get("LLM_REPAIR_MODEL")
        or os.environ.get("HALLIDAY_TAG_MODEL")
        or (
            "qwen2.5:3b"
            if os.environ.get("LLM_PROVIDER", "").strip().lower() in {"local", "ollama"}
            or os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
            else "qwen3.6-35b"
        )
    )
    manifest = run_pipeline(
        paths,
        only_slugs=only_slugs,
        incremental=incremental,
        full_rebuild=args.full,
        metadata_overrides=overrides,
        use_llm_topics=use_llm_topics,
        llm_repair=args.llm_repair or args.repair_all,
        llm_reset_progress=args.llm_reset_progress,
        repair_all=args.repair_all,
        llm_model=llm_model,
    )
    extra = manifest.extra
    print(
        f"Run {extra.get('run_id')}: batch={extra.get('batch_problems')} problems, "
        f"corpus total={manifest.problems_extracted}, "
        f"errors={extra.get('records_with_errors')}"
    )
    if extra.get("processed_slugs"):
        print("Processed:", ", ".join(extra["processed_slugs"]))
    return 0


def cmd_translate(args: argparse.Namespace, paths: PipelinePaths) -> int:
    ids = {part.strip() for part in args.ids.split(",") if part.strip()} if args.ids else None
    slugs = {args.slug} if args.slug else None
    try:
        summary = run_translate_pipeline(
            paths,
            ids=ids,
            slugs=slugs,
            catalog_only=not args.all_gold,
            force=args.force,
            limit=args.limit,
            reset_progress=args.llm_reset_progress,
            dry_run=args.dry_run,
            sync_catalog_after=not args.no_sync_catalog,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in summary.items() if k != "usage_summary"}, indent=2))
    usage = summary.get("usage_summary")
    if usage:
        print("")
        print(usage)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incremental ingest for physics problem sets (PDF → bronze → silver)."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: cwd or PHYSICS_DB_ROOT)",
    )
    parser.add_argument("--pdf-dir", type=Path, default=None, help="Raw PDF directory")
    parser.add_argument("--bronze-dir", type=Path, default=None, help="Marker output directory")
    parser.add_argument("--parsed-dir", type=Path, default=None, help="Parsed output directory")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show ingest registry summary")
    sub.add_parser("scan", help="Scan PDF/bronze dirs and update registry")

    convert = sub.add_parser("convert", help="Run Marker PDF → bronze markdown")
    convert.add_argument(
        "--pending",
        action="store_true",
        help="Convert all PDFs missing bronze markdown (default when no --pdf/--slug)",
    )
    convert.add_argument("--pdf", type=Path, help="Convert one PDF by path")
    convert.add_argument("--slug", help="Convert one PDF by document slug")
    convert.add_argument("--max", type=int, help="Convert at most N pending PDFs")
    convert.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-PDF timeout in seconds (default: none)",
    )
    convert.add_argument("--dry-run", action="store_true", help="List queue only")
    convert.add_argument(
        "--then-process",
        action="store_true",
        help="After conversion, run silver extraction on bronze-ready docs",
    )
    convert.add_argument("--quiet", action="store_true", help="Less progress output")

    process = sub.add_parser("process", help="Run silver extraction pipeline")
    process.add_argument("--pdf", type=Path, help="Process a single PDF (by path)")
    process.add_argument("--slug", help="Process one document by folder/PDF stem name")
    process.add_argument(
        "--full",
        action="store_true",
        help="Reprocess all bronze folders (merge into existing corpus)",
    )
    process.add_argument(
        "--use-llm-topics",
        action="store_true",
        help="LLM tie-break for mixed/low-confidence coarse topics",
    )
    process.add_argument(
        "--llm-repair",
        action="store_true",
        help="Run LLM repair on processed records with errors",
    )
    process.add_argument(
        "--repair-all",
        action="store_true",
        help="LLM-repair every record with errors in the corpus (no bronze re-extract)",
    )
    process.add_argument("--llm-reset-progress", action="store_true")
    process.add_argument("--level", help="Override olympiad level (OSK/OSP/OSN)")
    process.add_argument("--year", type=int, help="Override competition year")
    process.add_argument("--round", help="Override round (final/semifinal)")
    process.add_argument("--variant", type=int, help="Override variant number")
    process.add_argument("--title", help="Override document title metadata")

    translate = sub.add_parser("translate", help="Translate gold problems to English via LLM")
    translate.add_argument("--ids", help="Comma-separated problem IDs")
    translate.add_argument("--slug", help="Translate one document slug")
    translate.add_argument(
        "--all-gold",
        action="store_true",
        help="Include ineligible gold records (default: catalog-eligible only)",
    )
    translate.add_argument("--force", action="store_true", help="Retranslate existing English fields")
    translate.add_argument("--limit", type=int, help="Translate at most N problems")
    translate.add_argument("--dry-run", action="store_true", help="List targets only")
    translate.add_argument("--no-sync-catalog", action="store_true")
    translate.add_argument("--llm-reset-progress", action="store_true")

    return parser


def main() -> int:
    load_local_env()
    parser = build_parser()
    args = parser.parse_args()

    paths = PipelinePaths.resolve(args.root)
    if args.pdf_dir:
        paths = PipelinePaths(
            root=paths.root,
            pdf_dir=args.pdf_dir.resolve(),
            bronze_dir=(args.bronze_dir or paths.bronze_dir).resolve(),
            parsed_dir=(args.parsed_dir or paths.parsed_dir).resolve(),
        )
    elif args.bronze_dir or args.parsed_dir:
        paths = PipelinePaths(
            root=paths.root,
            pdf_dir=paths.pdf_dir,
            bronze_dir=(args.bronze_dir or paths.bronze_dir).resolve(),
            parsed_dir=(args.parsed_dir or paths.parsed_dir).resolve(),
        )

    if args.command == "status":
        return cmd_status(paths)
    if args.command == "scan":
        return cmd_scan(paths)
    if args.command == "convert":
        if not (args.pdf or args.slug or getattr(args, "pending", False)):
            args.pending = True
        return cmd_convert(args, paths)
    if args.command == "process":
        return cmd_process(args, paths)
    if args.command == "translate":
        return cmd_translate(args, paths)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
