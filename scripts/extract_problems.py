#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_loader import load_local_env
from src.llm.llm_client import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TIMEOUT_S
from src.llm.llm_progress import format_usage_summary
from src.paths import PipelinePaths
from src.pipeline import run_pipeline
from src.schema import MetadataOverrides


def main() -> None:
    load_local_env()

    parser = argparse.ArgumentParser(
        description="Extract per-problem records from marker markdown output."
    )
    parser.add_argument("--root", type=Path, default=None, help="Project root directory")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Bronze directory (Marker output); default: output/",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing source PDF files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Parsed output directory",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Process only this PDF (incremental)",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Process only this document slug (PDF/bronze folder stem)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Reprocess all bronze folders and merge into corpus",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Process all bronze-ready documents not just pending registry entries",
    )
    parser.add_argument("--level", help="Override olympiad level (OSK/OSP/OSN)")
    parser.add_argument("--year", type=int, help="Override competition year")
    parser.add_argument("--round", help="Override round (final/semifinal)")
    parser.add_argument("--variant", type=int, help="Override variant number")
    parser.add_argument("--title", help="Override document title metadata")
    parser.add_argument(
        "--use-llm-topics",
        action="store_true",
        help="Use LLM for ambiguous topic classification (requires NETRA_API_KEY)",
    )
    parser.add_argument(
        "--llm-repair",
        action="store_true",
        help="Repair records with parse errors via Netra LLM (requires NETRA_API_KEY)",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_MODEL,
        help=f"Netra model for parse repair (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--llm-cache-dir",
        type=Path,
        default=None,
        help="Cache directory for LLM repair responses and progress",
    )
    parser.add_argument(
        "--llm-reset-progress",
        action="store_true",
        help="Clear parsed/llm_cache/repair_progress.json before running repair",
    )
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Netra HTTP timeout per attempt in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max completion tokens per repair call (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--llm-quiet",
        action="store_true",
        help="Only print summary lines, not per-phase repair logs",
    )
    args = parser.parse_args()

    paths = PipelinePaths.resolve(args.root)
    if args.input_dir or args.pdf_dir or args.out_dir:
        paths = PipelinePaths(
            root=paths.root,
            pdf_dir=(args.pdf_dir or paths.pdf_dir).resolve(),
            bronze_dir=(args.input_dir or paths.bronze_dir).resolve(),
            parsed_dir=(args.out_dir or paths.parsed_dir).resolve(),
        )

    only_slugs: set[str] | None = None
    if args.pdf:
        only_slugs = {Path(args.pdf).resolve().stem}
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

    incremental = not args.full and not args.no_incremental and only_slugs is None

    manifest = run_pipeline(
        paths,
        only_slugs=only_slugs,
        incremental=incremental,
        full_rebuild=args.full,
        metadata_overrides=overrides,
        use_llm_topics=args.use_llm_topics,
        llm_repair=args.llm_repair,
        llm_model=args.llm_model,
        llm_cache_dir=args.llm_cache_dir or paths.llm_cache_dir,
        llm_reset_progress=args.llm_reset_progress,
        llm_timeout_s=args.llm_timeout,
        llm_max_tokens=args.llm_max_tokens,
        llm_verbose=not args.llm_quiet,
    )
    extra = manifest.extra
    msg = (
        f"Done: {manifest.problems_extracted} problems in corpus "
        f"(batch={extra.get('batch_problems', 0)} from {len(extra.get('processed_slugs', []))} docs, "
        f"{manifest.low_confidence_count} low-confidence, "
        f"{len(manifest.parse_errors)} document errors, "
        f"{extra.get('records_with_errors', 0)} with parse errors"
    )
    if args.llm_repair:
        msg += (
            f", {extra.get('llm_repair_succeeded', 0)} LLM repaired"
            f" ({extra.get('llm_repair_cached', 0)} from cache)"
            f", {extra.get('llm_repair_failed', 0)} failed"
            f" (api={extra.get('llm_repair_api_errors', 0)},"
            f" parse={extra.get('llm_repair_parse_errors', 0)},"
            f" truncated={extra.get('llm_repair_truncated', 0)},"
            f" rejected={extra.get('llm_repair_rejected', 0)})"
        )
    msg += f"). Registry: {paths.registry_path}"
    print(msg)
    if args.llm_repair:
        usage = extra.get("llm_usage_totals")
        if usage:
            print(format_usage_summary(usage, model=args.llm_model))


if __name__ == "__main__":
    main()
