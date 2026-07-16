#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_loader import load_local_env
from src.llm.llm_client import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TIMEOUT_S
from src.paths import PipelinePaths
from src.translate.translate_pipeline import run_translate_pipeline


def _parse_ids(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return ids or None


def main() -> int:
    load_local_env()

    parser = argparse.ArgumentParser(
        description="Translate Indonesian physics problems to English via Netra LLM."
    )
    parser.add_argument("--root", type=Path, default=None, help="Project root directory")
    parser.add_argument(
        "--parsed-dir",
        type=Path,
        default=None,
        help="Parsed output directory (default: parsed/)",
    )
    parser.add_argument(
        "--ids",
        help="Comma-separated problem IDs (e.g. OSK-2003-01,OSK-2003-02)",
    )
    parser.add_argument(
        "--slug",
        help="Translate all eligible problems from one document slug",
    )
    parser.add_argument(
        "--all-gold",
        action="store_true",
        help="Include ineligible gold records (default: catalog-eligible only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retranslate even if English fields already exist",
    )
    parser.add_argument("--limit", type=int, help="Translate at most N problems")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent API requests (default: 1)")
    parser.add_argument("--dry-run", action="store_true", help="List targets without calling LLM")
    parser.add_argument(
        "--no-sync-catalog",
        action="store_true",
        help="Skip refreshing parsed/catalog/problems.jsonl after translation",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Clear parsed/llm_cache/translate_progress.json before running",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Netra model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"HTTP timeout per attempt in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max completion tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    args = parser.parse_args()

    paths = PipelinePaths.resolve(args.root)
    if args.parsed_dir:
        paths = PipelinePaths(
            root=paths.root,
            pdf_dir=paths.pdf_dir,
            bronze_dir=paths.bronze_dir,
            parsed_dir=args.parsed_dir.resolve(),
        )

    ids = _parse_ids(args.ids)
    slugs = {args.slug} if args.slug else None

    try:
        summary = run_translate_pipeline(
            paths,
            ids=ids,
            slugs=slugs,
            catalog_only=not args.all_gold,
            force=args.force,
            limit=args.limit,
            workers=max(1, args.workers),
            model=args.model,
            timeout_s=args.timeout,
            max_tokens=args.max_tokens,
            reset_progress=args.reset_progress,
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


if __name__ == "__main__":
    raise SystemExit(main())
