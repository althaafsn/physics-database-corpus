#!/usr/bin/env python3
"""Backfill language metadata on records created before locale detection existed."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog import sync_catalog
from src.paths import PipelinePaths
from src.record_store import load_jsonl, save_jsonl
from src.schema import ProblemRecord
from src.text.detect_language import detect_content_locale


def backfill_records(
    records: list[ProblemRecord],
) -> tuple[list[ProblemRecord], list[tuple[str, str, str]]]:
    updated: list[ProblemRecord] = []
    changes: list[tuple[str, str, str]] = []
    for record in records:
        detected = detect_content_locale(
            record.body_md,
            slug=record.document_slug,
            filename=Path(record.source.pdf).name,
        )
        if detected == record.content_locale:
            updated.append(record)
            continue
        changed = record.model_copy(deep=True)
        changes.append((record.id, record.content_locale, detected))
        changed.content_locale = detected
        updated.append(changed)
    return updated, changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true", help="Write silver/gold and refresh catalog")
    args = parser.parse_args()
    paths = PipelinePaths.resolve(args.root)

    result: dict[str, object] = {"apply": args.apply, "files": {}}
    for name, path in (
        ("silver", paths.silver_problems_path),
        ("gold", paths.gold_problems_path),
    ):
        records = load_jsonl(path, lenient=True)
        updated, changes = backfill_records(records)
        result["files"][name] = {
            "total": len(records),
            "changed": len(changes),
            "ids": [problem_id for problem_id, _before, _after in changes],
        }
        if args.apply and changes:
            save_jsonl(path, updated)

    if args.apply:
        result["catalog"] = sync_catalog(paths)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
