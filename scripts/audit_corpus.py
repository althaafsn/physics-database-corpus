#!/usr/bin/env python3
"""Audit gold corpus: validation errors, images, duplicates. Writes parsed/review/audit.json."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.text.attach_images import extract_image_refs, body_expects_attached_figure
from src.catalog import is_catalog_eligible
from src.paths import PipelinePaths
from src.record_store import load_jsonl
from src.repair.repair_images import needs_image_repair
from src.repair.vision_repair import needs_vision_image_repair


def audit(paths: PipelinePaths) -> dict:
    records = load_jsonl(paths.gold_problems_path, lenient=True)
    error_codes: Counter[str] = Counter()
    unfixed: list[dict] = []
    image_issues: list[dict] = []
    dup_image_refs: list[str] = []
    asset_dupes: dict[str, list[str]] = defaultdict(list)

    for rec in records:
        for err in rec.errors:
            error_codes[err.code] += 1
        if rec.errors:
            unfixed.append(
                {
                    "id": rec.id,
                    "slug": rec.document_slug,
                    "codes": [e.code for e in rec.errors],
                    "catalog_eligible": is_catalog_eligible(rec),
                }
            )

        refs = extract_image_refs(rec.body_md)
        if len(refs) != len(set(refs)):
            dup_image_refs.append(rec.id)

        for img in rec.images:
            if img.path:
                asset_dupes[img.path].append(rec.id)

        if needs_image_repair(rec) or needs_vision_image_repair(rec):
            output = Path(rec.source.md).parent
            marker_count = len(list(output.glob("_page_*"))) if output.is_dir() else 0
            image_issues.append(
                {
                    "id": rec.id,
                    "slug": rec.document_slug,
                    "refs": refs,
                    "images": len(rec.images),
                    "marker_images": marker_count,
                    "flags": [f for f in rec.flags if "image" in f],
                    "errors": [e.code for e in rec.errors if "image" in e.code],
                    "figure_hints": body_expects_attached_figure(rec.body_md),
                }
            )

    shared_assets = {
        path: sorted(set(ids))
        for path, ids in asset_dupes.items()
        if len(set(ids)) > 1
    }

    return {
        "gold_total": len(records),
        "with_errors": len(unfixed),
        "catalog_eligible": sum(1 for r in records if is_catalog_eligible(r)),
        "error_codes": dict(error_codes),
        "unfixed": unfixed,
        "duplicate_image_refs_in_body": dup_image_refs,
        "shared_asset_paths": {k: v[:8] for k, v in list(shared_assets.items())[:50]},
        "shared_asset_path_count": len(shared_assets),
        "image_issues": image_issues,
        "image_issue_count": len(image_issues),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit physics problem corpus")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    paths = PipelinePaths.resolve(args.root or ROOT)
    paths.review_dir.mkdir(parents=True, exist_ok=True)
    report = audit(paths)
    out = paths.review_dir / "audit.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in {"unfixed", "image_issues"}}, indent=2))
    print(f"Wrote {out}")
    print(f"Unfixed: {report['with_errors']} | Image issues: {report['image_issue_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
