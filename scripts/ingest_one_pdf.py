#!/usr/bin/env python3
"""CLI wrapper for single-PDF ingest (same path as admin upload API)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingest.ingest_jobs import IngestJobStore, jobs_dir_for
from src.ingest.ingest_one import run_single_pdf_ingest, slugify_upload_name
from src.paths import PipelinePaths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to problem PDF")
    parser.add_argument("--locale", choices=["auto", "id", "en"], default="auto")
    parser.add_argument("--publish", action="store_true", help="Run npm export:data after ingest")
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        print(f"Not found: {pdf_path}", file=sys.stderr)
        return 1

    paths = PipelinePaths.resolve()
    store = IngestJobStore(jobs_dir_for(paths))
    job = store.create(
        slug=slugify_upload_name(pdf_path.name),
        pdf_filename=pdf_path.name,
        content_locale_hint=args.locale,
        publish=args.publish,
    )
    result = run_single_pdf_ingest(job, pdf_bytes=pdf_path.read_bytes(), paths=paths, store=store)
    print(result.to_dict() if hasattr(result, "to_dict") else result)
    return 0 if result.status.value == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
