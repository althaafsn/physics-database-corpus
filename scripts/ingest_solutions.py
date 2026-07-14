#!/usr/bin/env python3
"""Ingest all_pdf/solutions/*.pdf into parsed/solutions/solutions.jsonl.

Pipeline per PDF: doc-type filter -> Marker (typed) or vision transcription
(handwriting) -> split into per-problem segments -> align to gold problem_id
-> safety gate -> SolutionRecord.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_loader import load_local_env

load_local_env(ROOT)

from src.text.clean import clean_text
from src.solutions.typed_markdown import resolve_typed_solution_markdown
from src.paths import PipelinePaths
from src.record_store import load_jsonl
from src.solutions.align import AlignResult, GoldIndex, align_solution
from src.solutions.classify_doc_type import classify_doc_type
from src.solutions.filename_meta import parse_solution_filename
from src.solutions.quality import parse_solution_quality
from src.solutions.safety_gate import solution_passes_safety_gate
from src.solutions.schema import SkippedSolutionDoc, SolutionRecord, SolutionSource
from src.solutions.split import split_solution_markdown
from src.solutions.store import (
    load_solutions,
    save_solutions,
    solutions_bronze_dir,
    solutions_jsonl_path,
)
from src.solutions.vision_transcribe import transcribe_pdf


def _log(msg: str) -> None:
    print(msg, flush=True)


def _typed_markdown(pdf_path: Path, bronze_dir: Path, *, force: bool) -> tuple[str, str | None]:
    md_text, method, error = resolve_typed_solution_markdown(
        pdf_path, bronze_dir, force=force
    )
    if error:
        return "", error
    _log(f"  text source: {method}")
    return md_text, None


def ingest_one(
    pdf_path: Path,
    *,
    gold_index: GoldIndex,
    bronze_dir: Path,
    force: bool,
) -> tuple[list[SolutionRecord], SkippedSolutionDoc | None, list[dict[str, object]]]:
    meta = parse_solution_filename(pdf_path.name)
    _log(f"[{pdf_path.name}] level={meta.level} year={meta.year} handwriting={meta.is_handwriting}")

    method = "typed"
    md_text = ""
    if not meta.is_handwriting:
        md_text, error = _typed_markdown(pdf_path, bronze_dir, force=force)
        if error:
            return [], SkippedSolutionDoc(pdf=str(pdf_path), reason="marker_failed", detail=error), []

    needs_vision = meta.is_handwriting or len(md_text.split()) < 30
    if needs_vision:
        method = "handwriting_vision"
        md_text, insufficient = transcribe_pdf(pdf_path, log=_log)
        if insufficient:
            return [], SkippedSolutionDoc(
                pdf=str(pdf_path),
                reason="vision_model_insufficient",
                detail=(
                    f"SOLUTION_VISION_MODEL={os.environ.get('SOLUTION_VISION_MODEL', 'moondream')} "
                    "could not reliably transcribe this scanned/handwritten PDF - needs a stronger "
                    "vision-capable model before this document can be ingested"
                ),
            ), []
        if not md_text.strip():
            return [], SkippedSolutionDoc(
                pdf=str(pdf_path), reason="vision_transcription_empty", detail="no legible pages"
            ), []

    doc_type = classify_doc_type(pdf_path.name, md_text)
    if not doc_type.is_solution:
        return [], SkippedSolutionDoc(
            pdf=str(pdf_path),
            reason="not_a_solution_document",
            detail=doc_type.reason,
        ), []

    segments = split_solution_markdown(md_text)
    if not segments:
        return [], SkippedSolutionDoc(
            pdf=str(pdf_path), reason="split_failed", detail="no confident numbered segments found"
        ), []

    records: list[SolutionRecord] = []
    unmatched: list[dict[str, object]] = []
    for problem_number, body in segments:
        body = clean_text(body)
        ok, gate_reason = solution_passes_safety_gate(body)
        quality = parse_solution_quality(body)
        result: AlignResult = align_solution(
            gold_index,
            level=meta.level,
            year=meta.year,
            problem_number=problem_number,
            variant_hint=meta.variant_hint,
            round_hint=meta.round_hint,
            solution_body=body,
        )
        if result.problem_id is None:
            unmatched.append(
                {
                    "pdf": str(pdf_path.resolve()),
                    "level": meta.level,
                    "year": meta.year,
                    "solution_number": problem_number,
                    "flags": list(result.flags),
                }
            )
            continue

        errors: list[str] = []
        flags = list(result.flags)
        if not ok:
            errors.append(gate_reason or "safety_gate_rejected")
        for quality_error in quality.errors:
            if quality_error not in errors:
                errors.append(quality_error)

        records.append(
            SolutionRecord(
                problem_id=result.problem_id,
                document_slug=meta.slug,
                level=meta.level,
                year=meta.year,
                solution_number=problem_number,
                body_md=body if ok else "",
                method=method,
                source=SolutionSource(pdf=str(pdf_path.resolve())),
                alignment_method=result.method,
                alignment_confidence=result.confidence,
                flags=flags,
                errors=errors,
                llm_model=None,
                steps=quality.steps,
                formatting_confidence=quality.formatting_confidence,
            )
        )
    return records, None, unmatched


def _append_unmatched(unmatched_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    unmatched_path.parent.mkdir(parents=True, exist_ok=True)
    with unmatched_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest worked-solution PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N PDFs")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated PDF filenames")
    parser.add_argument("--force", action="store_true", help="Re-run Marker even if bronze exists")
    args = parser.parse_args()

    paths = PipelinePaths.resolve(ROOT)
    solutions_dir = ROOT / "all_pdf" / "solutions"
    bronze_dir = solutions_bronze_dir(ROOT)
    bronze_dir.mkdir(parents=True, exist_ok=True)

    gold_records = load_jsonl(paths.gold_problems_path, lenient=True)
    gold_index = GoldIndex(gold_records)
    _log(f"Loaded {len(gold_records)} gold problems for alignment")

    pdfs = sorted(solutions_dir.glob("*.pdf"))
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        pdfs = [p for p in pdfs if p.name in wanted]
    if args.limit is not None:
        pdfs = pdfs[: args.limit]

    _log(f"Found {len(pdfs)} solution PDFs to process")

    out_path = solutions_jsonl_path(paths.parsed_dir)
    skipped_path = out_path.parent / "skipped.jsonl"
    unmatched_path = out_path.parent / "unmatched.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # CPU Marker conversion can take tens of minutes per PDF (documented in
    # scripts/convert_pending_cpu.sh) - persist after EVERY document instead
    # of only at the end, so a long batch run is resumable and partial
    # progress is usable immediately by downstream steps.
    existing_all = load_solutions(out_path)
    existing_skipped = (
        [json_line for json_line in skipped_path.read_text(encoding="utf-8").splitlines() if json_line.strip()]
        if skipped_path.is_file()
        else []
    )

    def _persist(new_records: list[SolutionRecord], touched_pdf: Path, skip: SkippedSolutionDoc | None) -> None:
        nonlocal existing_all, existing_skipped
        touched_key = str(touched_pdf.resolve())
        if new_records or skip is None:
            existing_all = [r for r in existing_all if r.source.pdf != touched_key] + new_records
            save_solutions(out_path, existing_all)
        touched_pdf_str = str(touched_pdf)
        existing_skipped = [
            line
            for line in existing_skipped
            if json.loads(line).get("pdf") not in (touched_key, touched_pdf_str)
        ]
        if skip is not None:
            existing_skipped.append(skip.model_dump_json())
        if existing_skipped:
            skipped_path.write_text("\n".join(existing_skipped) + "\n", encoding="utf-8")
        elif skipped_path.is_file():
            skipped_path.unlink()

    if unmatched_path.is_file() and not args.force:
        pass  # keep prior unmatched log across runs
    elif args.force and unmatched_path.is_file():
        unmatched_path.unlink()

    total_new = 0
    total_skipped = 0
    for i, pdf_path in enumerate(pdfs, start=1):
        _log(f"\n=== [{i}/{len(pdfs)}] {pdf_path.name} ===")
        try:
            records, skip, unmatched = ingest_one(
                pdf_path, gold_index=gold_index, bronze_dir=bronze_dir, force=args.force
            )
        except Exception as exc:  # noqa: BLE001 - keep batch going on a single bad PDF
            _log(f"  ✗ unexpected error: {exc}")
            skip = SkippedSolutionDoc(pdf=str(pdf_path), reason="exception", detail=str(exc))
            records = []
            unmatched = []
        if skip:
            _log(f"  skipped: {skip.reason} ({skip.detail})")
            total_skipped += 1
            _persist([], pdf_path, skip)
            _log(f"  [checkpoint saved -> {out_path}]")
            continue

        aligned = sum(1 for r in records if r.alignment_method == "exact")
        ambiguous = sum(1 for r in records if r.alignment_method == "ambiguous")
        rejected = sum(1 for r in records if r.errors)
        if unmatched:
            _log(f"  ! {len(unmatched)} segments had no gold problem match")
        _log(f"  -> {len(records)} solutions ({aligned} exact, {ambiguous} ambiguous, {rejected} gate-rejected)")
        total_new += len(records)
        _append_unmatched(unmatched_path, unmatched)
        _persist(records, pdf_path, skip)
        _log(f"  [checkpoint saved -> {out_path}]")

    review_count = sum(1 for r in existing_all if r.needs_review)
    _log(
        f"\nDone. {total_new} new solution records this run "
        f"({len(existing_all)} total in {out_path}, {review_count} flagged for review)\n"
        f"  {total_skipped} PDFs skipped this run -> {skipped_path}\n"
        f"  unmatched segments -> {unmatched_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
