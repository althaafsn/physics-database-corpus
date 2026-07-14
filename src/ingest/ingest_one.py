"""Run the full problems pipeline for a single uploaded PDF."""
from __future__ import annotations

import re
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from src.bronze.bronze_convert import convert_pdf_to_bronze_for_ingest
from src.catalog import sync_catalog
from src.text.detect_language import detect_content_locale
from src.ingest.ingest_jobs import IngestJob, IngestJobStage, IngestJobStatus, IngestJobStore
from src.ingest.ingest_registry import IngestEntry, IngestRegistryStore, IngestStage, bronze_content_hash, file_hash
from src.paths import PipelinePaths
from src.bronze.pdf_text import extract_pdf_text, strip_pdf_footers
from src.pipeline import run_pipeline
from src.record_store import load_jsonl, save_jsonl


def slugify_upload_name(filename: str) -> str:
    stem = Path(filename).stem.strip()
    stem = re.sub(r"[^\w\s\-+().]", "_", stem, flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem or "upload"


def resolve_upload_slug(paths: PipelinePaths, base: str) -> str:
    """Stable slug from filename — re-uploads replace the same document."""
    return base


def prepare_upload_paths(paths: PipelinePaths, slug: str) -> Path:
    """Overwrite PDF and clear stale bronze so re-ingest is idempotent."""
    pdf_path = paths.pdf_dir / f"{slug}.pdf"
    bronze = paths.bronze_folder(slug)
    if bronze.is_dir():
        shutil.rmtree(bronze)
    return pdf_path


def stamp_content_locale(
    paths: PipelinePaths,
    *,
    document_slug: str,
    content_locale: str,
) -> list[str]:
    records = load_jsonl(paths.gold_problems_path, lenient=True)
    ids: list[str] = []
    changed = False
    for rec in records:
        if rec.document_slug != document_slug:
            continue
        if rec.content_locale != content_locale:
            rec.content_locale = content_locale
            changed = True
        ids.append(rec.id)
    if changed:
        save_jsonl(paths.gold_problems_path, records)
        save_jsonl(paths.silver_problems_path, records)
    return ids


def export_static_site(paths: PipelinePaths) -> None:
    subprocess.run(
        ["npm", "run", "export:data"],
        cwd=paths.root,
        check=True,
        capture_output=True,
        text=True,
    )


def run_single_pdf_ingest(
    job: IngestJob,
    *,
    pdf_bytes: bytes,
    paths: PipelinePaths | None = None,
    store: IngestJobStore | None = None,
) -> IngestJob:
    paths = paths or PipelinePaths.resolve()
    paths.ensure_dirs()
    store = store or IngestJobStore(paths.parsed_dir / "ingest_jobs")

    def progress(stage: IngestJobStage, **extra) -> None:
        store.update(
            job.id,
            status=IngestJobStatus.RUNNING,
            stage=stage,
            **extra,
        )

    try:
        import os

        os.environ.setdefault("PHYSICS_MARKER_CPU", "1")

        progress(IngestJobStage.SAVING)
        slug = resolve_upload_slug(paths, job.slug)
        job.slug = slug
        pdf_path = prepare_upload_paths(paths, slug)
        pdf_path.write_bytes(pdf_bytes)

        sample_text = strip_pdf_footers(extract_pdf_text(pdf_path))
        locale_hint = job.content_locale_hint if job.content_locale_hint != "auto" else None
        content_locale = detect_content_locale(
            sample_text,
            hint=locale_hint,
            slug=slug,
            filename=job.pdf_filename,
        )
        job.content_locale = content_locale
        progress(IngestJobStage.BRONZE, content_locale=content_locale)

        bronze = convert_pdf_to_bronze_for_ingest(
            pdf_path,
            bronze_dir=paths.bronze_dir,
            log=lambda msg: store.update(job.id, detail=msg),
        )
        if not bronze.ok:
            raise RuntimeError(f"Bronze conversion failed ({bronze.detail})")
        bronze_detail = bronze.detail or "bronze"

        registry = IngestRegistryStore(paths.registry_path)
        bronze_folder = paths.bronze_folder(slug)
        registry.upsert(
            IngestEntry(
                slug=slug,
                pdf_path=str(pdf_path.resolve()),
                pdf_hash=file_hash(pdf_path),
                bronze_path=str(bronze_folder.resolve()),
                bronze_hash=bronze_content_hash(bronze_folder),
                stage=IngestStage.BRONZE_READY,
                first_seen_at=datetime.now(UTC).isoformat(),
            )
        )
        registry.save()

        progress(IngestJobStage.PARSING, detail=bronze_detail)
        manifest = run_pipeline(
            paths,
            only_slugs={slug},
            incremental=True,
            llm_repair=False,
        )

        progress(IngestJobStage.TAGGING)
        problem_ids = stamp_content_locale(
            paths,
            document_slug=slug,
            content_locale=content_locale,
        )

        progress(IngestJobStage.CATALOG)
        meta = sync_catalog(paths)
        catalog_total = int(meta.get("catalog_total", 0))

        if job.publish:
            progress(IngestJobStage.EXPORT)
            export_static_site(paths)

        store.update(
            job.id,
            status=IngestJobStatus.SUCCEEDED,
            stage=IngestJobStage.DONE,
            slug=slug,
            content_locale=content_locale,
            problems_count=len(problem_ids),
            problem_ids=problem_ids,
            catalog_total=catalog_total,
            detail=(
                f"Parsed {len(problem_ids)} problems ({content_locale.upper()}). "
                f"Batch: {manifest.problems_extracted} total gold."
            ),
            error=None,
        )
        return store.get(job.id) or job

    except Exception as exc:
        store.update(
            job.id,
            status=IngestJobStatus.FAILED,
            stage=IngestJobStage.DONE,
            error=str(exc),
        )
        return store.get(job.id) or job
