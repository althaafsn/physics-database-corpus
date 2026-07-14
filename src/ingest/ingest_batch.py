"""Batch and ZIP PDF ingest for the admin API."""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

from src.ingest.ingest_jobs import IngestJob, IngestJobStage, IngestJobStatus, IngestJobStore
from src.ingest.ingest_one import export_static_site, run_single_pdf_ingest, slugify_upload_name
from src.paths import PipelinePaths

_MAX_BATCH_FILES = 30
_MAX_ZIP_BYTES = 200 * 1024 * 1024


@dataclass
class PdfUploadItem:
    filename: str
    data: bytes


def extract_pdfs_from_zip(data: bytes) -> list[PdfUploadItem]:
    if len(data) > _MAX_ZIP_BYTES:
        raise ValueError(f"ZIP too large (max {_MAX_ZIP_BYTES // (1024 * 1024)} MB)")
    items: list[PdfUploadItem] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            if not name.lower().endswith(".pdf"):
                continue
            if name.startswith("._"):
                continue
            payload = archive.read(info)
            if payload:
                items.append(PdfUploadItem(filename=name, data=payload))
    if not items:
        raise ValueError("ZIP contains no PDF files")
    if len(items) > _MAX_BATCH_FILES:
        raise ValueError(f"Too many PDFs in ZIP (max {_MAX_BATCH_FILES})")
    return items


def validate_pdf_uploads(files: list[PdfUploadItem]) -> list[PdfUploadItem]:
    if not files:
        raise ValueError("No PDF files provided")
    if len(files) > _MAX_BATCH_FILES:
        raise ValueError(f"Too many files (max {_MAX_BATCH_FILES})")
    seen: set[str] = set()
    unique: list[PdfUploadItem] = []
    for item in files:
        key = item.filename.lower()
        if key in seen:
            continue
        seen.add(key)
        if not item.data:
            raise ValueError(f"Empty file: {item.filename}")
        unique.append(item)
    return unique


def run_batch_ingest(
    job: IngestJob,
    *,
    files: list[PdfUploadItem],
    paths: PipelinePaths,
    store: IngestJobStore,
) -> IngestJob:
    job.kind = "batch"
    job.total_files = len(files)
    job.pdf_filename = f"{len(files)} files"
    job.slug = f"batch-{job.id}"
    job.files = [
        {
            "pdf_filename": item.filename,
            "slug": slugify_upload_name(item.filename),
            "status": "pending",
            "content_locale": "id",
            "problems_count": 0,
            "problem_ids": [],
            "error": None,
        }
        for item in files
    ]
    store.save(job)

    succeeded = 0
    all_problem_ids: list[str] = []
    last_catalog_total = 0
    errors: list[str] = []

    for index, item in enumerate(files):
        job.files[index]["status"] = "running"
        job.current_file_index = index
        store.update(job.id, files=job.files, current_file_index=index, stage=IngestJobStage.BRONZE)

        child = store.create(
            slug=slugify_upload_name(item.filename),
            pdf_filename=item.filename,
            content_locale_hint=job.content_locale_hint,
            publish=False,
            kind="single",
            parent_id=job.id,
        )
        result = run_single_pdf_ingest(
            child,
            pdf_bytes=item.data,
            paths=paths,
            store=store,
        )

        entry = job.files[index]
        entry["slug"] = result.slug
        if result.status == IngestJobStatus.SUCCEEDED:
            entry["status"] = "succeeded"
            entry["content_locale"] = result.content_locale
            entry["problems_count"] = result.problems_count
            entry["problem_ids"] = list(result.problem_ids)
            succeeded += 1
            all_problem_ids.extend(result.problem_ids)
            last_catalog_total = max(last_catalog_total, result.catalog_total)
        else:
            entry["status"] = "failed"
            entry["error"] = result.error or "ingest failed"
            errors.append(f"{item.filename}: {entry['error']}")

        job.completed_files = index + 1
        store.update(job.id, files=job.files, completed_files=job.completed_files)

    if job.publish and succeeded > 0:
        store.update(job.id, stage=IngestJobStage.EXPORT)
        try:
            export_static_site(paths)
        except Exception as exc:
            errors.append(f"export: {exc}")

    if succeeded == 0:
        status = IngestJobStatus.FAILED
        error = "; ".join(errors[:3]) if errors else "All files failed"
    else:
        status = IngestJobStatus.SUCCEEDED
        error = "; ".join(errors[:2]) if errors and succeeded < len(files) else None

    detail = f"Batch: {succeeded}/{len(files)} files, {len(all_problem_ids)} problems"
    store.update(
        job.id,
        status=status,
        stage=IngestJobStage.DONE,
        problems_count=len(all_problem_ids),
        problem_ids=all_problem_ids,
        catalog_total=last_catalog_total,
        error=error,
        detail=detail,
    )
    return store.get(job.id) or job
