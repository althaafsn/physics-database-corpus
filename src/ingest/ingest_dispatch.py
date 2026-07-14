"""Route ingest jobs to local worker thread or S3+SQS+Airflow queue."""
from __future__ import annotations

from collections.abc import Callable

from src.airflow.airflow_trigger import trigger_ingest_dag
from src.ingest.ingest_batch import PdfUploadItem
from src.ingest.ingest_jobs import IngestJob, IngestJobStage, IngestJobStatus, IngestJobStore, enqueue_ingest
from src.ingest.ingest_one import run_single_pdf_ingest
from src.ingest.ingest_queue import enqueue_ingest_message, message_from_job, queue_ingest_enabled, upload_pdf_bytes
from src.paths import PipelinePaths


def dispatch_queued_pdf(
    job: IngestJob,
    *,
    pdf_bytes: bytes,
    store: IngestJobStore,
    batch_index: int | None = None,
    trigger_airflow: bool = True,
) -> None:
    s3_key = upload_pdf_bytes(job_id=job.id, slug=job.slug, pdf_bytes=pdf_bytes)
    enqueue_ingest_message(message_from_job(job, s3_key=s3_key, batch_index=batch_index))
    store.update(
        job.id,
        status=IngestJobStatus.QUEUED,
        stage=IngestJobStage.QUEUED,
        detail=f"Queued on S3 ({s3_key})",
    )
    try:
        if trigger_airflow:
            run_id = trigger_ingest_dag(reason="upload", job_id=job.id)
            if run_id:
                store.update(job.id, detail=f"Queued; Airflow run {run_id}")
    except Exception as exc:  # noqa: BLE001 — queue already durable; don't fail upload
        store.update(job.id, detail=f"Queued on S3; Airflow trigger failed: {exc}")


def dispatch_single_pdf_ingest(
    job: IngestJob,
    *,
    pdf_bytes: bytes,
    paths: PipelinePaths,
    store: IngestJobStore,
) -> None:
    if queue_ingest_enabled():
        dispatch_queued_pdf(job, pdf_bytes=pdf_bytes, store=store)
        return

    def runner(j: IngestJob) -> None:
        run_single_pdf_ingest(j, pdf_bytes=pdf_bytes, paths=paths, store=store)

    enqueue_ingest(store, job, runner)


def dispatch_batch_pdf_ingest(
    job: IngestJob,
    *,
    files: list[PdfUploadItem],
    paths: PipelinePaths,
    store: IngestJobStore,
    run_batch: Callable[[IngestJob], None],
) -> None:
    if not queue_ingest_enabled():
        enqueue_ingest(store, job, run_batch)
        return

    from src.ingest.ingest_one import resolve_upload_slug, slugify_upload_name

    job.kind = "batch"
    job.total_files = len(files)
    store.save(job)

    for index, item in enumerate(files, start=1):
        child = store.create(
            slug=resolve_upload_slug(paths, slugify_upload_name(item.filename)),
            pdf_filename=item.filename,
            content_locale_hint=job.content_locale_hint,
            publish=job.publish,
            kind="single",
            parent_id=job.id,
        )
        dispatch_queued_pdf(
            child, pdf_bytes=item.data, store=store, batch_index=index, trigger_airflow=False
        )
        job.completed_files = index - 1
        job.current_file_index = index
        store.save(job)

    store.update(
        job.id,
        status=IngestJobStatus.QUEUED,
        stage=IngestJobStage.QUEUED,
        detail=f"Queued {len(files)} PDFs on S3",
    )
    try:
        trigger_ingest_dag(reason="batch_upload", job_id=job.id)
    except Exception:  # noqa: BLE001 — children already queued on S3/SQS
        pass
