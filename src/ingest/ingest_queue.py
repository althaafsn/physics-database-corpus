"""S3 landing + SQS enqueue for queue-based ingest (Airflow / Batch worker)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from src.ingest.ingest_jobs import IngestJob


@dataclass(frozen=True)
class IngestQueueMessage:
    job_id: str
    slug: str
    pdf_filename: str
    s3_key: str
    content_locale_hint: str = "auto"
    publish: bool = False
    kind: str = "single"
    parent_id: str | None = None
    batch_index: int | None = None
    batch_total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def queue_ingest_enabled() -> bool:
    return os.environ.get("INGEST_MODE", "local").strip().lower() == "queue"


def _bucket() -> str:
    bucket = os.environ.get("CORPUS_S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("CORPUS_S3_BUCKET is required when INGEST_MODE=queue")
    return bucket


def _queue_url() -> str:
    url = os.environ.get("INGEST_SQS_QUEUE_URL", "").strip()
    if not url:
        raise RuntimeError("INGEST_SQS_QUEUE_URL is required when INGEST_MODE=queue")
    return url


def raw_pdf_s3_key(job_id: str, slug: str) -> str:
    safe_slug = slug.replace("/", "_")
    return f"raw/pdfs/{job_id}/{safe_slug}.pdf"


def upload_pdf_bytes(*, job_id: str, slug: str, pdf_bytes: bytes) -> str:
    import boto3

    key = raw_pdf_s3_key(job_id, slug)
    boto3.client("s3").put_object(
        Bucket=_bucket(),
        Key=key,
        Body=pdf_bytes,
        ContentType="application/pdf",
    )
    return key


def enqueue_ingest_message(message: IngestQueueMessage) -> None:
    import boto3

    boto3.client("sqs").send_message(
        QueueUrl=_queue_url(),
        MessageBody=message.to_json(),
    )


def sqs_approximate_depth() -> int:
    import boto3

    attrs = boto3.client("sqs").get_queue_attributes(
        QueueUrl=_queue_url(),
        AttributeNames=["ApproximateNumberOfMessages"],
    )
    return int(attrs.get("Attributes", {}).get("ApproximateNumberOfMessages", "0"))


def message_from_job(job: IngestJob, *, s3_key: str, batch_index: int | None = None) -> IngestQueueMessage:
    return IngestQueueMessage(
        job_id=job.id,
        slug=job.slug,
        pdf_filename=job.pdf_filename,
        s3_key=s3_key,
        content_locale_hint=job.content_locale_hint,
        publish=job.publish,
        kind=job.kind,
        parent_id=job.parent_id,
        batch_index=batch_index,
        batch_total=job.total_files or None,
    )
