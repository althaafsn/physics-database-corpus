"""Live ingest ops summaries (SQS depth, Batch job counts, bronze provenance)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.paths import PipelinePaths


def _validate_slug(slug: str) -> str:
    if not slug or Path(slug).name != slug or slug in {".", ".."}:
        raise ValueError("invalid bronze slug")
    return slug


def read_bronze_source(paths: PipelinePaths, slug: str) -> dict[str, Any] | None:
    slug = _validate_slug(slug)
    path = paths.bronze_folder(slug) / f"{slug}_bronze_source.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def sqs_queue_stats() -> dict[str, Any]:
    """Return SQS depth stats when queue mode is configured; else empty."""
    url = os.environ.get("INGEST_SQS_QUEUE_URL", "").strip()
    if not url:
        return {
            "configured": False,
            "queue_url": None,
            "messages_available": None,
            "messages_in_flight": None,
        }
    try:
        import boto3

        attrs = boto3.client("sqs").get_queue_attributes(
            QueueUrl=url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
            ],
        )["Attributes"]
        return {
            "configured": True,
            "queue_url": url,
            "messages_available": int(attrs.get("ApproximateNumberOfMessages", "0")),
            "messages_in_flight": int(attrs.get("ApproximateNumberOfMessagesNotVisible", "0")),
        }
    except Exception as exc:  # noqa: BLE001 — surface to ops UI
        return {
            "configured": True,
            "queue_url": url,
            "messages_available": None,
            "messages_in_flight": None,
            "error": str(exc),
        }


def batch_queue_stats(*, max_per_status: int = 5) -> dict[str, Any]:
    """Summarize recent AWS Batch jobs for the ingest GPU queue."""
    queue = os.environ.get("BATCH_JOB_QUEUE", "").strip()
    definition = os.environ.get("BATCH_JOB_DEFINITION", "").strip()
    if not queue:
        return {
            "configured": False,
            "job_queue": None,
            "job_definition": definition or None,
            "by_status": {},
            "recent": [],
        }
    try:
        import boto3

        client = boto3.client("batch")
        by_status: dict[str, int] = {}
        recent: list[dict[str, Any]] = []
        for status in (
            "SUBMITTED",
            "PENDING",
            "RUNNABLE",
            "STARTING",
            "RUNNING",
            "SUCCEEDED",
            "FAILED",
        ):
            resp = client.list_jobs(
                jobQueue=queue,
                jobStatus=status,
                maxResults=max_per_status,
            )
            jobs = resp.get("jobSummaryList", [])
            by_status[status.lower()] = len(jobs)
            for job in jobs:
                recent.append(
                    {
                        "job_id": job.get("jobId"),
                        "job_name": job.get("jobName"),
                        "status": job.get("status"),
                        "status_reason": job.get("statusReason"),
                        "created_at": job.get("createdAt"),
                        "started_at": job.get("startedAt"),
                        "stopped_at": job.get("stoppedAt"),
                    }
                )
        # Prefer active jobs first in the recent list
        order = {
            "RUNNING": 0,
            "STARTING": 1,
            "RUNNABLE": 2,
            "PENDING": 3,
            "SUBMITTED": 4,
            "FAILED": 5,
            "SUCCEEDED": 6,
        }
        recent.sort(key=lambda j: (order.get(str(j.get("status")), 9), -(j.get("created_at") or 0)))
        return {
            "configured": True,
            "job_queue": queue,
            "job_definition": definition or None,
            "by_status": by_status,
            "recent": recent[:15],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "configured": True,
            "job_queue": queue,
            "job_definition": definition or None,
            "by_status": {},
            "recent": [],
            "error": str(exc),
        }


def bronze_path_exists(paths: PipelinePaths, slug: str) -> bool:
    slug = _validate_slug(slug)
    return (paths.bronze_folder(slug) / f"{slug}.md").is_file()
