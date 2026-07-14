"""Persist and track single-PDF ingest jobs for the admin API."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from src.paths import PipelinePaths


class IngestJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class IngestJobStage(str, Enum):
    QUEUED = "queued"
    SAVING = "saving"
    BRONZE = "bronze"
    PARSING = "parsing"
    TAGGING = "tagging"
    CATALOG = "catalog"
    TRANSLATE = "translate"
    EXPORT = "export"
    DONE = "done"


@dataclass
class IngestJob:
    id: str
    status: IngestJobStatus
    stage: IngestJobStage
    slug: str
    pdf_filename: str
    content_locale: str = "id"
    content_locale_hint: str = "auto"
    publish: bool = False
    problems_count: int = 0
    problem_ids: list[str] = field(default_factory=list)
    catalog_total: int = 0
    error: str | None = None
    detail: str | None = None
    created_at: str = ""
    updated_at: str = ""
    kind: str = "single"
    parent_id: str | None = None
    total_files: int = 0
    completed_files: int = 0
    current_file_index: int = 0
    files: list[dict] = field(default_factory=list)
    translate: bool = True
    translated_count: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        data["stage"] = self.stage.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> IngestJob:
        return cls(
            id=data["id"],
            status=IngestJobStatus(data["status"]),
            stage=IngestJobStage(data["stage"]),
            slug=data["slug"],
            pdf_filename=data["pdf_filename"],
            content_locale=data.get("content_locale", "id"),
            content_locale_hint=data.get("content_locale_hint", "auto"),
            publish=bool(data.get("publish", False)),
            problems_count=int(data.get("problems_count", 0)),
            problem_ids=list(data.get("problem_ids", [])),
            catalog_total=int(data.get("catalog_total", 0)),
            error=data.get("error"),
            detail=data.get("detail"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            kind=data.get("kind", "single"),
            parent_id=data.get("parent_id"),
            total_files=int(data.get("total_files", 0)),
            completed_files=int(data.get("completed_files", 0)),
            current_file_index=int(data.get("current_file_index", 0)),
            files=list(data.get("files", [])),
            translate=bool(data.get("translate", True)),
            translated_count=int(data.get("translated_count", 0)),
        )


class IngestJobStore:
    def __init__(self, jobs_dir: Path) -> None:
        self.jobs_dir = jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def save(self, job: IngestJob) -> None:
        job.updated_at = datetime.now(UTC).isoformat()
        path = self._path(job.id)
        tmp = path.with_suffix(".json.tmp")
        payload = json.dumps(job.to_dict(), indent=2, ensure_ascii=False) + "\n"
        with self._lock:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, path)

    def get(self, job_id: str) -> IngestJob | None:
        path = self._path(job_id)
        if not path.is_file():
            return None
        with self._lock:
            for attempt in range(5):
                try:
                    text = path.read_text(encoding="utf-8")
                    if not text.strip():
                        raise json.JSONDecodeError("empty file", text, 0)
                    return IngestJob.from_dict(json.loads(text))
                except json.JSONDecodeError:
                    if attempt == 4:
                        raise
                    time.sleep(0.02 * (attempt + 1))
        return None

    def create(
        self,
        *,
        slug: str,
        pdf_filename: str,
        content_locale_hint: str = "auto",
        publish: bool = False,
        kind: str = "single",
        parent_id: str | None = None,
        translate: bool = True,
    ) -> IngestJob:
        now = datetime.now(UTC).isoformat()
        job = IngestJob(
            id=uuid.uuid4().hex[:12],
            status=IngestJobStatus.QUEUED,
            stage=IngestJobStage.QUEUED,
            slug=slug,
            pdf_filename=pdf_filename,
            content_locale_hint=content_locale_hint,
            publish=publish,
            kind=kind,
            parent_id=parent_id,
            translate=translate,
            created_at=now,
            updated_at=now,
        )
        self.save(job)
        return job

    def update(self, job_id: str, **fields) -> IngestJob | None:
        job = self.get(job_id)
        if job is None:
            return None
        for key, value in fields.items():
            if hasattr(job, key):
                setattr(job, key, value)
        self.save(job)
        return job

    def list_recent(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
    ) -> list[IngestJob]:
        """Return newest jobs first (by updated_at), optionally filtered by status."""
        jobs: list[IngestJob] = []
        with self._lock:
            paths = sorted(
                self.jobs_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        for path in paths:
            if path.name.endswith(".tmp"):
                continue
            try:
                job = self.get(path.stem)
            except (json.JSONDecodeError, KeyError, ValueError, OSError):
                continue
            if job is None:
                continue
            if status and job.status.value != status:
                continue
            jobs.append(job)
            if len(jobs) >= limit:
                break
        jobs.sort(key=lambda j: j.updated_at or j.created_at, reverse=True)
        return jobs[:limit]


_executor_lock = threading.Lock()
_running = 0
_MAX_CONCURRENT = 1


def enqueue_ingest(
    store: IngestJobStore,
    job: IngestJob,
    runner: Callable[[IngestJob], None],
) -> None:
    global _running

    def _wrapped() -> None:
        global _running
        try:
            runner(job)
        finally:
            with _executor_lock:
                _running = max(0, _running - 1)

    with _executor_lock:
        if _running >= _MAX_CONCURRENT:
            store.update(
                job.id,
                status=IngestJobStatus.FAILED,
                stage=IngestJobStage.DONE,
                error="Another ingest job is already running. Try again shortly.",
            )
            return
        _running += 1

    thread = threading.Thread(target=_wrapped, name=f"ingest-{job.id}", daemon=True)
    thread.start()


def jobs_dir_for(paths: PipelinePaths) -> Path:
    return paths.parsed_dir / "ingest_jobs"
