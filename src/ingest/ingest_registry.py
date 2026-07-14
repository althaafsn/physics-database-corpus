from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.text.parse_filename import parse_document
from src.paths import PipelinePaths


class IngestStage(StrEnum):
    PDF_ONLY = "pdf_only"
    BRONZE_READY = "bronze_ready"
    SILVER_DONE = "silver_done"
    GOLD_DONE = "gold_done"


class IngestEntry(BaseModel):
    slug: str
    pdf_path: str
    pdf_hash: str
    bronze_path: str | None = None
    bronze_hash: str | None = None
    stage: IngestStage = IngestStage.PDF_ONLY
    level: str | None = None
    year: int | None = None
    round: str | None = None
    variant: int | None = None
    title: str | None = None
    meta_source: str = "unknown"
    problems_count: int = 0
    errors_count: int = 0
    silver_bronze_hash: str | None = None
    notes: str | None = None
    first_seen_at: str | None = None
    last_processed_at: str | None = None
    last_run_id: str | None = None


class IngestRegistry(BaseModel):
    version: int = 1
    updated_at: str | None = None
    documents: dict[str, IngestEntry] = Field(default_factory=dict)


def file_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bronze_content_hash(bronze_folder: Path) -> str | None:
    md_files = sorted(bronze_folder.glob("*.md"))
    if not md_files:
        return None
    digest = hashlib.sha256()
    for md in md_files:
        digest.update(md.name.encode())
        digest.update(md.read_bytes())
    meta_files = sorted(bronze_folder.glob("*_meta.json"))
    for meta in meta_files:
        digest.update(meta.name.encode())
        digest.update(meta.read_bytes())
    return digest.hexdigest()


class IngestRegistryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data = self._load()

    def _load(self) -> IngestRegistry:
        if not self.path.is_file():
            return IngestRegistry()
        try:
            return IngestRegistry.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return IngestRegistry()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data.updated_at = datetime.now(UTC).isoformat()
        self.path.write_text(
            self._data.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    @property
    def documents(self) -> dict[str, IngestEntry]:
        return self._data.documents

    def get(self, slug: str) -> IngestEntry | None:
        return self._data.documents.get(slug)

    def upsert(self, entry: IngestEntry) -> None:
        existing = self._data.documents.get(entry.slug)
        if existing and existing.first_seen_at:
            entry.first_seen_at = existing.first_seen_at
        elif not entry.first_seen_at:
            entry.first_seen_at = datetime.now(UTC).isoformat()
        self._data.documents[entry.slug] = entry

    def scan_paths(
        self,
        paths: PipelinePaths,
        *,
        silver_by_slug: dict[str, list] | None = None,
    ) -> list[IngestEntry]:
        """Scan pdf_dir and bronze_dir; update registry; return entries needing silver."""
        changed: list[IngestEntry] = []
        now = datetime.now(UTC).isoformat()
        silver_by_slug = silver_by_slug or {}

        for pdf in sorted(paths.pdf_dir.glob("*.pdf")):
            slug = pdf.stem
            pdf_hash = file_hash(pdf)
            bronze_folder = paths.bronze_folder(slug)
            bronze_md = bronze_folder / f"{slug}.md"
            bronze_hash = bronze_content_hash(bronze_folder) if bronze_folder.is_dir() else None

            existing = self.get(slug)
            md_text = bronze_md.read_text(encoding="utf-8") if bronze_md.is_file() else None
            doc_meta = parse_document(slug, pdf_dir=paths.pdf_dir, md_text=md_text, pdf_path=pdf)

            if bronze_hash:
                if (
                    existing
                    and existing.silver_bronze_hash
                    and existing.silver_bronze_hash == bronze_hash
                    and existing.stage in (IngestStage.SILVER_DONE, IngestStage.GOLD_DONE)
                ):
                    stage = existing.stage
                elif not existing and slug in silver_by_slug:
                    stage = IngestStage.SILVER_DONE
                else:
                    stage = IngestStage.BRONZE_READY
            elif existing and existing.stage in (IngestStage.SILVER_DONE, IngestStage.GOLD_DONE):
                stage = existing.stage
            else:
                stage = IngestStage.PDF_ONLY

            entry = IngestEntry(
                slug=slug,
                pdf_path=str(pdf.resolve()),
                pdf_hash=pdf_hash,
                bronze_path=str(bronze_folder.resolve()) if bronze_folder.is_dir() else None,
                bronze_hash=bronze_hash,
                stage=stage,
                level=doc_meta.level,
                year=doc_meta.year,
                round=doc_meta.round,
                variant=doc_meta.variant,
                title=doc_meta.title,
                meta_source=doc_meta.meta_source,
                problems_count=existing.problems_count if existing else len(silver_by_slug.get(slug, [])),
                errors_count=(
                    existing.errors_count
                    if existing
                    else sum(1 for r in silver_by_slug.get(slug, []) if r.errors)
                ),
                silver_bronze_hash=(
                    existing.silver_bronze_hash
                    if existing
                    else (bronze_hash if slug in silver_by_slug else None)
                ),
                notes=existing.notes if existing else None,
                first_seen_at=existing.first_seen_at if existing else now,
                last_processed_at=existing.last_processed_at if existing else None,
                last_run_id=existing.last_run_id if existing else None,
            )

            is_new = existing is None
            content_changed = existing is not None and (
                existing.pdf_hash != pdf_hash or existing.bronze_hash != bronze_hash
            )
            needs_silver = entry.stage == IngestStage.BRONZE_READY
            if is_new or content_changed or needs_silver:
                if needs_silver:
                    changed.append(entry)

            self.upsert(entry)

        self.save()
        return changed

    def pending_silver_slugs(self) -> list[str]:
        pending: list[str] = []
        for slug, entry in self._data.documents.items():
            if entry.stage == IngestStage.BRONZE_READY and entry.bronze_hash:
                pending.append(slug)
        return sorted(pending)

    def mark_silver(
        self,
        slug: str,
        *,
        problems_count: int,
        errors_count: int,
        run_id: str,
    ) -> None:
        entry = self.get(slug)
        if entry is None:
            return
        entry.stage = IngestStage.SILVER_DONE
        entry.problems_count = problems_count
        entry.errors_count = errors_count
        entry.silver_bronze_hash = entry.bronze_hash
        entry.last_processed_at = datetime.now(UTC).isoformat()
        entry.last_run_id = run_id
        self.upsert(entry)
        self.save()

    def mark_gold(self, slug: str, *, run_id: str) -> None:
        entry = self.get(slug)
        if entry is None:
            return
        entry.stage = IngestStage.GOLD_DONE
        entry.last_processed_at = datetime.now(UTC).isoformat()
        entry.last_run_id = run_id
        self.upsert(entry)
        self.save()

    def summary(self) -> dict[str, Any]:
        by_stage: dict[str, int] = {}
        for entry in self._data.documents.values():
            by_stage[entry.stage] = by_stage.get(entry.stage, 0) + 1
        return {
            "total_documents": len(self._data.documents),
            "by_stage": by_stage,
            "pending_silver": len(self.pending_silver_slugs()),
        }
