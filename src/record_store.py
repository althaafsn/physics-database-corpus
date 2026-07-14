from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.schema import ProblemRecord


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def load_jsonl(path: Path, *, lenient: bool = False) -> list[ProblemRecord]:
    if not path.is_file():
        return []
    records: list[ProblemRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if lenient:
            data = json.loads(line)
            if not data.get("document_slug"):
                pdf = data.get("source", {}).get("pdf", "")
                data["document_slug"] = Path(pdf).stem if pdf else data.get("id", "unknown")
            records.append(ProblemRecord(**data))
        else:
            records.append(ProblemRecord.model_validate_json(line))
    return records


def save_jsonl(path: Path, records: list[ProblemRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def source_pdf_key(record: ProblemRecord) -> str:
    return str(Path(record.source.pdf).resolve())


def document_slug_key(record: ProblemRecord) -> str:
    if record.document_slug:
        return record.document_slug
    return Path(record.source.pdf).stem


def merge_records(
    existing: list[ProblemRecord],
    new_records: list[ProblemRecord],
    *,
    replace_source_pdfs: set[str],
    replace_slugs: set[str] | None = None,
) -> list[ProblemRecord]:
    """Replace records for given source PDFs/slugs; keep all others."""
    replace_slugs = replace_slugs or set()
    kept = [
        rec
        for rec in existing
        if source_pdf_key(rec) not in replace_source_pdfs
        and document_slug_key(rec) not in replace_slugs
    ]
    merged = kept + new_records
    merged.sort(key=lambda r: (document_slug_key(r), r.problem_number, r.id))
    return merged


def append_run_history(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def migrate_legacy_problems(paths_silver: Path, legacy_path: Path) -> list[ProblemRecord]:
    """Load silver if present, else legacy problems.jsonl (with document_slug backfill)."""
    source = paths_silver if paths_silver.is_file() else legacy_path
    records = load_jsonl(source, lenient=True)
    migrated: list[ProblemRecord] = []
    for rec in records:
        data = rec.model_dump()
        if not data.get("document_slug"):
            data["document_slug"] = Path(rec.source.pdf).stem
        migrated.append(ProblemRecord(**data))
    return migrated
