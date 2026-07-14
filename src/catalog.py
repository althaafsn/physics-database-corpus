from __future__ import annotations

import json
from datetime import UTC, datetime

from src.paths import PipelinePaths
from src.record_store import load_jsonl, save_jsonl
from src.schema import ProblemRecord
from src.graph.id_migration import normalize_graph_files

LOW_CONFIDENCE_THRESHOLD = 0.6


def record_content_locale(record: ProblemRecord) -> str:
    locale = getattr(record, "content_locale", None) or "id"
    return locale if locale in {"id", "en"} else "id"


def is_in_locale_catalog(record: ProblemRecord, locale: str) -> bool:
    """Whether a catalog-eligible record belongs in the public catalog for ``locale``."""
    content = record_content_locale(record)
    if locale == "id":
        return content != "en"
    if locale == "en":
        if content == "en":
            return True
        return bool(record.body_md_en and record.body_md_en.strip())
    return True


def is_catalog_eligible(record: ProblemRecord) -> bool:
    """Problems exposed to end users: no validation errors, confident topic."""
    if record.errors:
        return False
    if record.topic == "mixed":
        return False
    if record.topic_confidence < LOW_CONFIDENCE_THRESHOLD:
        return False
    return True


def sync_catalog(paths: PipelinePaths | None = None) -> dict[str, int | str]:
    """Write parsed/catalog/problems.jsonl from gold (eligible records only)."""
    paths = paths or PipelinePaths.resolve()
    paths.ensure_dirs()

    catalog_dir = paths.parsed_dir / "catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = catalog_dir / "problems.jsonl"
    meta_path = catalog_dir / "manifest.json"

    gold = load_jsonl(paths.gold_problems_path, lenient=True)
    if not gold and paths.silver_problems_path.is_file():
        gold = load_jsonl(paths.silver_problems_path, lenient=True)

    eligible = [rec for rec in gold if is_catalog_eligible(rec)]
    eligible.sort(key=lambda r: (r.level or "", r.year or 0, r.document_slug, r.problem_number, r.id))

    save_jsonl(catalog_path, eligible)

    meta = {
        "updated_at": datetime.now(UTC).isoformat(),
        "source": str(paths.gold_problems_path),
        "gold_total": len(gold),
        "catalog_total": len(eligible),
        "excluded_errors": sum(1 for r in gold if r.errors),
        "excluded_low_confidence": sum(
            1 for r in gold if not r.errors and not is_catalog_eligible(r)
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    normalize_graph_files(paths.parsed_dir, eligible)

    return meta
