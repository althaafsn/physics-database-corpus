"""Re-derive problem bodies from body_md_raw (original Marker OCR)."""
from __future__ import annotations

from src.text.clean import clean_record, clean_text
from src.schema import ProblemRecord, SubPart
from src.text.split_problems import extract_subparts
from src.validate import apply_validation


def rebaseline_from_raw(record: ProblemRecord) -> bool:
    """Replace body_md with a cleaned copy of body_md_raw. Returns True if changed."""
    raw = record.body_md_raw
    if not raw or not raw.strip():
        return False

    cleaned = clean_text(raw)
    if not cleaned.strip():
        return False

    if cleaned == record.body_md:
        return False

    record.body_md = cleaned
    record.subparts = [SubPart(**sp) for sp in extract_subparts(cleaned)]
    clean_record(record)
    apply_validation(record)
    return True
