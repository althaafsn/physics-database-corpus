"""Preserve raw Marker markdown before hybrid / doc-structure post-processing."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

_POST_MARKER_SOURCES = frozenset(
    {
        "doc_structure",
        "doc_align",
        "doc_fuse",
        "llm_merge",
        "hybrid_pdftotext",
    }
)


def marker_backup_path(bronze_folder: Path) -> Path:
    slug = bronze_folder.name
    return bronze_folder / f"{slug}.marker.md"


def bronze_source_text(bronze_folder: Path) -> str:
    slug = bronze_folder.name
    meta = bronze_folder / f"{slug}_bronze_source.json"
    if not meta.is_file():
        return ""
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return str(data.get("text_source") or "").strip()


def save_marker_backup(bronze_folder: Path, *, force: bool = False) -> Path | None:
    """Copy current bronze ``.md`` to ``.marker.md`` when it is still raw Marker output."""
    backup = marker_backup_path(bronze_folder)
    if backup.is_file() and not force:
        return backup

    slug = bronze_folder.name
    current = bronze_folder / f"{slug}.md"
    if not current.is_file():
        return None

    source = bronze_source_text(bronze_folder)
    if source in _POST_MARKER_SOURCES and not force:
        return backup if backup.is_file() else None

    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current, backup)
    return backup


def ensure_marker_backup(bronze_folder: Path) -> Path | None:
    """Return existing backup or create one from raw Marker bronze if possible."""
    backup = marker_backup_path(bronze_folder)
    if backup.is_file():
        return backup
    return save_marker_backup(bronze_folder)


def load_marker_md(bronze_folder: Path) -> str | None:
    backup = marker_backup_path(bronze_folder)
    if backup.is_file():
        text = backup.read_text(encoding="utf-8").strip()
        return text if text else None

    slug = bronze_folder.name
    current = bronze_folder / f"{slug}.md"
    if not current.is_file():
        return None

    source = bronze_source_text(bronze_folder)
    if source in _POST_MARKER_SOURCES:
        return None

    text = current.read_text(encoding="utf-8").strip()
    return text if text else None
