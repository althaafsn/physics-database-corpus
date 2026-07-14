from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.halliday.classify import ProblemTags
from src.schema import ProblemRecord

CACHE_VERSION = 3


def tag_cache_key(record: ProblemRecord) -> str:
    blob = json.dumps(
        {
            "v": CACHE_VERSION,
            "id": record.id,
            "topic": record.topic,
            "title": record.title,
            "body": record.body_md[:2000],
            "subparts": [sp.text[:200] for sp in record.subparts[:6]],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def halliday_cache_dir(cache_root: Path) -> Path:
    return cache_root / "halliday"


def _parse_tags(data: dict) -> ProblemTags:
    # Support v1 (chapters/sections) and v2 (topics/details)
    topics = data.get("topics") or data.get("chapters", [])
    details = data.get("details") or data.get("sections", [])
    disciplines = data.get("disciplines", [])
    return ProblemTags(
        problem_id=data["problem_id"],
        topics=topics,
        details=details,
        disciplines=disciplines,
        confidence=float(data.get("confidence", 0.5)),
        method=data.get("method", "llm"),
        model=data.get("model"),
    )


def load_cached_tags(cache_root: Path, record_id: str, key: str) -> ProblemTags | None:
    path = halliday_cache_dir(cache_root) / f"{record_id}_{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("topics") or data.get("details"):
            return _parse_tags(data)
        return None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_cached_tags(cache_root: Path, record_id: str, key: str, tags: ProblemTags) -> None:
    dest = halliday_cache_dir(cache_root)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{record_id}_{key}.json"
    path.write_text(json.dumps(tags.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
