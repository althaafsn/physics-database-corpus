from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src.schema import ProblemRecord

TIMESTAMP_LINE_RE = re.compile(
    r"^No\.?\s*(\d+)\s*([a-z])?(?:\s+[^-]*)?\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class VideoTimestamp:
    key: str
    problem_number: int
    subpart: str | None
    seconds: int
    label: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "problem_number": self.problem_number,
            "subpart": self.subpart,
            "seconds": self.seconds,
            "label": self.label,
        }


def parse_timecode(value: str) -> int:
    parts = [int(p) for p in value.strip().split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Invalid timecode: {value!r}")


def parse_description_timestamps(description: str | None) -> dict[str, VideoTimestamp]:
    if not description:
        return {}

    found: dict[str, VideoTimestamp] = {}
    for match in TIMESTAMP_LINE_RE.finditer(description):
        number = int(match.group(1))
        subpart = (match.group(2) or "").lower() or None
        label = match.group(3)
        key = f"{number}{subpart}" if subpart else str(number)
        seconds = parse_timecode(label)
        found[key] = VideoTimestamp(
            key=key,
            problem_number=number,
            subpart=subpart,
            seconds=seconds,
            label=label,
        )
    return found


def timestamp_lookup_keys(problem: ProblemRecord) -> list[str]:
    number = problem.problem_number
    keys: list[str] = []
    if problem.subparts:
        first = problem.subparts[0].label.strip().lower()
        if first:
            keys.append(f"{number}{first}")
    keys.append(str(number))
    return keys


def lookup_timestamp(
    timestamps: dict[str, VideoTimestamp],
    problem: ProblemRecord,
) -> VideoTimestamp | None:
    for key in timestamp_lookup_keys(problem):
        hit = timestamps.get(key)
        if hit:
            return hit
    return None


def youtube_watch_url(video_id: str, *, start_seconds: int | None = None) -> str:
    base = f"https://www.youtube.com/watch?v={video_id}"
    if start_seconds is None:
        return base
    return f"{base}&t={start_seconds}"


def load_manual_descriptions(root: Path) -> dict[str, str]:
    path = root / "data" / "youtube" / "manual_descriptions.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}
