from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from src.schema import ProblemRecord
from src.youtube.scrape import CHANNEL_HANDLE, ChannelVideo
from src.youtube.timestamps import (
    lookup_timestamp,
    parse_description_timestamps,
    youtube_watch_url,
)

FULL_PEMBAHASAN_RE = re.compile(
    r"pembahasan\s+(OSK|OSP|OSN)\s+fisika\s+sma\s+(\d{4})",
    re.IGNORECASE,
)
LEVEL_YEAR_RE = re.compile(r"\b(OSK|OSP|OSN)\b[^0-9]{0,40}(\d{4})\b", re.IGNORECASE)
SOAL_NUMBER_RE = re.compile(
    r"(?:soal|nomor|no\.?|problem)\s*#?\s*(\d{1,2})\b",
    re.IGNORECASE,
)
ONE_DAY_RE = re.compile(r"1\s*day\s*1\s*problem", re.IGNORECASE)
DAY_NUMBER_RE = re.compile(r"\bday\s+(\d{1,2})\b", re.IGNORECASE)
FZTI_RE = re.compile(r"\bFZTI\b", re.IGNORECASE)
LOOF_RE = re.compile(r"\bLOOF\b", re.IGNORECASE)
PRA_SESSION_RE = re.compile(r"\bpra\s*-?\s*os[kp]\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProblemVideoLink:
    problem_id: str
    video_id: str
    title: str
    url: str
    match_type: str
    confidence: float
    channel: str = CHANNEL_HANDLE
    start_seconds: int | None = None
    start_label: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


def _normalize_title(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _title_similarity(a: str, b: str) -> float:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _problems_for_exam(
    problems: list[ProblemRecord],
    level: str,
    year: int,
) -> list[ProblemRecord]:
    return [
        p
        for p in problems
        if (p.level or "").upper() == level.upper() and p.year == year
    ]


def _problem_id_for(
    problems_by_key: dict[tuple[str, int, int], ProblemRecord],
    level: str,
    year: int,
    number: int,
) -> str | None:
    rec = problems_by_key.get((level.upper(), year, number))
    return rec.id if rec else None


def _match_exam_full(
    video: ChannelVideo,
    problems: list[ProblemRecord],
) -> list[ProblemVideoLink]:
    match = FULL_PEMBAHASAN_RE.search(video.title)
    if not match:
        return []
    level, year_s = match.group(1).upper(), int(match.group(2))
    matched = _problems_for_exam(problems, level, year_s)
    if not matched:
        return []

    timestamps = parse_description_timestamps(video.description)
    links: list[ProblemVideoLink] = []
    for problem in matched:
        stamp = lookup_timestamp(timestamps, problem)
        start_seconds = stamp.seconds if stamp else None
        start_label = stamp.label if stamp else None
        match_type = "exam_timestamp" if stamp else "exam_full"
        confidence = 0.98 if stamp else 0.95
        links.append(
            ProblemVideoLink(
                problem_id=problem.id,
                video_id=video.video_id,
                title=video.title,
                url=youtube_watch_url(video.video_id, start_seconds=start_seconds),
                match_type=match_type,
                confidence=confidence,
                start_seconds=start_seconds,
                start_label=start_label,
            )
        )
    return links


def _match_explicit_soal(
    video: ChannelVideo,
    problems_by_key: dict[tuple[str, int, int], ProblemRecord],
) -> list[ProblemVideoLink]:
    level_year = LEVEL_YEAR_RE.search(video.title)
    soal = SOAL_NUMBER_RE.search(video.title)
    if not level_year or not soal:
        return []
    level, year = level_year.group(1).upper(), int(level_year.group(2))
    number = int(soal.group(1))
    problem_id = _problem_id_for(problems_by_key, level, year, number)
    if not problem_id:
        return []
    return [
        ProblemVideoLink(
            problem_id=problem_id,
            video_id=video.video_id,
            title=video.title,
            url=video.url,
            match_type="problem_number",
            confidence=0.98,
        )
    ]


def _match_title_fuzzy(
    video: ChannelVideo,
    problems: list[ProblemRecord],
    *,
    min_score: float = 0.52,
) -> list[ProblemVideoLink]:
    if FZTI_RE.search(video.title) or LOOF_RE.search(video.title):
        return []
    if FULL_PEMBAHASAN_RE.search(video.title):
        return []

    level_year = LEVEL_YEAR_RE.search(video.title)
    year = int(level_year.group(2)) if level_year else None
    level = level_year.group(1).upper() if level_year else None

    candidates = problems
    if level and year:
        candidates = _problems_for_exam(problems, level, year)
    elif year:
        candidates = [p for p in problems if p.year == year]

    best: ProblemRecord | None = None
    best_score = 0.0
    for problem in candidates:
        for probe in (problem.title, problem.title_en or ""):
            if not probe:
                continue
            score = _title_similarity(video.title, probe)
            if score > best_score:
                best_score = score
                best = problem

    if not best or best_score < min_score:
        return []

    confidence = min(0.85, 0.45 + best_score * 0.5)
    return [
        ProblemVideoLink(
            problem_id=best.id,
            video_id=video.video_id,
            title=video.title,
            url=video.url,
            match_type="title_fuzzy",
            confidence=round(confidence, 3),
        )
    ]


def match_videos_to_problems(
    videos: list[ChannelVideo],
    problems: list[ProblemRecord],
    *,
    fuzzy_min_score: float = 0.52,
) -> tuple[list[ProblemVideoLink], list[dict]]:
    """Return (links, unmatched_videos)."""
    problems_by_key: dict[tuple[str, int, int], ProblemRecord] = {}
    for p in problems:
        if p.level and p.year:
            problems_by_key[(p.level.upper(), p.year, p.problem_number)] = p

    links: list[ProblemVideoLink] = []
    unmatched: list[dict] = []

    for video in videos:
        video_links: list[ProblemVideoLink] = []

        video_links.extend(_match_exam_full(video, problems))
        if not video_links:
            video_links.extend(_match_explicit_soal(video, problems_by_key))
        if not video_links and ONE_DAY_RE.search(video.title):
            video_links.extend(
                _match_title_fuzzy(video, problems, min_score=fuzzy_min_score)
            )
        if not video_links and not PRA_SESSION_RE.search(video.title):
            video_links.extend(
                _match_title_fuzzy(video, problems, min_score=fuzzy_min_score + 0.08)
            )

        if video_links:
            links.extend(video_links)
        else:
            reason = "no_match"
            if FZTI_RE.search(video.title):
                reason = "fzti_sample"
            elif LOOF_RE.search(video.title):
                reason = "loof_workshop"
            elif PRA_SESSION_RE.search(video.title):
                reason = "pra_session_topic"
            unmatched.append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "reason": reason,
                }
            )

    return links, unmatched


def build_problem_links(links: list[ProblemVideoLink]) -> dict[str, list[dict]]:
    """Group links by problem_id, dedupe by video_id (keep highest confidence)."""
    by_problem: dict[str, dict[str, ProblemVideoLink]] = {}
    for link in links:
        bucket = by_problem.setdefault(link.problem_id, {})
        existing = bucket.get(link.video_id)
        if existing is None or link.confidence > existing.confidence:
            bucket[link.video_id] = link

    return {
        problem_id: [lnk.to_dict() for lnk in sorted(videos.values(), key=lambda x: -x.confidence)]
        for problem_id, videos in sorted(by_problem.items())
    }
