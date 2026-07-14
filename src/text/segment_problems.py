"""Format-agnostic problem segmentation for physics olympiad PDFs.

Tries multiple numbering/layout conventions (Indonesian OSK, IPhO, APhO, generic
English) and picks the best-scoring split instead of assuming one document shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.bronze.pdf_text import strip_pdf_footers

# --- Normalization -----------------------------------------------------------

_PROMO_BOUNDARY_RE = re.compile(
    r"^#{0,6}\s*.*(?:FZTI|Dimensi Sains|Program Persiapan|Unduh Buku|Pendaftaran dibuka|Promo Awal)",
    re.MULTILINE | re.IGNORECASE,
)

_SINGLE_DOC_SKIP_RE = re.compile(
    r"^(?:theory|experiment|english|official|problem|question|task|"
    r"\w+\s*\(official\)|q\d+-\d+)\s*$",
    re.IGNORECASE,
)

_PLAIN_NUMBERED_RE = re.compile(
    r"^(?P<num>\d{1,2})\.\s+(?P<title>.+)$",
    re.MULTILINE,
)

_INLINE_NUMBERED_RE = re.compile(
    r"(?:^|\n)(?:-\s*)?(?P<num>\d{1,2})\)\s",
    re.MULTILINE,
)

_LABELED_LINE_RE = re.compile(
    r"^(?:Problem|Question|Soal|Nomor|Task)\s+"
    r"(?:No\.?|Number|Nomor)?\s*(?P<num>\d{1,2})\s*(?:[.:\-]\s*(?P<title>.+))?$",
    re.IGNORECASE | re.MULTILINE,
)

_SOAL_NOMOR_RE = re.compile(
    r"^Soal\s+Nomor\s+(?P<num>\d{1,2})\s*[:\-]\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

_NOMOR_ONLY_RE = re.compile(
    r"^Nomor\s+(?P<num>\d{1,2})\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_NUMBER_HEADING_RE = re.compile(
    r"^.*?\bNumber\s+(?P<num>\d{1,2})\b[:\-]?\s*(?P<title>.*)$",
    re.IGNORECASE | re.MULTILINE,
)

_MARKDOWN_HEADING_RE = re.compile(
    r"^#{1,6}\s+\*{0,2}(\d{1,2}\.\s+.+?)\*{0,2}\s*$",
    re.MULTILINE,
)
_VARIANT_MARKER_RE = re.compile(
    r"^(?:#{0,6}\s*)?(?:Versi|Version)\s+(?P<variant>\d+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class SegmentResult:
    problems: dict[int, tuple[str, str]]
    strategy: str
    score: float


def normalize_exam_text(text: str) -> str:
    text = strip_pdf_footers(text or "")
    boundary = _PROMO_BOUNDARY_RE.search(text)
    if boundary:
        text = text[: boundary.start()]
    text = re.sub(r"\f+", "\n", text)
    return text.strip()


def _score_problems(problems: dict[int, tuple[str, str]], *, text: str = "") -> float:
    if not problems:
        return 0.0
    lengths = [len(body.strip()) for _, body in problems.values()]
    if not lengths or max(lengths) < 40:
        return 0.0

    count = len(problems)
    avg_len = sum(lengths) / count
    tiny = sum(1 for length in lengths if length < 60)

    score = avg_len / 120.0 + min(count, 25) * 1.5
    if count == 1 and lengths[0] >= 400:
        score += 12.0
    if count >= 2:
        score += 4.0
    score -= tiny * 4.0

    if count == 1 and text:
        if len(_PLAIN_NUMBERED_RE.findall(text)) >= 2:
            score -= 8.0
        if len(_INLINE_NUMBERED_RE.findall(text)) >= 2:
            score -= 6.0
        if len(_LABELED_LINE_RE.findall(text)) >= 2:
            score -= 6.0
    return score


def _dict_from_spans(
    text: str,
    markers: list[tuple[int, int, int, str | None]],
    *,
    min_body: int = 40,
) -> dict[int, tuple[str, str]]:
    """markers: (body_start, body_end_anchor, problem_num, optional_title)"""
    if not markers:
        return {}

    if len(markers) >= 2:
        min_body = min(min_body, 15)

    problems: dict[int, tuple[str, str]] = {}
    for index, (body_start, _anchor_end, number, title_hint) in enumerate(markers):
        body_end = markers[index + 1][0] if index + 1 < len(markers) else len(text)
        body = text[body_start:body_end].strip()
        if len(body) < min_body:
            continue
        title = (title_hint or "").strip()
        if not title:
            title = _infer_title_from_body(number, body)
        problems[number] = (title, body)
    return problems


def _infer_title_from_body(number: int, body: str) -> str:
    for line in body.splitlines():
        candidate = line.strip()
        candidate = re.sub(r"^[-*#]+\s*", "", candidate)
        candidate = re.sub(r"\*{1,2}", "", candidate).strip()
        if len(candidate) < 8:
            continue
        if _SINGLE_DOC_SKIP_RE.match(candidate):
            continue
        if re.match(r"^part\s+[a-z]\b", candidate, re.IGNORECASE):
            continue
        return candidate[:160]
    return f"Problem {number}"


def _accept_monotonic_markers(
    markers: list[tuple[int, int, int, str | None]],
) -> list[tuple[int, int, int, str | None]]:
    if len(markers) < 2:
        return markers

    deduped: list[tuple[int, int, int, str | None]] = []
    for marker in markers:
        if deduped and marker[2] == deduped[-1][2]:
            continue
        deduped.append(marker)

    accepted: list[tuple[int, int, int, str | None]] = [deduped[0]]
    for marker in deduped[1:]:
        if marker[2] >= accepted[-1][2]:
            accepted.append(marker)
    return accepted if len(accepted) >= 2 else markers[:1]


def _collect_regex_markers(
    text: str,
    pattern: re.Pattern[str],
    *,
    title_group: str | None = "title",
) -> list[tuple[int, int, int, str | None]]:
    markers: list[tuple[int, int, int, str | None]] = []
    seen: set[int] = set()
    for match in pattern.finditer(text):
        start = match.start()
        if start in seen:
            continue
        line_end = text.find("\n", start)
        line = text[start : line_end if line_end >= 0 else None]
        if re.match(r"^\d{1,2}-\d{1,2}\s", line):
            continue
        seen.add(start)
        title = match.group(title_group).strip() if title_group and match.groupdict().get(title_group) else None
        markers.append((match.end(), match.end(), int(match.group("num")), title))
    markers.sort(key=lambda item: item[0])
    return markers


def _strategy_plain_numbered(text: str) -> dict[int, tuple[str, str]]:
    matches = list(_PLAIN_NUMBERED_RE.finditer(text))
    if len(matches) < 2:
        return {}
    markers = [
        (match.end(), match.end(), int(match.group("num")), match.group("title").strip())
        for match in matches
    ]
    return _dict_from_spans(text, markers)


def _strategy_inline_numbered(text: str) -> dict[int, tuple[str, str]]:
    matches = list(_INLINE_NUMBERED_RE.finditer(text))
    if len(matches) < 2:
        return {}
    markers = [
        (match.start("num"), match.end(), int(match.group("num")), None) for match in matches
    ]
    return _dict_from_spans(text, _accept_monotonic_markers(markers))


def _strategy_labeled_lines(text: str) -> dict[int, tuple[str, str]]:
    markers: list[tuple[int, int, int, str | None]] = []
    for pattern, title_group in (
        (_SOAL_NOMOR_RE, "title"),
        (_LABELED_LINE_RE, "title"),
        (_NOMOR_ONLY_RE, None),
        (_NUMBER_HEADING_RE, "title"),
    ):
        markers.extend(_collect_regex_markers(text, pattern, title_group=title_group))

    markers.sort(key=lambda item: item[0])
    if len(markers) < 2:
        return {}
    return _dict_from_spans(text, _accept_monotonic_markers(markers))


def _strategy_markdown_headings(text: str) -> dict[int, tuple[str, str]]:
    from src.text.split_problems import split_markdown

    if not _MARKDOWN_HEADING_RE.search(text):
        return {}
    problems = split_markdown(text)
    if len(problems) < 1:
        return {}
    return {number: (title, body) for number, title, body in problems}


def _strategy_markdown_auto(text: str, *, year: int | None) -> dict[int, tuple[str, str]]:
    from src.text.split_problems import split_markdown_auto

    problems = split_markdown_auto(text, year=year)
    if not problems:
        return {}
    return {number: (title, body) for number, title, body in problems}


def _strategy_single_document(text: str, *, slug: str) -> dict[int, tuple[str, str]]:
    body = text.strip()
    if len(body) < 120:
        return {}

    title = slug.replace("_", " ").replace("-", " ").strip() or "Problem 1"
    for line in body.splitlines():
        candidate = line.strip()
        if len(candidate) < 12:
            continue
        if _SINGLE_DOC_SKIP_RE.match(candidate):
            continue
        if re.match(r"^part\s+[a-z]\b", candidate, re.IGNORECASE):
            continue
        title = candidate[:160]
        break
    return {1: (title, body)}


def segment_exam_text(
    text: str,
    *,
    slug: str = "",
    year: int | None = None,
) -> SegmentResult:
    """Segment exam text using the best-scoring strategy."""
    normalized = normalize_exam_text(text)
    if not normalized:
        return SegmentResult({}, "empty", 0.0)

    candidates: list[tuple[float, str, dict[int, tuple[str, str]]]] = []

    def add(name: str, problems: dict[int, tuple[str, str]]) -> None:
        score = _score_problems(problems, text=normalized)
        if score > 0:
            candidates.append((score, name, problems))

    add("plain_numbered", _strategy_plain_numbered(normalized))
    add("inline_numbered", _strategy_inline_numbered(normalized))
    add("labeled_lines", _strategy_labeled_lines(normalized))
    add("markdown_headings", _strategy_markdown_headings(normalized))
    add("markdown_auto", _strategy_markdown_auto(normalized, year=year))

    if candidates:
        _score, strategy, problems = max(candidates, key=lambda item: item[0])
        return SegmentResult(problems, strategy, _score)

    single = _strategy_single_document(normalized, slug=slug)
    if single:
        single_score = _score_problems(single, text=normalized)
        return SegmentResult(single, "single_document", single_score)

    return SegmentResult({}, "none", 0.0)


def segment_exam_text_to_list(
    text: str,
    *,
    slug: str = "",
    year: int | None = None,
) -> list[tuple[int, str, str]]:
    result = segment_exam_text(text, slug=slug, year=year)
    return [(num, title, body) for num, (title, body) in sorted(result.problems.items())]


def segment_exam_text_to_variant_list(
    text: str,
    *,
    slug: str = "",
    year: int | None = None,
) -> list[tuple[int, str, str, int | None]]:
    """Return ordered PDF segments without collapsing repeated variant numbers."""
    normalized = normalize_exam_text(text)
    result = segment_exam_text(normalized, slug=slug, year=year)
    if not normalized:
        return []

    matches = list(_PLAIN_NUMBERED_RE.finditer(normalized))
    if len(matches) >= 2:
        variant_markers = list(_VARIANT_MARKER_RE.finditer(normalized))
        out: list[tuple[int, str, str, int | None]] = []
        min_body = min(40, 15) if len(matches) >= 2 else 40
        for index, match in enumerate(matches):
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
            body = normalized[match.end() : body_end].strip()
            if len(body) < min_body:
                continue
            variant = None
            for marker in variant_markers:
                if marker.start() > match.start():
                    break
                variant = int(marker.group("variant"))
            out.append((int(match.group("num")), match.group("title").strip(), body, variant))
        if out:
            return out

    from src.text.split_problems import split_markdown_auto_with_variants

    if result.strategy in {"markdown_headings", "markdown_auto"}:
        return split_markdown_auto_with_variants(normalized, year=year)
    return [(num, title, body, None) for num, title, body in segment_exam_text_to_list(normalized, slug=slug, year=year)]
