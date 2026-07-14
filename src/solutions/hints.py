"""Split worked-solution markdown into progressive hint chunks for the reader UI."""
from __future__ import annotations

import re

# Indonesian answer keys often use a-, b-, c- or (a) subparts.
_SUBPART_START_RE = re.compile(
    r"^(?:\(\s*[a-z]\s*\)\s*|[a-z][\-\.)]\s+)",
    re.IGNORECASE | re.MULTILINE,
)


def _merge_small(blocks: list[str], *, min_chars: int) -> list[str]:
    if not blocks:
        return []
    merged: list[str] = []
    buf = ""
    for block in blocks:
        if not buf:
            buf = block
        elif len(buf) < min_chars:
            buf = f"{buf}\n\n{block}"
        else:
            merged.append(buf)
            buf = block
    if buf:
        merged.append(buf)
    return merged


def _cap_hints(hints: list[str], *, max_hints: int) -> list[str]:
    if len(hints) <= max_hints:
        return hints
    kept = hints[: max_hints - 1]
    kept.append("\n\n".join(hints[max_hints - 1 :]))
    return kept


def _split_by_subparts(text: str, *, min_chars: int) -> list[str] | None:
    matches = list(_SUBPART_START_RE.finditer(text))
    if len(matches) < 2:
        return None

    segments: list[str] = []
    intro = text[: matches[0].start()].strip()
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append(text[match.start() : end].strip())

    if intro:
        if len(intro) >= min_chars or not segments:
            segments.insert(0, intro)
        else:
            segments[0] = f"{intro}\n\n{segments[0]}"

    return segments if len(segments) >= 2 else None


def _split_by_paragraphs(text: str, *, min_chars: int) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n{2,}", text) if b.strip()]
    if len(blocks) < 2:
        return [text] if text else []
    merged = _merge_small(blocks, min_chars=min_chars)
    return merged if len(merged) >= 2 else blocks


def split_solution_into_hints(
    body_md: str,
    *,
    max_hints: int = 8,
    min_hint_chars: int = 60,
) -> list[str]:
    """Return ordered hint chunks; never includes empty strings."""
    text = body_md.strip()
    if not text:
        return []

    hints = _split_by_subparts(text, min_chars=min_hint_chars)
    if hints is None:
        hints = _split_by_paragraphs(text, min_chars=min_hint_chars)

    hints = [h for h in hints if h.strip()]
    if not hints:
        return [text]

    return _cap_hints(hints, max_hints=max_hints)
