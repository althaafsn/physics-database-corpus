from __future__ import annotations

import re

from src.schema import ProblemRecord, ValidationIssue

from src.text.clean import FOOTER_INLINE_RE, PROMO_CONTAMINATION_RE, clean_text
from src.text.attach_images import body_expects_attached_figure, extract_image_refs
from src.math.math_normalize import MATH_ITALIC_CHAR_RE
BLANK_SLOT_RE = re.compile(
    r"\(\s*,\s*\)|,\s*,\s*|dalam\s+,\s*,\s*dan|Nyatakan dalam\s+,\s*,\s*dan"
    r"|(?:\$[^$\n]{1,20}\$\s*,\s*)+dan\s*[.!?]",
    re.IGNORECASE,
)
# "noun adalah/rotasinya <blank>" only counts as a missing-symbol slot when the
# word is immediately followed by punctuation (nothing meaningful in between).
# Definitional sentences like "massa adalah osilasi ..." or "Sudut adalah sudut
# antara ..." must NOT match - there's a real description there, not a blank.
MISSING_SYMBOL_NOUN_RE = re.compile(
    r"(?<![\w$\\])"
    r"(?:massa|kecepatan|sudut|periode|gravitasi|bermassa|jari-jari|konstanta)"
    r"\s+(?:=(?!\s*\.{2,})|adalah\s*(?!\.{2,})[,.]|rotasinya\s*(?!\.{2,})[,.]|,|\.(?!\.))",
    re.IGNORECASE,
)
MISSING_G_RE = re.compile(r"=\s*10\s*m/s\s*2", re.IGNORECASE)
ORPHAN_SUPERSCRIPT_RE = re.compile(
    r"(?:bermassa|konstanta pegas|kecepatan|partikel|balok|pegas|massa)\s*<sup>",
    re.IGNORECASE,
)
HTML_MATH_RE = re.compile(r"<su[bp]>", re.IGNORECASE)

# Signatures left behind by broken/failed LLM repair passes - these should
# never appear in a real problem body and indicate the record needs a redo.
PLACEHOLDER_LEFTOVER_RE = re.compile(
    r"<\s*(?:cleaned\s+markdown|partial|fixed\s+markdown|body_md|unchanged)\s*>",
    re.IGNORECASE,
)
# Concatenated nonsense LaTeX like "$m_1gT$" - multiple physics symbols glued
# together with no operator is never valid notation, only ever a hallucination.
GARBLED_SYMBOL_RE = re.compile(r"\$[a-zA-Z]+_\d[a-zA-Z]{2,}\$")
# A backslash-escape that got mangled into a literal tab/control character,
# e.g. "$\theta$" corrupted to "$\theta$" with a raw tab instead of backslash-t.
MANGLED_ESCAPE_RE = re.compile(r"\$[^$\n]*[\t\x00-\x08\x0b-\x1f][^$\n]*\$")

SYMBOL_REPAIR_CODES = frozenset(
    {
        "missing_symbol_after_noun",
        "blank_variable_slot",
        "html_math_markup",
        "missing_g_constant",
        "orphan_superscript",
    }
)

FULL_LLM_REPAIR_CODES = frozenset(
    {
        "footer_contamination",
        "promo_contamination",
        "content_placeholder_leftover",
        "garbled_llm_symbol",
        "mangled_latex_escape",
        "content_loss_vs_raw",
        "missing_image",
        "expected_image_missing",
        "manual_review_required",
    }
)

CONTENT_LOSS_THRESHOLD = 0.7
_WORD_RE = re.compile(r"[a-zA-Zà-ÿ]{3,}")


def _word_set(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _content_loss_ratio(record: ProblemRecord) -> float | None:
    """Fraction of the original (cleaned) raw text's words still present in
    the current body + subparts. None when there's no raw baseline to compare."""
    if not record.body_md_raw:
        return None
    raw_words = _word_set(clean_text(record.body_md_raw))
    if not raw_words:
        return None
    current = record.body_md + "\n" + "\n".join(sp.text for sp in record.subparts)
    body_words = _word_set(current)
    return len(raw_words & body_words) / len(raw_words)


def _snippet(text: str, match: re.Match[str], width: int = 60) -> str:
    start = max(0, match.start() - width // 2)
    end = min(len(text), match.end() + width // 2)
    return text[start:end].replace("\n", " ").strip()


def _has_g_symbol(text: str, pos: int) -> bool:
    prefix = text[max(0, pos - 30) : pos]
    return bool(re.search(r"\$g\$|\bg\s*=|\bg\s", prefix))


def _has_math_symbol_after(text: str, pos: int, window: int = 35) -> bool:
    """True when OCR/markdown already has $...$ or math-italic Unicode soon after a noun phrase."""
    chunk = text[pos : pos + window]
    chunk = re.sub(r"\*+", "", chunk).lstrip()
    if re.search(r"\$[^$\n]+\$", chunk):
        return True
    return bool(MATH_ITALIC_CHAR_RE.search(chunk))


def _issues_from_text(text: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for match in FOOTER_INLINE_RE.finditer(text):
        issues.append(
            ValidationIssue(
                code="footer_contamination",
                message="Promotional footer or contact text detected",
                snippet=_snippet(text, match),
            )
        )
        break

    for match in PROMO_CONTAMINATION_RE.finditer(text):
        issues.append(
            ValidationIssue(
                code="promo_contamination",
                message="FZTI or course signup promotional text detected",
                snippet=_snippet(text, match),
            )
        )
        break

    for match in BLANK_SLOT_RE.finditer(text):
        issues.append(
            ValidationIssue(
                code="blank_variable_slot",
                message="Blank variable slot detected in problem text",
                snippet=_snippet(text, match),
            )
        )
        break

    for match in MISSING_SYMBOL_NOUN_RE.finditer(text):
        if _has_math_symbol_after(text, match.end()):
            continue
        issues.append(
            ValidationIssue(
                code="missing_symbol_after_noun",
                message="Physics quantity appears without its symbol",
                snippet=_snippet(text, match),
            )
        )
        break

    for match in MISSING_G_RE.finditer(text):
        if not _has_g_symbol(text, match.start()):
            issues.append(
                ValidationIssue(
                    code="missing_g_constant",
                    message="Gravitational acceleration written without g symbol",
                    snippet=_snippet(text, match),
                )
            )
            break

    for match in ORPHAN_SUPERSCRIPT_RE.finditer(text):
        issues.append(
            ValidationIssue(
                code="orphan_superscript",
                message="Superscript without base variable detected",
                snippet=_snippet(text, match),
            )
        )
        break

    if HTML_MATH_RE.search(text):
        match = HTML_MATH_RE.search(text)
        assert match is not None
        issues.append(
            ValidationIssue(
                code="html_math_markup",
                message="HTML sup/sub markup should be LaTeX math",
                snippet=_snippet(text, match),
            )
        )

    match = PLACEHOLDER_LEFTOVER_RE.search(text)
    if match:
        issues.append(
            ValidationIssue(
                code="content_placeholder_leftover",
                message="Body text contains an unfilled LLM template placeholder",
                snippet=_snippet(text, match),
            )
        )

    match = GARBLED_SYMBOL_RE.search(text)
    if match:
        issues.append(
            ValidationIssue(
                code="garbled_llm_symbol",
                message="Concatenated/nonsensical symbol likely hallucinated by LLM repair",
                snippet=_snippet(text, match),
            )
        )

    match = MANGLED_ESCAPE_RE.search(text)
    if match:
        issues.append(
            ValidationIssue(
                code="mangled_latex_escape",
                message="LaTeX escape corrupted into a control character",
                snippet=_snippet(text, match),
            )
        )

    return issues


def _issues_from_flags(flags: list[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for flag in flags:
        if flag.startswith("missing_image:"):
            issues.append(
                ValidationIssue(
                    code="missing_image",
                    message=f"Referenced image not found: {flag.split(':', 1)[1]}",
                    snippet=flag,
                )
            )
        elif flag == "expected_image_missing":
            issues.append(
                ValidationIssue(
                    code="expected_image_missing",
                    message="Problem text references a diagram but no image is attached",
                    snippet=None,
                )
            )
        elif flag == "manual_review_required":
            issues.append(
                ValidationIssue(
                    code="manual_review_required",
                    message=(
                        "Raw OCR was too degraded for automated symbol restoration to be "
                        "trusted - needs a human check against the source PDF"
                    ),
                    snippet=None,
                )
            )
    return issues


def _dedupe_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[str] = set()
    unique: list[ValidationIssue] = []
    for issue in issues:
        if issue.code in seen:
            continue
        seen.add(issue.code)
        unique.append(issue)
    return unique


def validate_text(text: str, *, flags: list[str] | None = None) -> list[ValidationIssue]:
    issues = _issues_from_text(text)
    if flags:
        issues.extend(_issues_from_flags(flags))
    return _dedupe_issues(issues)


def validate_record(record: ProblemRecord) -> list[ValidationIssue]:
    """Return deterministic parse errors for a problem record."""
    texts = [record.body_md]
    texts.extend(sp.text for sp in record.subparts)
    issues: list[ValidationIssue] = []
    for text in texts:
        issues.extend(_issues_from_text(text))
    issues.extend(_issues_from_flags(record.flags))

    ratio = _content_loss_ratio(record)
    if ratio is not None and ratio < CONTENT_LOSS_THRESHOLD:
        issues.append(
            ValidationIssue(
                code="content_loss_vs_raw",
                message=(
                    f"Only {ratio:.0%} of original wording survived repair - "
                    "likely truncated or over-summarized by an LLM pass"
                ),
                snippet=None,
            )
        )

    return _dedupe_issues(issues)


def sync_flags_from_errors(errors: list[ValidationIssue], attach_flags: list[str]) -> list[str]:
    """Mirror error codes into flags while preserving attach-specific flags."""
    codes = {issue.code for issue in errors}
    flags = list(attach_flags)
    for code in codes:
        if code not in flags:
            flags.append(code)
    return flags


def apply_validation(record: ProblemRecord) -> ProblemRecord:
    """Populate body_md_raw, errors, and flags on a record."""
    if record.body_md_raw is None:
        record.body_md_raw = record.body_md
    attach_flags = [
        f
        for f in record.flags
        if f.startswith("missing_image:") or f == "manual_review_required"
    ]
    if (
        body_expects_attached_figure(record.body_md)
        and not extract_image_refs(record.body_md)
        and not record.images
    ):
        attach_flags.append("expected_image_missing")
    record.flags = attach_flags
    record.errors = validate_record(record)
    record.flags = sync_flags_from_errors(record.errors, attach_flags)
    return record


def filter_symbol_restore_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return [issue for issue in issues if issue.code in SYMBOL_REPAIR_CODES]


def needs_symbol_restore(issues: list[ValidationIssue]) -> bool:
    return any(issue.code in SYMBOL_REPAIR_CODES for issue in issues)


def needs_full_llm_repair(issues: list[ValidationIssue]) -> bool:
    return any(issue.code in FULL_LLM_REPAIR_CODES for issue in issues)
