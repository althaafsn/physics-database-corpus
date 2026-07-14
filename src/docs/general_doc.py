"""Marker + pdftotext hybrid parser for general (non-exam) documents.

Mirrors the corpus hybrid philosophy (:mod:`src.bronze.hybrid_bronze`,
:mod:`src.solutions.typed_markdown`) but operates on *arbitrary* documents at the
section level instead of the problem-number level:

  Marker markdown (layout, headings, figures, tables)
        +
  pdftotext text layer (faithful body text for typed PDFs)
        ->  clean Markdown

For typed documents (e.g. arXiv papers) the embedded text layer is the
highest-fidelity source for prose and math, while Marker supplies the structure
(headings) and figure references it could not recover from raw text. When
Marker's own OCR/text is degraded, the pdftotext body takes over entirely.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.bronze.pdf_text import extract_pdf_text, has_usable_text_layer, strip_pdf_footers
from src.text.attach_images import extract_image_refs

# ---------------------------------------------------------------------------
# Boilerplate / noise patterns common in research papers (esp. arXiv preprints)
# ---------------------------------------------------------------------------

# arXiv side stamp, e.g. "arXiv:2606.09498v1 [cs.CL] 8 Jun 2026"
_ARXIV_STAMP_RE = re.compile(
    r"arXiv:\d{4}\.\d{4,5}(v\d+)?\s*(?:\[[\w.\-]+\])?\s*\d{1,2}\s+\w{3}\s+\d{4}",
    re.IGNORECASE,
)
# A line that is only whitespace + a page number (1-3 digits), optionally with a
# lone trailing/leading dot or comma.
_PAGE_NUMBER_LINE_RE = re.compile(r"^\s*\d{1,3}\s*\.?\s*$")
# Repeated publisher / venue one-liners.
_RUNNING_HEADER_RE = re.compile(
    r"^\s*(?:Preprint\.?|Submitted to .*|To appear in .*|Draft v\d+)\s*$",
    re.IGNORECASE,
)
# LaTeX inline/display delimiters that Marker sometimes fails to convert.
_INLINE_MATH_OPEN_RE = re.compile(r"(?<!\\)\\\(")
_INLINE_MATH_CLOSE_RE = re.compile(r"(?<!\\)\\\)")
_DISPLAY_MATH_OPEN_RE = re.compile(r"(?<!\\)\\\[")
_DISPLAY_MATH_CLOSE_RE = re.compile(r"(?<!\\)\\\]")
# Replacement-character / private-use junk signalling OCR garbling.
_GARBLE_RE = re.compile(r"[\ufffd\ue000-\uf8ff]")
# Collapsed blank lines.
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _strip_arxiv_stamps(text: str) -> str:
    return _ARXIV_STAMP_RE.sub("", text)


def _normalize_math_delimiters(text: str) -> str:
    """Convert stray \\(…\\) and \\[…\\] to $…$ / $$…$$ (LaTeX → Markdown math)."""
    text = _DISPLAY_MATH_OPEN_RE.sub("$$", text)
    text = _DISPLAY_MATH_CLOSE_RE.sub("$$", text)
    text = _INLINE_MATH_OPEN_RE.sub("$", text)
    text = _INLINE_MATH_CLOSE_RE.sub("$", text)
    return text


def _drop_boilerplate_lines(text: str) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if _PAGE_NUMBER_LINE_RE.match(line) and len(line.strip()) <= 4:
            continue
        if _RUNNING_HEADER_RE.match(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_general_markdown(md: str, *, slug: str = "") -> str:
    """Normalize boilerplate and math delimiters in parsed document markdown.

    Pure string transform — no I/O — so it is straightforward to unit-test.
    """
    if not md or not md.strip():
        return ""
    cleaned = _strip_arxiv_stamps(md)
    cleaned = _drop_boilerplate_lines(cleaned)
    cleaned = _normalize_math_delimiters(cleaned)
    # Trim trailing whitespace on each line and collapse blank runs.
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = _BLANK_RUN_RE.sub("\n\n", cleaned)
    return cleaned.strip() + "\n"


# ---------------------------------------------------------------------------
# Degradation detection — when Marker's own text is worse than pdftotext.
# ---------------------------------------------------------------------------


def marker_degraded(marker_md: str) -> bool:
    """Heuristic: True when Marker output looks OCR-garbled and should be replaced.

    Triggers on a high density of replacement/private-use characters or a very
    low ratio of real words — both signs that Marker struggled with the page.
    """
    if not marker_md or not marker_md.strip():
        return True
    garble = len(_GARBLE_RE.findall(marker_md))
    if garble >= 12:
        return True
    words = re.findall(r"[A-Za-z][A-Za-z\-']{1,}", marker_md)
    chars = len(marker_md)
    if chars > 400 and len(words) / chars < 0.04:
        return True
    return False


# ---------------------------------------------------------------------------
# Hybrid merge: Marker structure/images + pdftotext body reconciliation.
# ---------------------------------------------------------------------------


def _attach_marker_images(target_md: str, marker_md: str) -> str:
    """Append Marker figure refs that the target body is missing."""
    refs = extract_image_refs(marker_md)
    if not refs:
        return target_md
    body = target_md.rstrip()
    existing = set(extract_image_refs(body))
    extras = [f"![]({ref})" for ref in refs if ref not in existing]
    if not extras:
        return body
    return body + "\n\n" + "\n\n".join(extras)


def merge_general_hybrid(marker_md: str, pdf_text: str, *, slug: str = "") -> str:
    """Combine Marker markdown with the pdftotext text layer for a general doc.

    Policy:
      * If Marker output is clean → keep it (structure + images), just cleaned.
      * If Marker output is degraded but pdftotext is usable → take the pdftotext
        body (cleaned) and re-attach Marker's figure references.
      * If only one source exists → use it, cleaned.
    """
    marker_md = marker_md or ""
    pdf_text = pdf_text or ""
    has_marker = bool(marker_md.strip())
    has_pdf = bool(pdf_text.strip())

    if has_marker and not marker_degraded(marker_md):
        merged = clean_general_markdown(marker_md, slug=slug)
        return _attach_marker_images(merged, marker_md)

    if has_pdf:
        body = clean_general_markdown(pdf_text, slug=slug)
        if has_marker:
            body = _attach_marker_images(body, marker_md)
        return body

    if has_marker:
        return clean_general_markdown(marker_md, slug=slug)
    return ""


# ---------------------------------------------------------------------------
# Marker subprocess + orchestration
# ---------------------------------------------------------------------------


def _resolve_marker_bin() -> str | None:
    """Return an executable path for marker_single if available, else None."""
    from src.bronze.bronze_convert import marker_single_path

    return str(marker_single_path()) if marker_single_path() else None


def _marker_env(*, force_cpu: bool) -> dict[str, str]:
    env = os.environ.copy()
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def convert_general_doc_marker(
    pdf_path: Path,
    out_dir: Path,
    *,
    timeout_s: float = 5400.0,
    disable_ocr: bool = True,
    force_cpu: bool = False,
) -> tuple[Path | None, str]:
    """Run marker_single on a PDF; return (markdown_path, detail).

    OCR is disabled by default for general documents because typed papers carry a
    faithful text layer and OCR is slow + error-prone on dense math.
    """
    marker_bin = _resolve_marker_bin()
    if marker_bin is None:
        return None, "marker_single not installed"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        marker_bin,
        str(pdf_path.resolve()),
        "--output_dir",
        str(out_dir.resolve()),
        "--disable_tqdm",
    ]
    if disable_ocr:
        cmd.append("--disable_ocr")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=_marker_env(force_cpu=force_cpu),
        )
    except subprocess.TimeoutExpired as exc:
        return None, f"timeout after {exc.timeout}s"
    md_path = out_dir / pdf_path.stem / f"{pdf_path.stem}.md"
    if proc.returncode == 0 and md_path.is_file():
        return md_path, "marker"
    err = (proc.stderr or proc.stdout or "").strip()
    if len(err) > 400:
        err = err[:400] + "..."
    return None, err or f"exit code {proc.returncode}"


@dataclass(frozen=True)
class GeneralDocResult:
    slug: str
    pdf_path: Path
    out_dir: Path
    md_path: Path
    method: str  # 'marker' | 'hybrid' | 'pdftotext' | 'failed'
    detail: str = ""
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pdf_path"] = str(self.pdf_path)
        d["out_dir"] = str(self.out_dir)
        d["md_path"] = str(self.md_path)
        return d


def _pdf_body_text(pdf_path: Path) -> str:
    if not has_usable_text_layer(pdf_path, min_words=40, min_math_italic=0):
        return ""
    return strip_pdf_footers(extract_pdf_text(pdf_path))


def convert_general_doc(
    pdf_path: Path,
    *,
    out_dir: Path,
    use_marker: bool = True,
    timeout_s: float = 5400.0,
    force_cpu: bool = False,
) -> GeneralDocResult:
    """Parse a general document into clean Markdown via the Marker+pdftotext hybrid.

    Writes ``<out_dir>/<slug>/<slug>.md`` and a ``<slug>_meta.json`` sidecar.
    """
    slug = pdf_path.stem
    folder = out_dir / slug
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / f"{slug}.md"

    pdf_text = _pdf_body_text(pdf_path)
    marker_md = ""

    if use_marker:
        marker_md_path, detail = convert_general_doc_marker(
            pdf_path,
            out_dir,
            timeout_s=timeout_s,
            disable_ocr=True,
            force_cpu=force_cpu,
        )
        if marker_md_path is not None:
            marker_md = marker_md_path.read_text(encoding="utf-8")
        else:
            # Fall back to whatever Marker previously produced, if anything.
            prev = folder / f"{slug}.md"
            if prev.is_file():
                marker_md = prev.read_text(encoding="utf-8")

    merged = merge_general_hybrid(marker_md, pdf_text, slug=slug)

    if not merged.strip():
        method = "failed"
        detail = "no text recovered from marker or pdftotext"
    elif marker_md and marker_degraded(marker_md) and pdf_text.strip():
        method = "hybrid"
        detail = "pdftotext body + marker images (marker degraded)"
    elif marker_md:
        method = "marker"
        detail = "marker markdown (clean)"
    elif pdf_text.strip():
        method = "pdftotext"
        detail = "pdftotext text layer only"
    else:
        method = "failed"
        detail = "no source"

    if method != "failed":
        md_path.write_text(merged, encoding="utf-8")

    words = len(re.findall(r"[A-Za-z][A-Za-z\-']{1,}", merged))
    stats = {
        "chars": len(merged),
        "words": words,
        "lines": merged.count("\n") + 1 if merged else 0,
        "has_marker": bool(marker_md.strip()),
        "has_pdf_text_layer": bool(pdf_text.strip()),
        "marker_degraded": marker_degraded(marker_md) if marker_md else False,
    }

    meta_path = folder / f"{slug}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "slug": slug,
                "source_pdf": str(pdf_path),
                "method": method,
                "detail": detail,
                "stats": stats,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    return GeneralDocResult(
        slug=slug,
        pdf_path=pdf_path,
        out_dir=out_dir,
        md_path=md_path if method != "failed" else folder / f"{slug}.md",
        method=method,
        detail=detail,
        stats=stats,
    )
