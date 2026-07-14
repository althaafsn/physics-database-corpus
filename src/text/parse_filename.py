from __future__ import annotations

import re
from pathlib import Path

from src.schema import DocumentMeta, MetadataOverrides

STANDARD_RE = re.compile(
    r"^Soal (?P<level>OSK|OSP|OSN) Fisika SMA (?P<year>\d{4})(?: \((?P<variant>\d+)\))?\.pdf$",
    re.IGNORECASE,
)
FINAL_RE = re.compile(
    r"^Soal (?P<level>OSK|OSP|OSN) Fisika SMA Final (?P<year>\d{4})\.pdf$",
    re.IGNORECASE,
)
SEMIFINAL_RE = re.compile(
    r"^Soal (?P<level>OSK|OSP|OSN) Fisika SMA (?P<year>\d{4}) \+ Semifinal\.pdf$",
    re.IGNORECASE,
)

# Generic filename patterns (PDF-agnostic olympiad naming).
GENERIC_LEVEL_YEAR_RE = re.compile(
    r"(?P<level>OSK|OSP|OSN)[^\d]{0,40}(?P<year>20\d{2})",
    re.IGNORECASE,
)
GENERIC_VARIANT_RE = re.compile(r"\((?P<variant>\d+)\)")
YEAR_ONLY_RE = re.compile(r"(20\d{2})")

# Markdown title patterns from converted problem sets.
MD_TITLE_STANDARD_RE = re.compile(
    r"Soal\s+(?P<level>OSK|OSP|OSN)\s+Fisika\s+SMA"
    r"(?:\s+Final|\s+\+\s+Semifinal)?\s*(?P<year>20\d{2})?",
    re.IGNORECASE,
)
MD_LEVEL_RE = re.compile(r"\b(OSK|OSP|OSN)\b", re.IGNORECASE)


def slug_from_stem(stem: str) -> str:
    return stem.strip()


def _meta_from_filename(name: str, pdf_path: Path) -> DocumentMeta | None:
    for pattern, round_name in (
        (FINAL_RE, "final"),
        (SEMIFINAL_RE, "semifinal"),
        (STANDARD_RE, None),
    ):
        match = pattern.match(name)
        if not match:
            continue
        groups = match.groupdict()
        variant = groups.get("variant")
        return DocumentMeta(
            slug=slug_from_stem(Path(name).stem),
            source_pdf=str(pdf_path),
            level=groups["level"].upper(),
            year=int(groups["year"]),
            round=round_name,
            variant=int(variant) if variant else None,
            title=Path(name).stem,
            meta_source="filename",
        )

    stem = Path(name).stem
    generic = GENERIC_LEVEL_YEAR_RE.search(stem)
    if generic:
        variant_match = GENERIC_VARIANT_RE.search(stem)
        round_name = None
        lower = stem.lower()
        if "final" in lower:
            round_name = "final"
        elif "semifinal" in lower:
            round_name = "semifinal"
        return DocumentMeta(
            slug=slug_from_stem(stem),
            source_pdf=str(pdf_path),
            level=generic.group("level").upper(),
            year=int(generic.group("year")),
            round=round_name,
            variant=int(variant_match.group("variant")) if variant_match else None,
            title=stem,
            meta_source="filename_generic",
        )

    year_match = YEAR_ONLY_RE.search(stem)
    if year_match:
        return DocumentMeta(
            slug=slug_from_stem(stem),
            source_pdf=str(pdf_path),
            level=None,
            year=int(year_match.group(1)),
            title=stem,
            meta_source="filename_year_only",
        )

    return None


def _meta_from_markdown(md_text: str, *, slug: str, pdf_path: Path) -> DocumentMeta | None:
    head = md_text[:4000]
    title_match = MD_TITLE_STANDARD_RE.search(head)
    if title_match:
        level = title_match.group("level").upper()
        year_raw = title_match.group("year")
        year = int(year_raw) if year_raw else None
        round_name = None
        if re.search(r"\bfinal\b", title_match.group(0), re.I):
            round_name = "final"
        elif "semifinal" in title_match.group(0).lower():
            round_name = "semifinal"
        return DocumentMeta(
            slug=slug,
            source_pdf=str(pdf_path),
            level=level,
            year=year,
            round=round_name,
            title=title_match.group(0).strip(),
            meta_source="markdown_title",
        )

    level_match = MD_LEVEL_RE.search(head)
    year_match = YEAR_ONLY_RE.search(head)
    if level_match or year_match:
        return DocumentMeta(
            slug=slug,
            source_pdf=str(pdf_path),
            level=level_match.group(1).upper() if level_match else None,
            year=int(year_match.group(1)) if year_match else None,
            title=slug,
            meta_source="markdown_partial",
        )
    return None


def parse_pdf_filename(name: str, base_dir: str | Path = "all_pdf") -> DocumentMeta:
    """Parse competition metadata from a PDF filename (strict patterns first)."""
    pdf_path = Path(base_dir) / name
    meta = _meta_from_filename(name, pdf_path)
    if meta is not None:
        return meta
    stem = Path(name).stem
    return DocumentMeta(
        slug=slug_from_stem(stem),
        source_pdf=str(pdf_path),
        title=stem,
        meta_source="filename_fallback",
    )


def parse_document(
    slug: str,
    *,
    pdf_dir: Path,
    md_text: str | None = None,
    pdf_path: Path | None = None,
    overrides: MetadataOverrides | None = None,
) -> DocumentMeta:
    """Resolve document metadata from filename, markdown, and optional CLI overrides."""
    resolved_pdf = pdf_path or (pdf_dir / f"{slug}.pdf")
    meta = _meta_from_filename(resolved_pdf.name, resolved_pdf)

    if meta is None:
        meta = DocumentMeta(
            slug=slug_from_stem(slug),
            source_pdf=str(resolved_pdf),
            title=slug,
            meta_source="unknown",
        )

    if md_text:
        md_meta = _meta_from_markdown(md_text, slug=slug, pdf_path=resolved_pdf)
        if md_meta:
            meta = _merge_meta(meta, md_meta, prefer="markdown")

    if overrides:
        meta = _apply_overrides(meta, overrides)

    return meta


def _merge_meta(base: DocumentMeta, incoming: DocumentMeta, *, prefer: str) -> DocumentMeta:
    data = base.model_dump()
    incoming_data = incoming.model_dump()
    filename_is_authoritative = base.meta_source.startswith("filename")
    for key in ("level", "year", "round", "variant", "title"):
        if incoming_data.get(key) is not None:
            # A cover heading can be copied from the wrong exam (a common
            # PDF-export error). Once the filename matched a strict exam
            # pattern, keep its year/level/round/variant/title and only fill
            # genuinely missing fields from Markdown.
            if data.get(key) is None or (not filename_is_authoritative and prefer == "markdown"):
                data[key] = incoming_data[key]
    if not filename_is_authoritative and incoming_data.get("meta_source", "").startswith("markdown"):
        data["meta_source"] = incoming_data["meta_source"]
    return DocumentMeta(**data)


def _apply_overrides(meta: DocumentMeta, overrides: MetadataOverrides) -> DocumentMeta:
    data = meta.model_dump()
    for field in ("level", "year", "round", "variant", "title"):
        val = getattr(overrides, field, None)
        if val is not None:
            data[field] = val
    data["meta_source"] = "cli_override"
    return DocumentMeta(**data)


def parse_folder_name(
    folder_name: str,
    *,
    pdf_dir: Path | None = None,
    md_text: str | None = None,
    overrides: MetadataOverrides | None = None,
) -> DocumentMeta:
    """Parse metadata from marker output folder name and optional markdown."""
    pdf_dir = pdf_dir or Path("all_pdf")
    return parse_document(
        folder_name,
        pdf_dir=pdf_dir,
        md_text=md_text,
        overrides=overrides,
    )
