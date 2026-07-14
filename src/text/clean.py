from __future__ import annotations

import re
from pathlib import Path

from src.bronze.doc_conflicts import resolve_align_conflicts
from src.text.attach_images import extract_image_refs
from src.math.math_normalize import fix_json_control_artifacts, normalize_problem_symbols
from src.math.symbol_heuristics import apply_symbol_heuristics
from src.schema import ProblemRecord, SubPart
from src.text.split_problems import SUBPART_RE, extract_subparts

# Shared with validate.py — lines/blocks that are never problem content.
FOOTER_LINE_RE = re.compile(
    r"^\s*(?:"
    r"Dimensi Sains[\s\-–—]*Ahmad Basyir Najwan"
    r"|(?:\[)?(?:https?://)?(?:www\.)?basyiralbanjari\.wordpress\.com[^\n]*"
    r"|www\.basyiralbanjari\.wordpress\.com[^\n]*"
    r"|(?:Youtube|Tiktok|Instagram)[^\n]*(?:Dimensi Sains|dimensi\.sains|dimensisains)[^\n]*"
    r"|Instagram:\s*dimensisains\.official[^\n]*"
    r"|(?:WA|Whatsapp):\s*089659856821[^\n]*"
    r"|089659856821\s*/\s*085217499402"
    r"|085217499402"
    r"|Naskah soal ini diketik ulang oleh Ahmad Basyir Najwan\.?"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

FOOTER_INLINE_RE = re.compile(
    r"Dimensi Sains|basyiralbanjari|dimensisains|089659856821|085217499402",
    re.IGNORECASE,
)

INLINE_FOOTER_RE = re.compile(
    r"(?:"
    r"#{1,6}\s*Dimensi Sains[\s\-–—]*Ahmad Basyir Najwan"
    r"|<u>[^<]*basyiralbanjari[^<]*</u>\s*\|\s*Youtube:\s*Dimensi Sains"
    r"|\[?(?:https?://)?(?:www\.)?basyiralbanjari\.wordpress\.com\]?(?:\([^)]*\))?"
    r"\s*\|\s*Youtube:\s*Dimensi Sains"
    r"|Instagram:\s*dimensisains\.official\s*\|\s*WA:\s*089659856821/085217499402"
    r"|(?:\|\s*)?Youtube:\s*Dimensi Sains"
    r")",
    re.IGNORECASE,
)

FZTI_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*#*\s*Program Persiapan OSN Fisika[\s\S]*\Z",
    re.IGNORECASE,
)

# Promo tail appended to real problems or split as fake "problems" (OSK 2020+, OSP 2019+).
PROMO_TAIL_START = (
    r"Program Persiapan OSN Fisika"
    r"|Apa itu FZTI\?"
    r"|From Zero to Infinity(?:\s+adalah|\s*$|\s+—|\s+-)"
    r"|(?:#{1,6}\s*)?\*{0,2}Target Peserta\*{0,2}"
    r"|(?:#{1,6}\s*)?Metode Pembelajaran"
    r"|(?:#{1,6}\s*)?\*{0,2}Diskusi & Pendampingan\*{0,2}"
    r"|(?:#{1,6}\s*)?Pilihan Paket"
    r"|(?:#{1,6}\s*)?🎉 Promo Awal Tahun"
    r"|#?\s*Waktu & Pendaftaran"
    r"|★\s*Daftar via Google Form"
    r"|Diskusi langsung dengan \*\*Kak Basyir\*\*"
    r"|(?:-\s*)?•?\s*Paket I:\s*FZTI"
    r"|#?\s*Unduh Buku \*\*Panduan FZTI\*\*"
    r"|Ingin mencari teman untuk daftar kolektif"
    r"|bit\.ly/(?:DaftarFZTI|BukuPanduanFZTI|DaftarKolektifFZTI)"
)

PROMO_TAIL_RE = re.compile(
    rf"(?:^|\n)\s*(?:{PROMO_TAIL_START})[\s\S]*\Z",
    re.IGNORECASE | re.MULTILINE,
)

PROMO_ONLY_START_RE = re.compile(
    r"^\s*(?:"
    r"(?:-\s*)?•?\s*Siswa SMP Kelas 9 dan SMA Kelas 10-11"
    r"|(?:-\s*)?·\s*Online & fleksibel"
    r"|(?:-\s*)?·\s*Modul PDF lengkap"
    r"|Diskusi langsung dengan \*\*Kak Basyir\*\*"
    r"|(?:-\s*)?•?\s*Paket I:\s*FZTI"
    r"|(?:-\s*)?•?\s*Diskon tambahan Rp100"
    r"|#?\s*Waktu & Pendaftaran"
    r"|Ingin mencari teman untuk daftar kolektif"
    r"|Program ini cocok untuk kamu yang"
    r"|From Zero to Infinity adalah program"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

PROMO_INLINE_RE = re.compile(
    r"bit\.ly/(?:DaftarFZTI|BukuPanduanFZTI|DaftarKolektifFZTI)\S*",
    re.IGNORECASE,
)

PROMO_LINE_RE = re.compile(
    r"^\s*(?:"
    r"(?:-\s*)?•?\s*Paket (?:I|II|III):\s*(?:FZTI|LOOF)"
    r"|(?:-\s*)?•?\s*Diskon tambahan Rp100"
    r"|(?:-\s*)?•?\s*Berlaku hingga \*\*akhir Februari"
    r"|(?:-\s*)?•?\s*Syarat dan Ketentuan Berlaku"
    r"|(?:-\s*)?Tersedia diskon besar untuk pendaftaran kolektif"
    r"|(?:-\s*)?Prersedia diskon besar untuk pendaftaran kolektif"
    r"|★\s*Daftar via Google Form[^\n]*"
    r"|#?\s*Unduh Buku \*\*Panduan FZTI\*\*[^\n]*"
    r"|Bukan cuma nonton materi — tapi dilatih cara berpikir fisikawan\.?"
    r")\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Shared with validate.py — FZTI signup / course promo markers.
PROMO_CONTAMINATION_RE = re.compile(
    r"bit\.ly/(?:DaftarFZTI|BukuPanduanFZTI|DaftarKolektifFZTI)"
    r"|(?:^|\n)\s*(?:-\s*)?•?\s*Paket I:\s*FZTI"
    r"|From Zero to Infinity"
    r"|Kak Basyir[^\n]*Focus Group"
    r"|Promo Awal Tahun"
    r"|Apa itu FZTI\?"
    r"|Program Persiapan OSN Fisika"
    r"|(?:-\s*)?•?\s*Siswa SMP Kelas 9 dan SMA Kelas 10-11"
    r"|David Morin Classical Mechanics",
    re.IGNORECASE | re.MULTILINE,
)

PROBLEM_CONTENT_RE = re.compile(
    r"\([a-d]\)|Hitung|Tentukan|Carilah|Diketahui|Sebuah|diagram|!\[\]|"
    r"bermassa|kecepatan|pegas|gravitasi",
    re.IGNORECASE,
)

WATERMARK_IMAGE_LINE_RE = re.compile(
    r"^\s*!\[\]\(_page_\d+_Picture_(?:0|1|4|5|6)\.(?:jpeg|jpg|png)\)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

HTML_SUP_RE = re.compile(r"<sup>(.*?)</sup>", re.IGNORECASE | re.DOTALL)
HTML_SUB_RE = re.compile(r"<sub>(.*?)</sub>", re.IGNORECASE | re.DOTALL)

NOUN_ORPHAN_SUP_RE = re.compile(
    r"\b(bermassa|konstanta pegas|kecepatan|partikel|balok|pegas|massa)\s*<sup>([^<]+)</sup>",
    re.IGNORECASE,
)

LEADING_ORPHAN_SUP_RE = re.compile(
    r"(?:^|\n)\s*<sup>([^<]+)</sup>\s*=",
    re.IGNORECASE | re.MULTILINE,
)

MISSING_G_RE = re.compile(r"=\s*10\s*m/s\s*2", re.IGNORECASE)
GRAVITY_PHRASE_RE = re.compile(
    r"(?:percepatan\s+gravitasi(?:\s+(?:di\s+tempat\s+itu|bumi))?|Anggap\s+percepatan\s+gravitasi)"
    r"(?:\s+(?:adalah|konstan|di\s+tempat\s+itu))?\s*=\s*10",
    re.IGNORECASE,
)
USE_G_RE = re.compile(r"\bGunakan\s+=\s*10\b", re.IGNORECASE)

MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _has_g_symbol(text: str, pos: int) -> bool:
    prefix = text[max(0, pos - 30) : pos]
    return bool(re.search(r"\$g\$|\bg\s*=|\bg\s", prefix))


def _latex_sup(content: str) -> str:
    inner = content.strip()
    if not inner:
        return ""
    if len(inner) == 1:
        return f"$^{inner}$"
    return f"$^{{{inner}}}$"


def _latex_sub(content: str) -> str:
    inner = content.strip()
    if not inner:
        return ""
    if len(inner) == 1:
        return f"$_{inner}$"
    return f"$_{inner}$"


def _fix_noun_orphan_superscripts(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        noun = match.group(1)
        val = match.group(2).strip()
        key = noun.lower()
        if key in {"massa", "bermassa", "balok", "partikel", "pegas"}:
            return f"{noun} $m_{{{val}}}$"
        if key == "kecepatan":
            return f"kecepatan $v_{{{val}}}$"
        if key == "konstanta pegas":
            return f"konstanta pegas $k_{{{val}}}$"
        return f"{noun} $x_{{{val}}}$"

    return NOUN_ORPHAN_SUP_RE.sub(repl, text)


def _fix_leading_orphan_superscripts(text: str) -> str:
    return LEADING_ORPHAN_SUP_RE.sub(r"\n$m_{\1}$ =", text)


def _html_to_latex(text: str) -> str:
    text = HTML_SUP_RE.sub(lambda m: _latex_sup(m.group(1)), text)
    text = HTML_SUB_RE.sub(lambda m: _latex_sub(m.group(1)), text)
    return text


def _fix_gravity_constant(text: str) -> str:
    text = GRAVITY_PHRASE_RE.sub(
        lambda m: m.group(0).split("=")[0].rstrip() + " = $g = 10$",
        text,
    )
    text = USE_G_RE.sub("Gunakan $g = 10$", text)

    def repl(match: re.Match[str]) -> str:
        if _has_g_symbol(text, match.start()):
            return match.group(0)
        return "$g = 10$ m/s$^2$"

    return MISSING_G_RE.sub(repl, text)


def _is_promo_only_body(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if PROMO_ONLY_START_RE.match(stripped):
        return True
    if len(stripped) < 900 and PROMO_CONTAMINATION_RE.search(stripped):
        return not PROBLEM_CONTENT_RE.search(stripped)
    return False


def _remove_fzti_promo(text: str) -> str:
    text = FZTI_BLOCK_RE.sub("", text)
    text = PROMO_TAIL_RE.sub("", text)
    if _is_promo_only_body(text):
        return ""
    text = PROMO_INLINE_RE.sub("", text)
    text = PROMO_LINE_RE.sub("", text)
    return text


def _remove_ads_and_watermarks(text: str) -> str:
    text = _remove_fzti_promo(text)
    text = WATERMARK_IMAGE_LINE_RE.sub("", text)
    text = FOOTER_LINE_RE.sub("", text)
    text = INLINE_FOOTER_RE.sub("", text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.strip()
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text


def clean_text(text: str) -> str:
    """Deterministic cleanup: ads, watermarks, HTML math, and safe symbol fixes."""
    if not text or not text.strip():
        return text
    text = fix_json_control_artifacts(text)
    text = resolve_align_conflicts(text)
    text = _remove_ads_and_watermarks(text)
    text = _fix_noun_orphan_superscripts(text)
    text = _fix_leading_orphan_superscripts(text)
    text = _html_to_latex(text)
    text = apply_symbol_heuristics(text)
    text = _fix_gravity_constant(text)
    text = normalize_problem_symbols(text)
    return _normalize_whitespace(text)


def _strip_inline_subparts(body: str | None, subparts: list[SubPart]) -> tuple[str | None, bool]:
    """Drop the inline ``(a) … (b) …`` block from a body when those subparts are
    already held in ``subparts`` (rendered separately as the parts list).

    Returns ``(new_body, changed)``. The body is only stripped when the subpart
    content is preserved in the list, so no text is ever lost.
    """
    if not body or not SUBPART_RE.search(body):
        return body, False
    if not subparts:
        return body, False
    stripped = _normalize_whitespace(SUBPART_RE.sub("", body))
    if stripped == body:
        return body, False
    return stripped, True


def strip_duplicate_subparts(record: ProblemRecord) -> bool:
    """Remove subpart text that is duplicated both inline in the body and in the
    ``subparts`` list, so the reader/preview render each subpart only once.

    Works on the Indonesian (``body_md``/``subparts``) and English
    (``body_md_en``/``subparts_en``) variants. Idempotent: once an inline block
    is removed, a second pass finds nothing to strip. Returns ``True`` when the
    record was modified.
    """
    changed = False

    if record.body_md and SUBPART_RE.search(record.body_md) and not record.subparts:
        record.subparts = [SubPart(**sp) for sp in extract_subparts(record.body_md)]
    new_body, did = _strip_inline_subparts(record.body_md, record.subparts)
    if did:
        record.body_md = new_body
        changed = True

    if record.body_md_en and SUBPART_RE.search(record.body_md_en) and not record.subparts_en:
        record.subparts_en = [SubPart(**sp) for sp in extract_subparts(record.body_md_en)]
    new_body_en, did_en = _strip_inline_subparts(record.body_md_en, record.subparts_en)
    if did_en:
        record.body_md_en = new_body_en
        changed = True

    return changed


def dedupe_body_image_refs(body_md: str) -> tuple[str, bool]:
    """Remove repeated ![](same.jpeg) lines while preserving first occurrence order."""
    seen: set[str] = set()
    changed = False
    lines: list[str] = []
    for line in body_md.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"!\[\]\(([^)]+)\)", stripped)
        if match:
            ref = match.group(1)
            if ref in seen:
                changed = True
                continue
            seen.add(ref)
        lines.append(line)
    return "\n".join(lines), changed


def sync_images_with_body(record: ProblemRecord) -> None:
    refs = set(extract_image_refs(record.body_md))
    record.images = [img for img in record.images if img.filename in refs]


def clean_record(record: ProblemRecord) -> ProblemRecord:
    """Apply deterministic cleaning to a problem record in place."""
    if record.body_md_raw is None:
        record.body_md_raw = record.body_md

    from src.text.attach_images import sanitize_image_refs_in_body

    record.body_md = sanitize_image_refs_in_body(clean_text(record.body_md))
    record.body_md, _ = dedupe_body_image_refs(record.body_md)
    record.subparts = [SubPart(**sp) for sp in extract_subparts(record.body_md)]
    sync_images_with_body(record)
    return record


def finalize_record_after_repair(
    record: ProblemRecord,
    output_folder: Path,
    assets_dir: Path,
) -> ProblemRecord:
    """Re-clean and repair images after LLM repair may have restored watermark refs."""
    from src.repair.repair_images import repair_record_images

    from src.text.attach_images import sanitize_image_refs_in_body

    record.body_md = sanitize_image_refs_in_body(clean_text(record.body_md))
    record.subparts = [SubPart(**sp) for sp in extract_subparts(record.body_md)]
    repair_record_images(record, output_folder, assets_dir)
    return record
