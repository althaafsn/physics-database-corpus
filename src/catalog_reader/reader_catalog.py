"""Map catalog-eligible gold records to the public reader JSON shape."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.text.attach_images import IMAGE_REF_RE, normalize_image_ref
from src.catalog import is_in_locale_catalog, record_content_locale
from src.record_store import load_jsonl
from src.schema import ProblemRecord, SubPart

TOPIC_MAP = {
    "mechanics": "mechanics",
    "electromagnetism": "electromagnetism",
    "thermodynamics": "thermodynamics",
    "waves_optics": "waves",
    "modern_physics": "modern",
    "mixed": "mixed",
}


def asset_url(relative_path: str, prefix: str = "/assets") -> str:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if normalized.startswith("assets/"):
        normalized = normalized[len("assets/") :]
    return f"{prefix}/{normalized}"


def _image_mapping(rec: ProblemRecord) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for img in rec.images:
        url = asset_url(img.path)
        mapping[img.filename] = url
        mapping[img.path.replace("\\", "/")] = url
    return mapping


def _default_asset_url(rec: ProblemRecord, filename: str) -> str:
    level = rec.level or "UNKNOWN"
    year = rec.year if rec.year is not None else 0
    num = f"{rec.problem_number:02d}"
    return asset_url(f"assets/{level}/{year}/{num}/{filename}")


def rewrite_markdown_images(
    text: str,
    rec: ProblemRecord,
    *,
    asset_prefix: str = "/assets",
) -> str:
    mapping = _image_mapping(rec)

    def repl(match: re.Match[str]) -> str:
        ref = normalize_image_ref(match.group(1))
        clean = ref.replace("\\", "/")
        mapped = mapping.get(clean) or mapping.get(Path(clean).name)
        if not mapped and clean.startswith("assets/"):
            mapped = asset_url(clean, asset_prefix)
        if not mapped and Path(clean).name.startswith("_page_"):
            mapped = _default_asset_url(rec, Path(clean).name)
        return f"![]({mapped})" if mapped else f"![]({ref})"

    return IMAGE_REF_RE.sub(repl, text)


def map_topic(raw: str) -> str:
    return TOPIC_MAP.get(raw.lower(), "mechanics")


def map_problem_record(rec: ProblemRecord, locale: str = "id") -> dict:
    content = record_content_locale(rec)
    is_en_native = content == "en"
    has_translation = bool(rec.body_md_en and rec.body_md_en.strip())
    has_english = is_en_native or has_translation

    if locale == "en" and is_en_native:
        body_source = rec.body_md
        title_source = rec.title
        subparts: list[SubPart] = rec.subparts
    elif locale == "en" and has_translation:
        body_source = rec.body_md_en or ""
        title_source = (
            rec.title_en.strip()
            if rec.title_en and rec.title_en.strip()
            else rec.title
        )
        subparts = rec.subparts_en if rec.subparts_en else rec.subparts
    else:
        body_source = rec.body_md
        title_source = rec.title
        subparts = rec.subparts

    body = rewrite_markdown_images(body_source or "", rec)
    first_image = rec.images[0] if rec.images else None

    parts_en = None
    if rec.subparts_en:
        parts_en = [
            {
                "label": sp.label,
                "prompt": rewrite_markdown_images(sp.text, rec),
            }
            for sp in rec.subparts_en
        ]

    return {
        "id": rec.id,
        "title": title_source,
        "level": rec.level or "OSK",
        "year": rec.year or 0,
        "topic": map_topic(rec.topic),
        "body": body,
        "parts": [
            {
                "label": sp.label,
                "prompt": rewrite_markdown_images(sp.text, rec),
            }
            for sp in subparts
        ],
        "figure": asset_url(first_image.path) if first_image else None,
        "quality": "repaired" if rec.llm_repaired else "clean",
        "needsReview": False,
        "topicConfidence": rec.topic_confidence,
        "titleEn": rec.title_en.strip() if rec.title_en and rec.title_en.strip() else None,
        "bodyEn": rewrite_markdown_images(rec.body_md_en, rec)
        if rec.body_md_en and rec.body_md_en.strip()
        else None,
        "partsEn": parts_en,
        "hasTranslation": bool(
            rec.llm_translated and rec.body_md_en and rec.body_md_en.strip()
        ),
        "contentLocale": content,
        "locale": locale,
        "hasEnglish": has_english,
        "usingFallback": locale == "en" and not has_english and not is_en_native,
    }


def load_catalog_records(catalog_path: Path) -> list[ProblemRecord]:
    if not catalog_path.is_file():
        return []
    records = load_jsonl(catalog_path, lenient=True)
    seen: set[str] = set()
    unique: list[ProblemRecord] = []
    for rec in records:
        if rec.id in seen:
            continue
        seen.add(rec.id)
        unique.append(rec)
    return unique


def build_catalog_payload(catalog_path: Path, locale: str = "id") -> dict:
    records = load_catalog_records(catalog_path)
    filtered = [rec for rec in records if is_in_locale_catalog(rec, locale)]
    problems = [map_problem_record(rec, locale) for rec in filtered]
    return {"total": len(problems), "problems": problems}


def read_catalog_manifest(meta_path: Path) -> dict | None:
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
