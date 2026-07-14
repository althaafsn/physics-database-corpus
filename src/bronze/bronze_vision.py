"""Vision-LLM bronze conversion for scanned problem-set PDFs.

When pdftotext finds no text layer and Marker is unavailable, render each page
with pdftoppm and transcribe via a vision-capable LLM (OpenRouter by default).
"""
from __future__ import annotations

import base64
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from src.bronze.bronze_convert import BronzeConvertResult, write_bronze_from_problems
from src.llm.llm_client import get_client, provider_info
from src.bronze.pdf_text import text_layer_stats
from src.text.segment_problems import segment_exam_text
from src.solutions.vision_transcribe import pdf_to_page_images

# OpenRouter vision models that read printed exam pages reliably.
DEFAULT_BRONZE_VISION_MODEL = "google/gemini-2.5-flash-lite"
FALLBACK_BRONZE_VISION_MODELS = (
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "qwen/qwen2.5-vl-72b-instruct",
)

EXAM_PAGE_PROMPT = (
    "This image is a page from a physics olympiad exam (problems only, not solutions). "
    "Transcribe all problem text into clean Markdown. Use $...$ for inline math and "
    "$$...$$ for display equations. Preserve problem numbering exactly as printed "
    "(e.g. '1.', 'Problem 1', '1)', 'Q1'). Include figure labels when visible. "
    "Skip page headers, footers, and copyright boilerplate. "
    "Output only the transcribed content of this page — no commentary. "
    "If the page is blank or contains no problem text, output exactly: [BLANK_PAGE]"
)

_DESCRIPTION_PHRASES_RE = re.compile(
    r"\bthe image (shows|depicts|contains)\b|\bthis (image|page|photo) (shows|depicts)\b",
    re.IGNORECASE,
)


def bronze_vision_enabled() -> bool:
    if os.environ.get("BRONZE_VISION_DISABLE", "").lower() in {"1", "true", "yes"}:
        return False
    provider = provider_info()["provider"]
    if provider == "openrouter":
        return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    if provider == "local":
        return True
    return bool(os.environ.get("NETRA_API_KEY", "").strip())


def bronze_vision_model() -> str:
    return os.environ.get("BRONZE_VISION_MODEL", DEFAULT_BRONZE_VISION_MODEL).strip()


def bronze_vision_models() -> list[str]:
    primary = bronze_vision_model()
    extras = [
        m.strip()
        for m in os.environ.get("BRONZE_VISION_MODEL_FALLBACK", ",".join(FALLBACK_BRONZE_VISION_MODELS)).split(",")
        if m.strip()
    ]
    seen: set[str] = set()
    out: list[str] = []
    for model in (primary, *extras):
        if model not in seen:
            seen.add(model)
            out.append(model)
    return out


def _image_to_data_url(path: Path) -> str:
    encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def transcribe_exam_page(
    image_path: Path,
    *,
    model: str,
    timeout_s: float = 120.0,
) -> str:
    client = get_client(timeout_s=timeout_s)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": EXAM_PAGE_PROMPT},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                ],
            }
        ],
        temperature=0,
        max_tokens=int(os.environ.get("BRONZE_VISION_MAX_TOKENS", "4096")),
    )
    text = (response.choices[0].message.content or "").strip()
    if text == "[BLANK_PAGE]":
        return ""
    if _DESCRIPTION_PHRASES_RE.search(text):
        return ""
    return text


def convert_pdf_to_bronze_vision(
    pdf_path: Path,
    *,
    bronze_dir: Path,
    log: Callable[[str], None] | None = None,
) -> BronzeConvertResult:
    slug = pdf_path.stem
    if not bronze_vision_enabled():
        return BronzeConvertResult(
            slug=slug,
            pdf_path=pdf_path,
            ok=False,
            detail="vision bronze disabled or no LLM API key",
        )
    if shutil.which("pdftoppm") is None:
        return BronzeConvertResult(
            slug=slug,
            pdf_path=pdf_path,
            ok=False,
            detail="pdftoppm not installed (install poppler-utils)",
        )

    dpi = int(os.environ.get("BRONZE_VISION_DPI", "160"))
    max_pages = int(os.environ.get("BRONZE_VISION_MAX_PAGES", "40"))
    timeout_s = float(os.environ.get("BRONZE_VISION_TIMEOUT_S", "120"))

    with tempfile.TemporaryDirectory(prefix="bronze_vision_") as tmp:
        try:
            pages = pdf_to_page_images(pdf_path, Path(tmp), dpi=dpi)
        except Exception as exc:
            return BronzeConvertResult(
                slug=slug,
                pdf_path=pdf_path,
                ok=False,
                detail=f"pdftoppm failed: {exc}",
            )

        if not pages:
            return BronzeConvertResult(slug=slug, pdf_path=pdf_path, ok=False, detail="no pages rendered")
        if len(pages) > max_pages:
            return BronzeConvertResult(
                slug=slug,
                pdf_path=pdf_path,
                ok=False,
                detail=f"too many pages ({len(pages)} > {max_pages})",
            )

        models = bronze_vision_models()
        parts: list[str] = []
        failed_pages = 0
        last_error = ""

        for index, page in enumerate(pages, start=1):
            if log:
                log(f"  → vision page {index}/{len(pages)}")
            page_text = ""
            for model in models:
                try:
                    page_text = transcribe_exam_page(page, model=model, timeout_s=timeout_s)
                    if page_text:
                        break
                except Exception as exc:
                    last_error = f"{model}: {exc}"
                    if log:
                        log(f"    ✗ {last_error}")
                    continue
            if page_text:
                parts.append(page_text)
            else:
                failed_pages += 1

        if not parts:
            detail = last_error or "no legible pages transcribed"
            return BronzeConvertResult(slug=slug, pdf_path=pdf_path, ok=False, detail=detail)

        combined = "\n\n".join(parts)
        year_match = re.search(r"(20\d{2})", slug)
        year = int(year_match.group(1)) if year_match else None
        segment = segment_exam_text(combined, slug=slug, year=year)
        if not segment.problems:
            return BronzeConvertResult(
                slug=slug,
                pdf_path=pdf_path,
                ok=False,
                detail=f"no problems detected after vision OCR ({segment.strategy})",
            )

        result = write_bronze_from_problems(
            slug,
            pdf_path,
            bronze_dir,
            segment.problems,
            text_source="vision_llm",
            strategy=segment.strategy,
            pdf_stats=text_layer_stats(combined),
        )
        if result.ok and failed_pages:
            result = BronzeConvertResult(
                slug=result.slug,
                pdf_path=result.pdf_path,
                ok=True,
                detail=f"{result.detail}; {failed_pages} blank pages skipped",
            )
        return result
