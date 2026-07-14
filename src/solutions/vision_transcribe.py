"""Transcribe handwritten solution PDFs page-by-page with a vision LLM.

Extends src/vision_repair.py's vision-client pattern (base64 page image +
chat.completions.create with an image_url part) to full-page handwriting
transcription instead of yes/no diagram picking. The local `moondream` model
used elsewhere in this repo is a small model unlikely to read handwriting
reliably - callers should treat output confidence as low unless a stronger
cloud vision model is explicitly configured via SOLUTION_VISION_MODEL.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path

from src.llm.llm_client import get_client

# Phrases a captioning model (rather than a true transcription-capable model)
# tends to use when it can't actually read the page content - e.g. moondream
# describing "a white piece of paper with equations" instead of transcribing
# them. Used to detect an unusable vision model instead of silently storing
# a caption as if it were a real worked solution.
_DESCRIPTION_PHRASES_RE = re.compile(
    r"\bthe image (shows|depicts|contains)\b|\bthis (image|page|photo) (shows|depicts)\b",
    re.IGNORECASE,
)

DEFAULT_VISION_MODEL = os.environ.get("VISION_MODEL", "minicpm-v")

TRANSCRIBE_PROMPT = (
    "This image is a handwritten page of a worked physics olympiad solution, "
    "in Indonesian. Transcribe the handwriting into clean Markdown, writing "
    "every equation in LaTeX ($...$ inline, $$...$$ for display). "
    "Preserve the original problem numbering exactly as written (e.g. a line "
    "starting with '1.' or '1-' marks the start of solution #1). "
    "Do not translate to English. Do not explain or add commentary - output "
    "only the transcribed markdown content of this page, nothing else. "
    "If the page is blank, illegible, or not physics content, output exactly: "
    "[ILLEGIBLE_PAGE]"
)


def solution_vision_model() -> str:
    return os.environ.get("SOLUTION_VISION_MODEL", DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL


def pdf_to_page_images(pdf_path: Path, out_dir: Path, *, dpi: int = 200) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob("page-*.png"))


def _image_to_data_url(path: Path) -> str:
    encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def transcribe_page(image_path: Path, *, model: str | None = None, timeout_s: float = 180.0) -> str:
    model = model or solution_vision_model()
    client = get_client(timeout_s=timeout_s)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                ],
            }
        ],
        temperature=0,
        max_tokens=2048,
    )
    text = (response.choices[0].message.content or "").strip()
    if text == "[ILLEGIBLE_PAGE]":
        return ""
    if _DESCRIPTION_PHRASES_RE.search(text):
        # The model captioned the page instead of transcribing it - treat as
        # a failed page rather than storing a caption as "solution content".
        return ""
    return text


def transcribe_pdf(pdf_path: Path, *, model: str | None = None, log=print) -> tuple[str, bool]:
    """Return (combined_markdown, model_looks_insufficient).

    model_looks_insufficient is True when most pages produced nothing usable
    (API errors, or the model captioned instead of transcribed) - a signal
    the configured SOLUTION_VISION_MODEL cannot actually read this content,
    distinct from a handful of individually illegible pages.
    """
    with tempfile.TemporaryDirectory(prefix="solution_vision_") as tmp:
        pages = pdf_to_page_images(pdf_path, Path(tmp))
        if not pages:
            return "", True

        parts: list[str] = []
        failed_pages = 0
        for i, page in enumerate(pages, start=1):
            if log:
                log(f"  → transcribing page {i}/{len(pages)}: {page.name}")
            try:
                text = transcribe_page(page, model=model)
            except Exception as exc:
                failed_pages += 1
                if log:
                    log(f"    ✗ vision API error on page {i}: {exc}")
                continue
            if text:
                parts.append(text)
            else:
                failed_pages += 1
        insufficient = failed_pages >= max(1, int(0.5 * len(pages)))
        return "\n\n".join(parts), insufficient
