"""Document-level LLM fuse of Marker markdown + pdftotext (problems)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.bronze.doc_prep import prep_marker_markdown, prep_pdftotext
from src.bronze.pdf_text import extract_pdf_text, has_usable_text_layer
from src.llm.llm_client import (
    DEFAULT_OPENROUTER_MODEL,
    ChatCompletionFailure,
    chat_completion_json,
)
from src.text.attach_images import extract_image_refs

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def doc_fuse_model() -> str:
    return os.environ.get("PHYSICS_DOC_FUSE_MODEL", DEFAULT_OPENROUTER_MODEL).strip()


def _provider_ready() -> bool:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in {"local", "ollama"}:
        return True
    if provider == "netra":
        return bool(os.environ.get("NETRA_API_KEY", "").strip())
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def fuse_cache_key(marker_md: str, pdf_text: str) -> str:
    payload = {"marker": marker_md, "pdf": pdf_text, "model": doc_fuse_model()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def build_fuse_messages(*, marker_md: str, pdf_text: str, slug: str) -> list[dict[str, str]]:
    system = (
        "You merge two full-document extractions of the same physics olympiad exam. "
        "Parser A (Marker) preserved structure and diagram references ![](...) but may "
        "have mangled text. Parser B (pdftotext) has clean accurate text but no diagrams. "
        "Produce ONE fused markdown document for the whole exam: use B's clean wording, "
        "keep A's problem numbering/headings and diagram references as ![](...), "
        "normalize math to $...$ LaTeX, omit promotional/WhatsApp/footer junk. "
        "Return strict JSON only: {\"fused_md\": \"...\"}."
    )
    user = json.dumps(
        {
            "document": slug,
            "parser_a_marker": marker_md,
            "parser_b_pdftotext": pdf_text,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _accept_fused(fused_md: str, marker_md: str) -> bool:
    if not fused_md.strip():
        return False
    # Prefer keeping at least one Marker image ref when Marker had any.
    marker_refs = extract_image_refs(marker_md)
    if not marker_refs:
        return True
    fused_refs = set(extract_image_refs(fused_md))
    return any(ref in fused_refs for ref in marker_refs)


def llm_fuse_document(
    marker_md: str,
    pdf_text: str,
    *,
    slug: str,
    model: str | None = None,
) -> str | None:
    if not _provider_ready():
        return None
    try:
        completion = chat_completion_json(
            messages=build_fuse_messages(marker_md=marker_md, pdf_text=pdf_text, slug=slug),
            model=model or doc_fuse_model(),
            temperature=0,
            timeout_s=float(os.environ.get("PHYSICS_DOC_FUSE_TIMEOUT_S", "180")),
            max_tokens=int(os.environ.get("PHYSICS_DOC_FUSE_MAX_TOKENS", "12000")),
            max_retries=int(os.environ.get("PHYSICS_DOC_FUSE_MAX_RETRIES", "2")),
        )
    except Exception:
        return None
    if isinstance(completion, ChatCompletionFailure) or completion.truncated:
        return None
    try:
        data = _extract_json_object(completion.content)
        fused = data.get("fused_md")
        if not isinstance(fused, str):
            return None
        fused = fused.strip()
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not _accept_fused(fused, marker_md):
        return None
    return fused + ("\n" if not fused.endswith("\n") else "")


def fuse_document(
    pdf_path: Path,
    marker_md: str,
    *,
    slug: str,
    bronze_folder: Path | None = None,
    cache_root: Path | None = None,
) -> str | None:
    """Prep + LLM fuse. Returns fused markdown or None."""
    if not has_usable_text_layer(pdf_path):
        return None
    prepared_marker = prep_marker_markdown(marker_md, bronze_folder)
    prepared_pdf = prep_pdftotext(extract_pdf_text(pdf_path))
    if not prepared_pdf.strip():
        return None

    cache_dir = None
    if cache_root is not None:
        cache_dir = cache_root / "doc_fuse"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = fuse_cache_key(prepared_marker, prepared_pdf)
        cache_path = cache_dir / f"{slug}_{key}.json"
        if cache_path.is_file():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                body = data.get("fused_md")
                if isinstance(body, str) and body.strip():
                    return body if body.endswith("\n") else body + "\n"
            except json.JSONDecodeError:
                pass

    fused = llm_fuse_document(prepared_marker, prepared_pdf, slug=slug)
    if fused is None:
        return None
    if cache_dir is not None:
        key = fuse_cache_key(prepared_marker, prepared_pdf)
        (cache_dir / f"{slug}_{key}.json").write_text(
            json.dumps({"fused_md": fused}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return fused
