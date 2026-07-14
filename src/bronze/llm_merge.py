"""LLM-primary hybrid bronze: fuse Marker + pdftotext via a text-only LLM.

Deterministic ``merge_hybrid_markdown`` is used only as a fallback when the LLM
call, parse, or light acceptance check fails. Coverage / usable-text gates match
hybrid bronze (skip merge entirely when there is nothing useful to fuse).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.bronze.doc_images import strip_image_refs_from_body
from src.bronze.hybrid_bronze import (
    HybridBronzeInfo,
    _attach_marker_images,
    _infer_year,
    _marker_preamble,
    apply_hybrid_bronze_to_folder,
    hybrid_bronze_enabled,
    write_hybrid_bronze_metadata,
)
from src.bronze.pdf_text import extract_pdf_text, has_usable_text_layer, text_layer_stats
from src.llm.llm_client import (
    DEFAULT_OPENROUTER_MODEL,
    ChatCompletionFailure,
    chat_completion_json,
)
from src.text.attach_images import extract_image_refs
from src.text.split_problems import (
    split_markdown_auto_with_variants,
    split_pdf_text_problem_variants,
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")


def llm_merge_model() -> str:
    return os.environ.get("PHYSICS_LLM_MERGE_MODEL", DEFAULT_OPENROUTER_MODEL).strip()


def merge_cache_key(
    marker_block: str,
    pdf_block: str,
    problem_number: int,
    problem_variant: int | None = None,
) -> str:
    payload = {
        "problem_number": problem_number,
        "problem_variant": problem_variant,
        "marker": marker_block,
        "pdf": pdf_block,
        "model": llm_merge_model(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:16]


def llm_merge_cache_dir(cache_root: Path) -> Path:
    return cache_root / "llm_merge"


def load_cached_merge(
    cache_dir: Path,
    slug: str,
    problem_number: int,
    key: str,
    problem_variant: int | None = None,
) -> str | None:
    variant = f"v{problem_variant}_" if problem_variant is not None else ""
    path = cache_dir / f"{slug}_{variant}{problem_number}_{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    body = data.get("body_md")
    return body if isinstance(body, str) and body.strip() else None


def save_cached_merge(
    cache_dir: Path,
    slug: str,
    problem_number: int,
    key: str,
    body_md: str,
    problem_variant: int | None = None,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    variant = f"v{problem_variant}_" if problem_variant is not None else ""
    path = cache_dir / f"{slug}_{variant}{problem_number}_{key}.json"
    path.write_text(
        json.dumps({"body_md": body_md}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    return data


def _accept_merged_body(body_md: str, marker_body: str) -> bool:
    """Text-only merge; images are bound afterward from Marker."""
    return bool(body_md.strip())


def build_merge_messages(
    *,
    problem_number: int,
    title: str,
    marker_body: str,
    pdf_body: str,
) -> list[dict[str, str]]:
    system = (
        "You merge two text extractions of the same physics olympiad problem. "
        "Parser A (Marker) preserved structure and diagram references ![](...) but may "
        "have mangled text. Parser B (pdftotext) has clean accurate text but no diagrams. "
        "Produce the final problem body: use B's clean text for wording and math. "
        "Do NOT include any ![](...) image references in body_md — figures are attached "
        "separately from Parser A layout. Preserve meaning, normalize math to $...$ LaTeX "
        "where appropriate. Return strict JSON only: {\"body_md\": \"...\"}. Do not include "
        "the problem heading/number in body_md — only the body content."
    )
    user = json.dumps(
        {
            "problem_number": problem_number,
            "title": title,
            "parser_a_marker": marker_body,
            "parser_b_pdftotext": pdf_body,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _provider_ready_for_merge() -> bool:
    """Fail fast when no LLM credentials are configured (avoid long client errors)."""
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in {"local", "ollama"}:
        return True
    if provider == "netra":
        return bool(os.environ.get("NETRA_API_KEY", "").strip())
    # openrouter (explicit or inferred)
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def llm_merge_problem(
    marker_body: str,
    pdf_body: str,
    problem_number: int,
    *,
    title: str = "",
    model: str | None = None,
    timeout_s: float | None = None,
) -> str | None:
    """Call the text LLM to fuse one problem. Returns body_md or None on failure."""
    if not _provider_ready_for_merge():
        return None

    try:
        completion = chat_completion_json(
            messages=build_merge_messages(
                problem_number=problem_number,
                title=title or f"Soal {problem_number}",
                marker_body=marker_body,
                pdf_body=pdf_body,
            ),
            model=model or llm_merge_model(),
            temperature=0,
            timeout_s=timeout_s
            if timeout_s is not None
            else float(os.environ.get("PHYSICS_LLM_MERGE_TIMEOUT_S", "90")),
            max_tokens=int(os.environ.get("PHYSICS_LLM_MERGE_MAX_TOKENS", "4096")),
            max_retries=int(os.environ.get("PHYSICS_LLM_MERGE_MAX_RETRIES", "2")),
        )
    except Exception:
        return None

    if isinstance(completion, ChatCompletionFailure):
        return None
    if completion.truncated:
        return None

    try:
        data = _extract_json_object(completion.content)
        body_md = data.get("body_md")
        if not isinstance(body_md, str):
            return None
        body_md = body_md.strip()
    except (json.JSONDecodeError, ValueError, TypeError):
        return None

    if not _accept_merged_body(body_md, marker_body):
        return None
    return body_md


def _finalize_problem_body(body_md: str, marker_body: str) -> str:
    """Strip LLM image refs; attach only Marker refs for this problem."""
    body = strip_image_refs_from_body(body_md)
    return _attach_marker_images(body, marker_body)


def _matching_pdf_segment(
    marker_variant: int | None,
    number: int,
    pdf_by_key: dict[tuple[int | None, int], tuple[str, str]],
    pdf_by_number: dict[int, list[tuple[int | None, str, str]]],
) -> tuple[int | None, str, str] | None:
    exact = pdf_by_key.get((marker_variant, number))
    if exact is not None:
        return marker_variant, exact[0], exact[1]
    candidates = pdf_by_number.get(number, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def merge_llm_markdown(
    pdf_path: Path,
    marker_md: str,
    *,
    slug: str | None = None,
    cache_root: Path | None = None,
) -> tuple[str, HybridBronzeInfo | None]:
    """LLM-primary hybrid merge; deterministic fallback per problem on failure."""
    slug = slug or pdf_path.stem
    year = _infer_year(slug)

    if not hybrid_bronze_enabled() or not has_usable_text_layer(pdf_path):
        return marker_md, None

    pdf_text = extract_pdf_text(pdf_path)
    pdf_problems = split_pdf_text_problem_variants(pdf_text, year=year, slug=slug)
    marker_problems = split_markdown_auto_with_variants(marker_md, year=year)
    if not pdf_problems:
        return marker_md, None

    marker_by_key = {
        (variant, number): (title, body)
        for number, title, body, variant in marker_problems
    }
    pdf_by_key = {
        (variant, number): (title, body)
        for number, title, body, variant in pdf_problems
    }
    pdf_by_number: dict[int, list[tuple[int | None, str, str]]] = {}
    for number, title, body, variant in pdf_problems:
        pdf_by_number.setdefault(number, []).append((variant, title, body))
    ordered_keys: list[tuple[int | None, int]] = []
    for number, _title, _body, variant in marker_problems:
        key = (variant, number)
        if key not in ordered_keys:
            ordered_keys.append(key)
    for number, _title, _body, variant in pdf_problems:
        key = (variant, number)
        if key not in ordered_keys:
            ordered_keys.append(key)
    if not ordered_keys:
        return marker_md, None

    matched = sum(
        1
        for variant, number in ordered_keys
        if _matching_pdf_segment(variant, number, pdf_by_key, pdf_by_number) is not None
    )
    coverage = matched / len(ordered_keys)
    if coverage < 0.25:
        return marker_md, None

    cache_dir = llm_merge_cache_dir(cache_root) if cache_root is not None else None
    llm_successes = 0
    merged_blocks: list[str] = []
    merged_problem_count = 0

    current_variant: int | None = None
    for variant, number in ordered_keys:
        marker_title, marker_body = marker_by_key.get(
            (variant, number), (f"Soal {number}", "")
        )
        pdf_segment = _matching_pdf_segment(variant, number, pdf_by_key, pdf_by_number)
        if pdf_segment is None:
            title, body = marker_title, marker_body
        else:
            _pdf_variant, pdf_title, pdf_body = pdf_segment
            title = marker_title if marker_title != f"Soal {number}" else pdf_title
            body: str | None = None
            key = merge_cache_key(marker_body, pdf_body, number, variant)
            if cache_dir is not None:
                body = load_cached_merge(cache_dir, slug, number, key, variant)
            if body is None:
                body = llm_merge_problem(
                    marker_body,
                    pdf_body,
                    number,
                    title=title,
                )
                if body is not None and cache_dir is not None:
                    save_cached_merge(cache_dir, slug, number, key, body, variant)
            if body is not None:
                llm_successes += 1
                body = _finalize_problem_body(body, marker_body)
            else:
                title, body = title, _finalize_problem_body(pdf_body, marker_body)

        if not body.strip():
            continue
        if variant is not None and variant != current_variant:
            merged_blocks.append(f"## **Versi {variant}**")
            current_variant = variant
        merged_blocks.append(f"## **{number}. {title}**\n\n{body.strip()}")
        merged_problem_count += 1

    if not merged_blocks:
        return marker_md, None

    preamble = _marker_preamble(marker_md)
    parts = [preamble] if preamble else []
    parts.append("\n\n".join(merged_blocks))
    merged_md = "\n\n".join(part for part in parts if part).strip() + "\n"

    text_source = "llm_merge" if llm_successes > 0 else "hybrid_pdftotext"
    info = HybridBronzeInfo(
        text_source=text_source,
        pdf_problem_coverage=round(coverage, 4),
        pdf_stats=text_layer_stats(pdf_text),
        problem_count=merged_problem_count,
    )
    return merged_md, info


def apply_llm_merge_to_folder(
    pdf_path: Path,
    bronze_folder: Path,
    *,
    cache_root: Path | None = None,
    parsed_dir: Path | None = None,
    caption: bool = True,
) -> HybridBronzeInfo | None:
    """Post-process Marker bronze with LLM-primary hybrid merge."""
    from src.bronze.hybrid_pipeline import apply_hybrid_pipeline_to_folder

    return apply_hybrid_pipeline_to_folder(
        pdf_path,
        bronze_folder,
        cache_root=cache_root,
        parsed_dir=parsed_dir,
        caption=caption,
    )


def _apply_llm_merge_to_folder_legacy(
    pdf_path: Path,
    bronze_folder: Path,
    *,
    cache_root: Path | None = None,
) -> HybridBronzeInfo | None:
    """Legacy entry without captions / marker backup (tests only)."""
    slug = bronze_folder.name
    md_path = bronze_folder / f"{slug}.md"
    if not md_path.is_file():
        return None

    if cache_root is None:
        from src.paths import PipelinePaths

        try:
            cache_root = PipelinePaths.resolve().llm_cache_dir
        except Exception:
            cache_root = bronze_folder / ".llm_cache"

    marker_md = md_path.read_text(encoding="utf-8")
    try:
        merged, info = merge_llm_markdown(
            pdf_path,
            marker_md,
            slug=slug,
            cache_root=cache_root,
        )
    except Exception:
        return apply_hybrid_bronze_to_folder(pdf_path, bronze_folder)

    if info is None:
        return None

    md_path.write_text(merged, encoding="utf-8")
    write_hybrid_bronze_metadata(bronze_folder, slug, info)
    return info
