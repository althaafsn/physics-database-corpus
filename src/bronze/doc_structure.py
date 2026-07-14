"""Structure fused exam markdown + figure captions into per-problem blocks."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from src.bronze.doc_images import rebind_problem_images, rebind_structured_markdown
from src.llm.llm_client import (
    ChatCompletionFailure,
    chat_completion_json,
)
from src.text.attach_images import extract_image_refs

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

# Speed-first free text model for structuring (captions stay on best free VL).
DEFAULT_DOC_STRUCTURE_MODEL = "nvidia/nemotron-nano-9b-v2:free"
FALLBACK_DOC_STRUCTURE_MODELS = (
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
)


def doc_structure_model() -> str:
    return os.environ.get("PHYSICS_DOC_STRUCTURE_MODEL", DEFAULT_DOC_STRUCTURE_MODEL).strip()


def doc_structure_model_fallbacks() -> list[str]:
    raw = os.environ.get(
        "PHYSICS_DOC_STRUCTURE_MODEL_FALLBACK",
        ",".join(FALLBACK_DOC_STRUCTURE_MODELS),
    )
    out: list[str] = []
    for part in raw.split(","):
        m = part.strip()
        if m and m not in out:
            out.append(m)
    return out


def _provider_ready() -> bool:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in {"local", "ollama"}:
        return True
    if provider == "netra":
        return bool(os.environ.get("NETRA_API_KEY", "").strip())
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def structure_cache_key(fused_md: str, captions: dict[str, str]) -> str:
    payload = {
        "fused": fused_md,
        "captions": captions,
        "model": doc_structure_model(),
    }
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


def build_structure_messages(
    *,
    fused_md: str,
    captions: dict[str, str],
    slug: str,
) -> list[dict[str, str]]:
    catalog = [
        {"filename": name, "caption": captions[name]}
        for name in sorted(captions)
        if captions[name].strip()
    ]
    allowed = sorted({Path(r).name for r in extract_image_refs(fused_md)} | set(captions))
    system = (
        "You structure a physics olympiad exam into numbered problems. "
        "The document is a deterministic merge of Marker + pdftotext. "
        "Agreed text appears once. Divergences are tagged as:\n"
        "<<<MARKER\\n...\\n===\\n...\\nPDFTEXT>>> — prefer PDFTEXT wording when sensible, "
        "keep Marker ![](...) image refs, and resolve conflicts into clean body text. "
        "You also receive figure captions (filename → description) for context only. "
        "Do NOT assign image_refs — figures are bound later from Marker layout. "
        "Omit promotional content. "
        "Normalize math to $...$ LaTeX. "
        "Return strict JSON only: "
        '{"problems":[{"number":1,"title":"...","body_md":"...","image_refs":["file.jpeg"]}]}.'
    )
    user = json.dumps(
        {
            "document": slug,
            "aligned_md": fused_md,
            "figure_captions": catalog,
            "allowed_image_filenames": allowed,
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def problems_to_markdown(problems: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    current_variant: int | None = None
    for item in problems:
        number = item.get("number")
        variant = item.get("variant")
        title = str(item.get("title") or f"Soal {number}").strip()
        body = str(item.get("body_md") or "").strip()
        refs = item.get("image_refs") or []
        if not isinstance(refs, list):
            refs = []
        existing = set(extract_image_refs(body))
        for ref in refs:
            name = Path(str(ref)).name
            if name and name not in existing:
                body = f"{body}\n\n![]({name})" if body else f"![]({name})"
                existing.add(name)
        if not body:
            continue
        if variant is not None and variant != current_variant:
            blocks.append(f"## **Versi {variant}**")
            current_variant = variant
        blocks.append(f"## **{number}. {title}**\n\n{body}")
    return "\n\n".join(blocks).strip() + ("\n" if blocks else "")


def _sanitize_problems(
    problems: list[Any],
    *,
    allowed_filenames: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in problems:
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("number"))
        except (TypeError, ValueError):
            continue
        title = str(item.get("title") or f"Soal {number}").strip()
        body = str(item.get("body_md") or "").strip()
        refs_raw = item.get("image_refs") or []
        refs: list[str] = []
        if isinstance(refs_raw, list):
            for ref in refs_raw:
                name = Path(str(ref)).name
                if name in allowed_filenames and name not in refs:
                    refs.append(name)
        for ref in extract_image_refs(body):
            name = Path(ref).name
            if name in allowed_filenames and name not in refs:
                refs.append(name)
        cleaned_lines: list[str] = []
        for line in body.splitlines():
            m = re.match(r"^[ \t]*!\[[^\]]*\]\(([^)]+)\)[ \t]*$", line)
            if m and Path(m.group(1)).name not in allowed_filenames:
                continue
            cleaned_lines.append(line)
        body = "\n".join(cleaned_lines).strip()
        out.append(
            {
                "number": number,
                "title": title,
                "body_md": body,
                "image_refs": refs,
                "variant": item.get("variant"),
            }
        )
    out.sort(key=lambda p: (p.get("variant") is None, p.get("variant") or 0, p["number"]))
    return out


def llm_structure_document(
    fused_md: str,
    captions: dict[str, str],
    *,
    slug: str,
    marker_md: str | None = None,
    layout_debug_path: Path | None = None,
    model: str | None = None,
) -> str | None:
    if not _provider_ready():
        return None
    allowed = {Path(r).name for r in extract_image_refs(fused_md)} | set(captions)
    messages = build_structure_messages(fused_md=fused_md, captions=captions, slug=slug)
    if model:
        chain = [model]
    else:
        chain = []
        seen: set[str] = set()
        for mid in (doc_structure_model(), *doc_structure_model_fallbacks()):
            if mid and mid not in seen:
                seen.add(mid)
                chain.append(mid)

    timeout_s = float(os.environ.get("PHYSICS_DOC_STRUCTURE_TIMEOUT_S", "120"))
    max_tokens = int(os.environ.get("PHYSICS_DOC_STRUCTURE_MAX_TOKENS", "12000"))
    max_retries = int(os.environ.get("PHYSICS_DOC_STRUCTURE_MAX_RETRIES", "2"))

    for mid in chain:
        try:
            completion = chat_completion_json(
                messages=messages,
                model=mid,
                temperature=0,
                timeout_s=timeout_s,
                max_tokens=max_tokens,
                max_retries=max_retries,
            )
        except Exception:
            continue
        if isinstance(completion, ChatCompletionFailure) or completion.truncated:
            continue
        try:
            data = _extract_json_object(completion.content)
            problems = data.get("problems")
            if not isinstance(problems, list) or not problems:
                continue
            sanitized = _sanitize_problems(problems, allowed_filenames=allowed)
            if not sanitized:
                continue
            if marker_md:
                sanitized = rebind_problem_images(
                    sanitized,
                    marker_md,
                    slug=slug,
                    layout_debug_path=layout_debug_path,
                )
            return problems_to_markdown(sanitized)
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return None


def structure_document(
    fused_md: str,
    captions: dict[str, str],
    *,
    slug: str,
    marker_md: str | None = None,
    layout_debug_path: Path | None = None,
    cache_root: Path | None = None,
) -> str | None:
    if cache_root is not None:
        cache_dir = cache_root / "doc_structure"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = structure_cache_key(fused_md, captions)
        cache_path = cache_dir / f"{slug}_{key}.json"
        if cache_path.is_file():
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
                body = data.get("structured_md")
                if isinstance(body, str) and body.strip():
                    cached = body if body.endswith("\n") else body + "\n"
                    if marker_md:
                        return rebind_structured_markdown(
                            cached,
                            marker_md,
                            slug=slug,
                            layout_debug_path=layout_debug_path,
                        )
                    return cached
            except json.JSONDecodeError:
                pass

    structured = llm_structure_document(
        fused_md,
        captions,
        slug=slug,
        marker_md=marker_md,
        layout_debug_path=layout_debug_path,
    )
    if structured is None:
        return None
    if cache_root is not None:
        key = structure_cache_key(fused_md, captions)
        (cache_root / "doc_structure" / f"{slug}_{key}.json").write_text(
            json.dumps({"structured_md": structured}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return structured if structured.endswith("\n") else structured + "\n"
