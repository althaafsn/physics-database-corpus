"""Default bronze post-process: LLM text merge + deterministic Marker images + offline captions.

Text: per-problem LLM fuses Marker + pdftotext (LaTeX / symbols).
Images: always from the matching Marker problem block — never from vision or structure LLM.
Captions: image-only vision, stored separately for the tutor (``tutor_context``).
"""
from __future__ import annotations

import os
from pathlib import Path

from src.bronze.doc_images import rebind_structured_markdown
from src.bronze.doc_prep import list_caption_candidate_images
from src.bronze.figure_captions import caption_images, write_slug_captions
from src.bronze.hybrid_bronze import HybridBronzeInfo, apply_hybrid_bronze_to_folder
from src.bronze.llm_merge import apply_llm_merge_to_folder, merge_llm_markdown
from src.bronze.marker_backup import ensure_marker_backup, load_marker_md


def hybrid_pipeline_enabled() -> bool:
    """When doc-structure is off, use per-problem LLM merge (default)."""
    return not os.environ.get("PHYSICS_DOC_STRUCTURE", "0").lower() in {"1", "true", "yes"}


def caption_bronze_figures(
    bronze_folder: Path,
    *,
    marker_md: str,
    cache_root: Path | None,
    parsed_dir: Path,
    slug: str,
) -> int:
    """Run image-only vision captions; does not modify bronze markdown."""
    if os.environ.get("PHYSICS_SKIP_FIGURE_CAPTIONS", "").lower() in {"1", "true", "yes"}:
        return 0
    candidates = list_caption_candidate_images(bronze_folder, marker_md)
    if not candidates:
        return 0
    results = caption_images(candidates, cache_root=cache_root)
    write_slug_captions(parsed_dir, slug, results)
    return sum(1 for r in results if r.status == "ok" and r.caption.strip())


def apply_hybrid_pipeline_to_folder(
    pdf_path: Path,
    bronze_folder: Path,
    *,
    cache_root: Path | None = None,
    parsed_dir: Path | None = None,
    caption: bool = True,
) -> HybridBronzeInfo | None:
    """LLM text merge per problem, Marker image layout, optional offline captions."""
    slug = bronze_folder.name
    md_path = bronze_folder / f"{slug}.md"
    if not md_path.is_file():
        return None

    ensure_marker_backup(bronze_folder)
    marker_md = load_marker_md(bronze_folder)
    if marker_md is None:
        marker_md = md_path.read_text(encoding="utf-8")
    else:
        from src.bronze.doc_conflicts import resolve_align_conflicts

        marker_md = resolve_align_conflicts(marker_md)

    if cache_root is None:
        from src.paths import PipelinePaths

        try:
            cache_root = PipelinePaths.resolve().llm_cache_dir
        except Exception:
            cache_root = bronze_folder / ".llm_cache"

    if parsed_dir is None:
        from src.paths import PipelinePaths

        try:
            parsed_dir = PipelinePaths.resolve().parsed_dir
        except Exception:
            parsed_dir = bronze_folder.parent.parent / "parsed"

    try:
        merged, info = merge_llm_markdown(
            pdf_path,
            marker_md,
            slug=slug,
            cache_root=cache_root,
        )
    except Exception:
        info = apply_hybrid_bronze_to_folder(pdf_path, bronze_folder)
        merged = md_path.read_text(encoding="utf-8") if info else None

    if info is None:
        return None

    # The LLM is text-only. Rebind figures after it returns so a shifted
    # problem number or hallucinated image filename cannot move a diagram.
    merged = rebind_structured_markdown(
        merged,
        marker_md,
        slug=slug,
        layout_debug_path=bronze_folder.parent / "debug_data" / slug / "blocks.json",
    )

    md_path.write_text(merged, encoding="utf-8")
    from src.bronze.hybrid_bronze import write_hybrid_bronze_metadata

    write_hybrid_bronze_metadata(bronze_folder, slug, info)

    if caption:
        caption_bronze_figures(
            bronze_folder,
            marker_md=marker_md,
            cache_root=cache_root,
            parsed_dir=parsed_dir,
            slug=slug,
        )

    return info
