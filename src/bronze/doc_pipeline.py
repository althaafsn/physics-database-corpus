"""Document-level bronze pipeline: align → caption → structure."""
from __future__ import annotations

import os
from pathlib import Path

from src.bronze.doc_align import align_document
from src.bronze.doc_prep import list_caption_candidate_images
from src.bronze.doc_structure import structure_document
from src.bronze.figure_captions import (
    caption_images,
    load_slug_captions,
    write_slug_captions,
)
from src.bronze.hybrid_bronze import HybridBronzeInfo, write_hybrid_bronze_metadata
from src.bronze.llm_merge import apply_llm_merge_to_folder
from src.bronze.marker_backup import ensure_marker_backup, load_marker_md
from src.bronze.pdf_text import extract_pdf_text, text_layer_stats


def doc_pipeline_enabled() -> bool:
    return os.environ.get("PHYSICS_DOC_STRUCTURE", "0").lower() in {"1", "true", "yes"}


def apply_doc_pipeline_to_folder(
    pdf_path: Path,
    bronze_folder: Path,
    *,
    cache_root: Path | None = None,
    parsed_dir: Path | None = None,
    caption: bool = True,
) -> HybridBronzeInfo | None:
    """Align Marker+pdftotext, caption figures, structure into problems.

    Falls back to per-problem ``llm_merge`` if alignment is impossible.
    """
    if not doc_pipeline_enabled():
        return apply_llm_merge_to_folder(pdf_path, bronze_folder, cache_root=cache_root)

    slug = bronze_folder.name
    md_path = bronze_folder / f"{slug}.md"
    if not md_path.is_file():
        return None

    ensure_marker_backup(bronze_folder)
    marker_md = load_marker_md(bronze_folder)
    if marker_md is None:
        return apply_llm_merge_to_folder(pdf_path, bronze_folder, cache_root=cache_root)

    marker_md = marker_md if marker_md.endswith("\n") else marker_md + "\n"

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

    aligned = align_document(
        pdf_path,
        marker_md,
        bronze_folder=bronze_folder,
    )
    if aligned is None:
        return apply_llm_merge_to_folder(pdf_path, bronze_folder, cache_root=cache_root)

    captions_map: dict[str, str] = {}
    if caption:
        candidates = list_caption_candidate_images(bronze_folder, aligned)
        if candidates:
            results = caption_images(candidates, cache_root=cache_root)
            write_slug_captions(parsed_dir, slug, results)
            captions_map = {
                c.filename: c.caption for c in results if c.status == "ok" and c.caption
            }
        else:
            captions_map = load_slug_captions(parsed_dir, slug)

    structured = structure_document(
        aligned,
        captions_map,
        slug=slug,
        marker_md=marker_md,
        layout_debug_path=bronze_folder.parent / "debug_data" / slug / "blocks.json",
        cache_root=cache_root,
    )
    pdf_text = extract_pdf_text(pdf_path)
    if structured is None:
        md_path.write_text(aligned, encoding="utf-8")
        info = HybridBronzeInfo(
            text_source="doc_align",
            pdf_problem_coverage=1.0,
            pdf_stats=text_layer_stats(pdf_text),
            problem_count=max(aligned.count("## **"), aligned.count("<<<MARKER")),
        )
        write_hybrid_bronze_metadata(bronze_folder, slug, info)
        return info

    md_path.write_text(structured, encoding="utf-8")
    info = HybridBronzeInfo(
        text_source="doc_structure",
        pdf_problem_coverage=1.0,
        pdf_stats=text_layer_stats(pdf_text),
        problem_count=structured.count("## **"),
    )
    write_hybrid_bronze_metadata(bronze_folder, slug, info)
    return info
