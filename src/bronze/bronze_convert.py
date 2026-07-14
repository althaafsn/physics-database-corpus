from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from src.ingest.ingest_registry import IngestRegistryStore, IngestStage
from src.bronze.hybrid_bronze import hybrid_bronze_enabled
from src.bronze.llm_merge import apply_llm_merge_to_folder
from src.paths import PipelinePaths


@dataclass
class BronzeConvertResult:
    slug: str
    pdf_path: Path
    ok: bool
    detail: str = ""


def bronze_md_path(paths: PipelinePaths, slug: str) -> Path:
    return paths.bronze_folder(slug) / f"{slug}.md"


def is_bronze_ready(paths: PipelinePaths, slug: str) -> bool:
    return bronze_md_path(paths, slug).is_file()


def list_pending_marker_pdfs(
    paths: PipelinePaths,
    *,
    registry: IngestRegistryStore | None = None,
    only_slugs: Iterable[str] | None = None,
) -> list[tuple[str, Path]]:
    """Return (slug, pdf_path) pairs that still need Marker conversion."""
    allowed = set(only_slugs) if only_slugs is not None else None
    pending: list[tuple[str, Path]] = []

    for pdf in sorted(paths.pdf_dir.glob("*.pdf")):
        slug = pdf.stem
        if allowed is not None and slug not in allowed:
            continue
        if is_bronze_ready(paths, slug):
            continue
        pending.append((slug, pdf))

    if registry is not None:
        # Keep registry order hints stable for status output.
        stage_rank = {
            IngestStage.PDF_ONLY: 0,
            IngestStage.BRONZE_READY: 1,
            IngestStage.SILVER_DONE: 2,
            IngestStage.GOLD_DONE: 3,
        }
        pending.sort(
            key=lambda item: (
                stage_rank.get(registry.get(item[0]).stage if registry.get(item[0]) else IngestStage.PDF_ONLY, 0),
                item[0],
            )
        )

    return pending


def marker_extra_args_from_env() -> list[str]:
    """Build optional Marker CLI flags from environment."""
    import os

    args: list[str] = []
    if os.environ.get("PHYSICS_MARKER_LAYOUT", "1").lower() in {"1", "true", "yes"}:
        # Marker emits blocks.json alongside Markdown, preserving page/bbox
        # ownership that the flattened Markdown path cannot represent.
        args.extend(["--debug", "--debug_json"])
    if os.environ.get("MARKER_HIGH_DPI", "1").lower() in {"1", "true", "yes"}:
        args.extend(["--highres_image_dpi", os.environ.get("MARKER_HIGHRES_DPI", "288")])
    force_ocr_default = "0" if hybrid_bronze_enabled() else "1"
    if os.environ.get("MARKER_FORCE_OCR", force_ocr_default).lower() in {"1", "true", "yes"}:
        args.append("--force_ocr")
    if os.environ.get("MARKER_USE_LLM", "").lower() in {"1", "true", "yes"}:
        args.extend(
            [
                "--use_llm",
                "--llm_service",
                os.environ.get(
                    "MARKER_LLM_SERVICE",
                    "marker.services.ollama.OllamaService",
                ),
            ]
        )
    return args


def marker_single_candidates() -> list[Path]:
    """Known marker_single locations (environment, project, and standard installs)."""
    import os

    candidates: list[Path] = []
    for raw in (os.environ.get("MARKER_SINGLE"), os.environ.get("MARKER_BIN")):
        if raw:
            candidates.append(Path(raw))

    db_root = os.environ.get("PHYSICS_DB_ROOT", "").strip()
    if db_root:
        candidates.append(Path(db_root) / ".venv-marker" / "bin" / "marker_single")

    candidates.extend(
        (
            Path("/opt/physics-database/.venv-marker/bin/marker_single"),
            Path.home() / ".venv-marker" / "bin" / "marker_single",
        )
    )
    return candidates


def marker_single_path() -> Path | None:
    """Return path to marker_single if installed, else None."""
    found = shutil.which("marker_single")
    if found:
        return Path(found)
    for candidate in marker_single_candidates():
        if candidate.is_file():
            return candidate
    return None


def resolve_marker_argv() -> list[str]:
    """Return argv prefix to invoke marker_single."""
    path = marker_single_path()
    if path is not None:
        return [str(path)]
    return [sys.executable, "-m", "marker.scripts.convert_single"]


def marker_available() -> bool:
    """True when marker_single is on disk (remote ingest prerequisite)."""
    return marker_single_path() is not None


def resolve_marker_single() -> list[str]:
    """Backward-compatible alias."""
    return resolve_marker_argv()


def marker_env(*, force_cpu: bool = False) -> dict[str, str]:
    """Environment for Marker subprocess (CPU fallback avoids GPU OOM on small laptops)."""
    import os

    env = os.environ.copy()
    use_cpu = force_cpu or env.get("PHYSICS_MARKER_CPU", "").lower() in {"1", "true", "yes"}
    if use_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def convert_pdf_to_bronze_pdftotext(
    pdf_path: Path,
    *,
    bronze_dir: Path,
) -> BronzeConvertResult:
    """Fast bronze from PDF text layer only (no Marker)."""
    from src.bronze.pdf_text import (
        extract_pdf_text,
        has_extractable_text_layer,
        strip_pdf_footers,
        text_layer_stats,
    )
    from src.text.segment_problems import segment_exam_text

    slug = pdf_path.stem
    if not has_extractable_text_layer(pdf_path):
        return BronzeConvertResult(
            slug=slug,
            pdf_path=pdf_path,
            ok=False,
            detail="no extractable text layer",
        )

    pdf_text = strip_pdf_footers(extract_pdf_text(pdf_path))
    if not pdf_text.strip():
        return BronzeConvertResult(slug=slug, pdf_path=pdf_path, ok=False, detail="empty text")

    import re

    year_match = re.search(r"(20\d{2})", slug)
    year = int(year_match.group(1)) if year_match else None
    segment = segment_exam_text(pdf_text, slug=slug, year=year)
    problems = segment.problems
    if not problems:
        return BronzeConvertResult(
            slug=slug,
            pdf_path=pdf_path,
            ok=False,
            detail=f"no problems detected ({segment.strategy})",
        )

    return write_bronze_from_problems(
        slug,
        pdf_path,
        bronze_dir,
        problems,
        text_source="pdftotext",
        strategy=segment.strategy,
        pdf_stats=text_layer_stats(pdf_text),
    )


def write_bronze_from_problems(
    slug: str,
    pdf_path: Path,
    bronze_dir: Path,
    problems: dict[int, tuple[str, str]],
    *,
    text_source: str,
    strategy: str,
    pdf_stats: dict[str, int] | None = None,
) -> BronzeConvertResult:
    """Write bronze markdown + meta from segmented problems."""
    from src.bronze.hybrid_bronze import HybridBronzeInfo, write_hybrid_bronze_metadata

    blocks: list[str] = []
    for number in sorted(problems):
        title, body = problems[number]
        if not body.strip():
            continue
        blocks.append(f"## **{number}. {title}**\n\n{body.strip()}")

    if not blocks:
        return BronzeConvertResult(slug=slug, pdf_path=pdf_path, ok=False, detail="empty problems")

    folder = bronze_dir / slug
    folder.mkdir(parents=True, exist_ok=True)
    md_path = folder / f"{slug}.md"
    md_path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")

    import json

    meta_path = folder / f"{slug}_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "table_of_contents": [
                    {
                        "title": f"{number}. {problems[number][0]}",
                        "heading_level": None,
                        "page_id": 0,
                    }
                    for number in sorted(problems)
                ]
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    info = HybridBronzeInfo(
        text_source=text_source,
        pdf_problem_coverage=1.0,
        pdf_stats=pdf_stats or {"chars": 0, "words": 0, "lines": 0},
        problem_count=len(blocks),
    )
    write_hybrid_bronze_metadata(folder, slug, info)
    return BronzeConvertResult(
        slug=slug,
        pdf_path=pdf_path,
        ok=True,
        detail=f"{text_source}/{strategy} ({len(blocks)} problems)",
    )


def convert_pdf_to_bronze(
    pdf_path: Path,
    *,
    bronze_dir: Path,
    marker_argv: list[str] | None = None,
    marker_extra_args: list[str] | None = None,
    timeout_s: float | None = None,
    force_cpu: bool = False,
) -> BronzeConvertResult:
    slug = pdf_path.stem
    bronze_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        *(marker_argv or resolve_marker_single()),
        str(pdf_path.resolve()),
        "--output_dir",
        str(bronze_dir.resolve()),
        "--disable_tqdm",
    ]
    extra_args = list(marker_extra_args or [])
    if "--debug_json" in extra_args and "--debug_data_folder" not in extra_args:
        extra_args.extend(["--debug_data_folder", str((bronze_dir / "debug_data").resolve())])
    if extra_args:
        cmd.extend(extra_args)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env=marker_env(force_cpu=force_cpu),
        )
    except subprocess.TimeoutExpired as exc:
        return BronzeConvertResult(
            slug=slug,
            pdf_path=pdf_path,
            ok=False,
            detail=f"timeout after {exc.timeout}s",
        )

    md_path = bronze_dir / slug / f"{slug}.md"
    if proc.returncode == 0 and md_path.is_file():
        from src.bronze.marker_backup import save_marker_backup

        save_marker_backup(bronze_dir / slug)
        merge_detail = ""
        if hybrid_bronze_enabled():
            from src.bronze.doc_pipeline import apply_doc_pipeline_to_folder, doc_pipeline_enabled
            from src.bronze.hybrid_pipeline import apply_hybrid_pipeline_to_folder

            if doc_pipeline_enabled():
                info = apply_doc_pipeline_to_folder(pdf_path, bronze_dir / slug)
            else:
                info = apply_hybrid_pipeline_to_folder(pdf_path, bronze_dir / slug)
            if info is not None:
                merge_detail = f"/{info.text_source}"
        base = "marker (cpu)" if force_cpu else "marker"
        detail = f"{base}{merge_detail}"
        return BronzeConvertResult(slug=slug, pdf_path=pdf_path, ok=True, detail=detail)

    err = (proc.stderr or proc.stdout or "").strip()
    oom = "OutOfMemoryError" in err or "CUDA out of memory" in err
    if oom and not force_cpu:
        retry = convert_pdf_to_bronze(
            pdf_path,
            bronze_dir=bronze_dir,
            marker_argv=marker_argv,
            marker_extra_args=marker_extra_args,
            timeout_s=timeout_s,
            force_cpu=True,
        )
        if retry.ok:
            return retry
        err = retry.detail or err

    if oom:
        err = "CUDA out of memory — re-run with PHYSICS_MARKER_CPU=1"
    elif len(err) > 400:
        err = err[:400] + "..."
    detail = err or f"exit code {proc.returncode}"
    return BronzeConvertResult(slug=slug, pdf_path=pdf_path, ok=False, detail=detail)


def convert_pdf_to_bronze_for_ingest(
    pdf_path: Path,
    *,
    bronze_dir: Path,
    log: Callable[[str], None] | None = None,
) -> BronzeConvertResult:
    """Marker-first bronze for admin uploads (layout, figures, hybrid text merge)."""
    import os

    slug = pdf_path.stem
    always_marker = os.environ.get("PHYSICS_INGEST_ALWAYS_MARKER", "1").lower() in {
        "1",
        "true",
        "yes",
    }

    if not always_marker:
        fast = convert_pdf_to_bronze_pdftotext(pdf_path, bronze_dir=bronze_dir)
        if fast.ok:
            return fast

    marker_timeout = float(os.environ.get("PHYSICS_MARKER_TIMEOUT_S", "3600"))
    if log:
        log("Running Marker (layout + images)...")
    marker = convert_pdf_to_bronze(
        pdf_path,
        bronze_dir=bronze_dir,
        marker_argv=resolve_marker_argv(),
        marker_extra_args=marker_extra_args_from_env(),
        timeout_s=marker_timeout,
    )
    if marker.ok:
        return marker

    if log:
        log("Marker failed; trying vision LLM fallback...")
    from src.bronze.bronze_vision import convert_pdf_to_bronze_vision

    vision = convert_pdf_to_bronze_vision(pdf_path, bronze_dir=bronze_dir, log=log)
    if vision.ok:
        return vision

    fast = convert_pdf_to_bronze_pdftotext(pdf_path, bronze_dir=bronze_dir)
    if fast.ok:
        return fast

    return BronzeConvertResult(
        slug=slug,
        pdf_path=pdf_path,
        ok=False,
        detail=(
            f"marker: {marker.detail}; vision: {vision.detail}; pdftotext: {fast.detail}"
        ),
    )


def convert_pending_pdfs(
    paths: PipelinePaths,
    items: list[tuple[str, Path]],
    *,
    marker_argv: list[str] | None = None,
    marker_extra_args: list[str] | None = None,
    force: bool = False,
    timeout_s: float | None = None,
    log: Callable[[str], None] | None = print,
) -> list[BronzeConvertResult]:
    import shutil

    results: list[BronzeConvertResult] = []
    total = len(items)
    for index, (slug, pdf_path) in enumerate(items, start=1):
        if force:
            target = paths.bronze_folder(slug)
            if target.is_dir():
                shutil.rmtree(target)
        if log:
            log(f"[Marker {index}/{total}] {slug}")
        result = convert_pdf_to_bronze(
            pdf_path,
            bronze_dir=paths.bronze_dir,
            marker_argv=marker_argv,
            marker_extra_args=marker_extra_args,
            timeout_s=timeout_s,
        )
        results.append(result)
        if log:
            status = "ok" if result.ok else f"failed: {result.detail}"
            log(f"  → {status}")
    return results
