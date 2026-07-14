from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from src.catalog import is_catalog_eligible
from src.paths import PipelinePaths
from src.pdf.pdf_export import export_pdf
from src.record_store import load_jsonl
from src.schema import ProblemRecord, SubPart

IMAGE_REF_RE = re.compile(r"!\[\]\(([^)]+)\)")

VALID_TOPICS = frozenset(
    {"mechanics", "electromagnetism", "thermodynamics", "waves_optics", "modern_physics", "mixed"}
)


class ProblemSetFilters(BaseModel):
    clean_only: bool = True
    levels: list[str] = Field(default_factory=list)
    years: list[int] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    document_slugs: list[str] = Field(default_factory=list)


class ProblemSetManifest(BaseModel):
    name: str
    created_at: str
    source: str
    filters: ProblemSetFilters
    selection: dict[str, object] = Field(default_factory=dict)
    problem_ids: list[str] = Field(default_factory=list)
    problems: list[ProblemRecord] = Field(default_factory=list)


@dataclass
class ProblemSetBuildResult:
    name: str
    output_dir: Path
    problems: list[ProblemRecord]
    manifest_path: Path | None = None
    markdown_path: Path | None = None
    pdf_path: Path | None = None
    pdf_error: str | None = None
    copied_assets: int = 0


@dataclass
class CorpusStats:
    total: int = 0
    clean: int = 0
    levels: dict[str, int] = field(default_factory=dict)
    topics: dict[str, int] = field(default_factory=dict)
    years: dict[int, int] = field(default_factory=dict)


def load_corpus(paths: PipelinePaths, *, use_gold: bool = True, use_catalog: bool = True) -> list[ProblemRecord]:
    if use_catalog and paths.catalog_problems_path.is_file():
        return load_jsonl(paths.catalog_problems_path, lenient=True)
    path = paths.gold_problems_path if use_gold else paths.silver_problems_path
    if not path.is_file():
        fallback = paths.silver_problems_path if use_gold else paths.gold_problems_path
        path = fallback if fallback.is_file() else path
    return load_jsonl(path, lenient=True)


def corpus_stats(records: list[ProblemRecord], *, clean_only: bool = False) -> CorpusStats:
    pool = [r for r in records if not r.errors] if clean_only else records
    stats = CorpusStats(total=len(records), clean=sum(1 for r in records if not r.errors))
    for rec in pool:
        if rec.level:
            stats.levels[rec.level] = stats.levels.get(rec.level, 0) + 1
        if rec.year is not None:
            stats.years[rec.year] = stats.years.get(rec.year, 0) + 1
        stats.topics[rec.topic] = stats.topics.get(rec.topic, 0) + 1
    return stats


def filter_problems(
    records: list[ProblemRecord],
    filters: ProblemSetFilters,
) -> list[ProblemRecord]:
    result: list[ProblemRecord] = []
    levels = {level.upper() for level in filters.levels}
    topics = {topic.lower() for topic in filters.topics}
    slugs = set(filters.document_slugs)

    for rec in records:
        if filters.clean_only and not is_catalog_eligible(rec):
            continue
        if levels and (rec.level or "").upper() not in levels:
            continue
        if filters.years and rec.year not in filters.years:
            continue
        if topics and rec.topic.lower() not in topics:
            continue
        if slugs and rec.document_slug not in slugs:
            continue
        result.append(rec)

    result.sort(key=lambda r: (r.level or "", r.year or 0, r.document_slug, r.problem_number, r.id))
    return result


def select_problems(
    candidates: list[ProblemRecord],
    *,
    ids: list[str] | None = None,
    count: int | None = None,
    seed: int | None = None,
) -> list[ProblemRecord]:
    if ids:
        by_id = {rec.id: rec for rec in candidates}
        missing = [problem_id for problem_id in ids if problem_id not in by_id]
        if missing:
            raise ValueError(f"Problem id(s) not found or filtered out: {', '.join(missing)}")
        return [by_id[problem_id] for problem_id in ids]

    if count is not None:
        if count <= 0:
            raise ValueError("--count must be positive")
        if count > len(candidates):
            raise ValueError(
                f"--count {count} exceeds matching pool size ({len(candidates)}). "
                "Relax filters or lower the count."
            )
        rng = random.Random(seed)
        return sorted(rng.sample(candidates, count), key=lambda r: (r.level or "", r.year or 0, r.id))

    return list(candidates)


def _rewrite_image_ref(ref: str, mapping: dict[str, str]) -> str:
    normalized = ref.replace("\\", "/")
    if normalized in mapping:
        return mapping[normalized]
    basename = Path(normalized).name
    for src, dest in mapping.items():
        if Path(src).name == basename:
            return dest
    return ref


def _rewrite_markdown_images(text: str, mapping: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        ref = match.group(1)
        return f"![]({_rewrite_image_ref(ref, mapping)})"

    return IMAGE_REF_RE.sub(repl, text)


def materialize_assets(
    problems: list[ProblemRecord],
    *,
    parsed_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    """Copy referenced images into output_dir/assets; return ref rewrite map."""
    mapping: dict[str, str] = {}
    copied = 0

    for rec in problems:
        for img in rec.images:
            src = Path(img.path)
            if not src.is_absolute():
                src = parsed_dir / src
            if not src.is_file():
                continue
            rel_dest = Path("assets") / src.relative_to(parsed_dir / "assets")
            dest = output_dir / rel_dest
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
                shutil.copy2(src, dest)
                copied += 1
            mapping[img.path.replace("\\", "/")] = rel_dest.as_posix()
            mapping[Path(img.path).name] = rel_dest.as_posix()
            mapping[src.name] = rel_dest.as_posix()

    return mapping


def _subparts_in_body(body: str, subparts: list[SubPart]) -> bool:
    if not subparts:
        return False
    sample = subparts[0].text.strip()
    if len(sample) < 40:
        return sample in body
    return sample[:80] in body


def _localized_content(rec: ProblemRecord, locale: str) -> tuple[str, str, list[SubPart]]:
    if locale == "en" and rec.body_md_en and rec.body_md_en.strip():
        title = rec.title_en.strip() if rec.title_en and rec.title_en.strip() else rec.title
        return title, rec.body_md_en, rec.subparts_en or rec.subparts
    return rec.title, rec.body_md, rec.subparts


def render_markdown(
    problems: list[ProblemRecord],
    *,
    title: str,
    image_mapping: dict[str, str] | None = None,
    locale: str = "id",
) -> str:
    image_mapping = image_mapping or {}
    lines = [f"# {title}", ""]
    if not problems:
        lines.append("_No problems selected._")
        return "\n".join(lines)

    for index, rec in enumerate(problems, start=1):
        rec_title, rec_body, rec_subparts = _localized_content(rec, locale)
        meta_bits = [bit for bit in (rec.level, str(rec.year) if rec.year else None, rec.topic) if bit]
        meta = " · ".join(meta_bits)
        lines.extend([f"## {index}. {rec_title}", ""])
        if meta:
            lines.extend([f"_{meta} · `{rec.id}`_", ""])
        body = _rewrite_markdown_images(rec_body, image_mapping)
        lines.append(body)
        if rec_subparts and not _subparts_in_body(body, rec_subparts):
            for sp in rec_subparts:
                part_text = _rewrite_markdown_images(sp.text, image_mapping)
                lines.extend(["", f"({sp.label}) {part_text}"])
        lines.append("")
        lines.append("---")
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    if lines and lines[-1] == "---":
        lines.pop()
    return "\n".join(lines)


def export_json(manifest: ProblemSetManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_problem_set(
    paths: PipelinePaths,
    *,
    name: str,
    filters: ProblemSetFilters,
    ids: list[str] | None = None,
    count: int | None = None,
    seed: int | None = None,
    use_gold: bool = True,
    title: str | None = None,
    output_dir: Path | None = None,
    write_markdown: bool = True,
    write_json: bool = True,
    write_pdf: bool = False,
    locale: str = "id",
) -> ProblemSetBuildResult:
    corpus = load_corpus(paths, use_gold=use_gold)
    if not corpus:
        raise FileNotFoundError("No corpus found. Run ingest + LLM repair first.")

    candidates = filter_problems(corpus, filters)
    selected = select_problems(candidates, ids=ids, count=count, seed=seed)

    dest = output_dir or (paths.parsed_dir / "sets" / name)
    dest.mkdir(parents=True, exist_ok=True)

    image_mapping = materialize_assets(selected, parsed_dir=paths.parsed_dir, output_dir=dest)
    copied = len({v for v in image_mapping.values()})

    display_title = title or name.replace("-", " ").replace("_", " ").title()
    manifest = ProblemSetManifest(
        name=name,
        created_at=datetime.now(UTC).isoformat(),
        source="gold" if use_gold else "silver",
        filters=filters,
        selection={"ids": ids, "count": count, "seed": seed},
        problem_ids=[rec.id for rec in selected],
        problems=selected,
    )

    result = ProblemSetBuildResult(
        name=name,
        output_dir=dest,
        problems=selected,
        copied_assets=copied,
    )

    if write_json:
        result.manifest_path = dest / "problem_set.json"
        export_json(manifest, result.manifest_path)

    if write_markdown:
        md_text = render_markdown(
            selected,
            title=display_title,
            image_mapping=image_mapping,
            locale=locale,
        )
        result.markdown_path = dest / "exam.md"
        result.markdown_path.write_text(md_text, encoding="utf-8")

    if write_pdf and result.markdown_path is not None:
        pdf_path = dest / "exam.pdf"
        ok, detail = export_pdf(result.markdown_path, pdf_path, resource_dir=dest)
        if ok:
            result.pdf_path = pdf_path
        else:
            result.pdf_error = _summarize_pdf_error(detail)

    return result


def _summarize_pdf_error(detail: str) -> str:
    detail = detail.strip()
    if not detail:
        return "LaTeX PDF export failed"
    latex_error = re.search(r"^! (.+)$", detail, re.MULTILINE)
    if latex_error:
        return f"LaTeX error: {latex_error.group(1)}"
    if "No LaTeX engine found" in detail:
        return detail
    return detail[-500:] if len(detail) > 500 else detail
