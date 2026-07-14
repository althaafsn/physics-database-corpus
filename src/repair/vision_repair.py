"""Pick physics diagrams with a local vision LLM when heuristics fail."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass
from pathlib import Path

from src.text.attach_images import attach_images, extract_image_refs
from src.llm.llm_client import get_client
from src.repair.repair_images import is_definite_watermark, strip_watermark_refs
from src.repair.repair_images import pick_best_diagram, pages_from_refs
from src.schema import ProblemRecord
from src.validate import apply_validation

DEFAULT_VISION_MODEL = "moondream"


def vision_model() -> str:
    return os.environ.get("VISION_MODEL", DEFAULT_VISION_MODEL).strip() or DEFAULT_VISION_MODEL


def _image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def list_diagram_candidates(output_folder: Path, *, max_images: int = 8) -> list[Path]:
    """Non-watermark marker images, preferring Figures and larger files."""
    if not output_folder.is_dir():
        return []

    scored: list[tuple[tuple[int, int, int], Path]] = []
    for path in sorted(output_folder.glob("_page_*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if is_definite_watermark(path.name, size):
            continue
        kind_rank = 2 if "_Figure_" in path.name else 1 if "_Picture_" in path.name else 0
        index_match = re.search(r"_(?:Picture|Figure)_(\d+)\.", path.name)
        index = int(index_match.group(1)) if index_match else 0
        if kind_rank == 1 and index < 7 and size < 8000:
            continue
        scored.append(((kind_rank, index, size), path))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in scored[:max_images]]


@dataclass
class VisionPickResult:
    filename: str | None
    reason: str | None = None


def pick_diagram_vision(
    record: ProblemRecord,
    candidates: list[Path],
    *,
    log=print,
) -> VisionPickResult:
    if not candidates:
        return VisionPickResult(None, "no_candidates")
    if len(candidates) == 1:
        return VisionPickResult(candidates[0].name, "single_candidate")

    excerpt = record.body_md[:900]
    model = vision_model()
    client = get_client(timeout_s=float(os.environ.get("VISION_TIMEOUT_S", "180")))
    yes_hits: list[tuple[int, Path]] = []

    for path in candidates:
        prompt = (
            "You are reviewing images for an Indonesian physics olympiad problem.\n"
            f"Problem: {record.title}\n"
            f"Text: {excerpt}\n\n"
            "Does THIS image show a physics diagram, graph, apparatus, or geometry figure "
            "that a student needs to solve the problem?\n"
            "Ignore logos, publisher banners, social-media footers, and decorative headers.\n"
            "Reply with exactly one word: YES or NO."
        )
        content: list[dict] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}},
        ]
        if log:
            log(f"  → vision check {path.name}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_tokens=16,
            )
            answer = (response.choices[0].message.content or "").strip().upper()
        except Exception as exc:
            if log:
                log(f"    api error: {exc}")
            continue
        if answer.startswith("YES"):
            size = path.stat().st_size
            kind_rank = 2 if "_Figure_" in path.name else 1
            yes_hits.append((kind_rank * 1_000_000 + size, path))

    if not yes_hits:
        return VisionPickResult(None, "no_yes_candidates")

    best = max(yes_hits, key=lambda item: item[0])[1]
    return VisionPickResult(best.name, "vision_yes_no")


def needs_vision_image_repair(record: ProblemRecord) -> bool:
    if any(e.code in {"expected_image_missing", "missing_image"} for e in record.errors):
        return True
    if any(f.startswith("missing_image:") or f == "expected_image_missing" for f in record.flags):
        return True
    refs = extract_image_refs(record.body_md)
    if not refs and record.images:
        return True
    return False


def repair_record_images_vision(
    record: ProblemRecord,
    output_folder: Path,
    assets_dir: Path,
    *,
    log=print,
) -> bool:
    """Attach a diagram using heuristics, then local vision LLM if needed."""
    from src.repair.repair_images import repair_record_images

    if repair_record_images(record, output_folder, assets_dir):
        apply_validation(record)
        return True

    if not needs_vision_image_repair(record):
        return False

    body = strip_watermark_refs(record.body_md)
    pages = pages_from_refs(extract_image_refs(body))
    candidates_paths = list_diagram_candidates(output_folder, max_images=20)
    if pages:
        page_candidates = [
            p for p in candidates_paths if any(f"_page_{page}_" in p.name for page in pages)
        ]
        if page_candidates:
            candidates_paths = page_candidates

    if not candidates_paths:
        best = pick_best_diagram(output_folder, pages or {0, 1, 2, 3})
        if best:
            candidates_paths = [output_folder / best]

    pick = pick_diagram_vision(record, candidates_paths, log=log)
    if not pick.filename:
        if log and pick.reason:
            log(f"  ✗ {record.id}: vision pick failed ({pick.reason})")
        return False

    ref = pick.filename
    new_body = body.strip()
    if ref not in extract_image_refs(new_body):
        new_body = f"{new_body}\n\n![]({ref})" if new_body else f"![]({ref})"

    images, flags = attach_images(
        new_body,
        output_folder,
        assets_dir,
        record.level,
        record.year,
        record.problem_number,
        document_slug=record.document_slug,
    )
    record.body_md = new_body
    record.images = images
    record.flags = [
        f
        for f in record.flags
        if not f.startswith("missing_image:") and f != "expected_image_missing"
    ]
    record.flags.extend(flags)
    apply_validation(record)
    if log:
        log(f"  ✓ {record.id}: attached {ref} ({pick.reason})")
    return True
