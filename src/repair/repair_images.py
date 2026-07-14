from __future__ import annotations

import re
from pathlib import Path

from src.text.attach_images import (
    IMAGE_NAME_RE,
    attach_images,
    extract_image_refs,
    parse_image_filename,
    sanitize_image_refs_in_body,
)
from src.text.attach_images import FIGURE_HINTS, body_expects_attached_figure
from src.text.clean import clean_text
from src.schema import ProblemRecord

# Publisher logo / footer images extracted by Marker (Dimensi Sains layout).
DEFINITE_WATERMARK_INDICES = frozenset({0, 1})

# Often promo banners when paired with 0+1; real diagrams use higher indices.
LIKELY_WATERMARK_INDICES = frozenset({4, 5, 6})

SMALL_IMAGE_BYTES = 6000


def is_definite_watermark(filename: str, size: int) -> bool:
    match = IMAGE_NAME_RE.match(Path(filename).name)
    if not match:
        return False
    kind = match.group(2)
    index = int(match.group(3))
    if kind == "Figure":
        return False
    if index in DEFINITE_WATERMARK_INDICES:
        return True
    return index in LIKELY_WATERMARK_INDICES and size < SMALL_IMAGE_BYTES


def pages_from_refs(refs: list[str]) -> set[int]:
    pages: set[int] = set()
    for ref in refs:
        page, _ = parse_image_filename(ref)
        if page is not None:
            pages.add(page)
    return pages


def pick_best_diagram(output_folder: Path, pages: set[int]) -> str | None:
    """Pick the most likely physics diagram on the given pages."""
    if not pages or not output_folder.is_dir():
        return None

    candidates: list[tuple[tuple[int, int, int], str]] = []
    for path in sorted(output_folder.glob("_page_*")):
        if not path.is_file():
            continue
        page, kind = parse_image_filename(path.name)
        if page is None or page not in pages:
            continue
        match = IMAGE_NAME_RE.match(path.name)
        if not match:
            continue
        index = int(match.group(3))
        size = path.stat().st_size
        if is_definite_watermark(path.name, size):
            continue
        kind_rank = 1 if kind == "Figure" else 0
        candidates.append(((kind_rank, index, size), path.name))

    if not candidates:
        return None

    strong = [c for c in candidates if c[0][0] == 1 or c[0][1] >= 7]
    pool = strong if strong else candidates
    return max(pool, key=lambda item: item[0])[1]


def strip_watermark_refs(body_md: str) -> str:
    """Remove publisher watermark image lines from markdown."""
    lines: list[str] = []
    for line in body_md.splitlines():
        stripped = line.strip()
        match = re.fullmatch(r"!\[\]\(([^)]+)\)", stripped)
        if not match:
            lines.append(line)
            continue
        ref = match.group(1)
        src_name = Path(ref).name
        page, kind = parse_image_filename(src_name)
        if kind != "Picture":
            lines.append(line)
            continue
        index_match = re.search(r"_Picture_(\d+)\.", src_name)
        if not index_match:
            lines.append(line)
            continue
        index = int(index_match.group(1))
        if index in DEFINITE_WATERMARK_INDICES or index in LIKELY_WATERMARK_INDICES:
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def needs_image_repair(record: ProblemRecord) -> bool:
    refs = extract_image_refs(record.body_md)
    if not refs:
        return "expected_image_missing" in record.flags or body_expects_attached_figure(record.body_md)
    if not record.images:
        return True
    return all(re.search(r"_Picture_(?:0|1|4|5|6)\.", ref) for ref in refs)


def repair_record_images(
    record: ProblemRecord,
    output_folder: Path,
    assets_dir: Path,
) -> bool:
    """Fix watermark-only refs and re-attach real diagrams. Returns True if changed."""
    if not output_folder.is_dir():
        return False

    body_md = sanitize_image_refs_in_body(record.body_md)
    original_refs = extract_image_refs(body_md)
    body = strip_watermark_refs(clean_text(body_md))
    pages = pages_from_refs(original_refs)
    if not pages and output_folder.is_dir():
        pages = {
            page
            for path in output_folder.glob("_page_*")
            for page in [parse_image_filename(path.name)[0]]
            if page is not None
        }

    refs_after_strip = extract_image_refs(body)
    valid_refs = [
        ref
        for ref in refs_after_strip
        if (output_folder / ref).is_file()
        and not is_definite_watermark(
            ref, (output_folder / ref).stat().st_size
        )
    ]

    if not valid_refs:
        wants_image = any(hint in body.lower() for hint in FIGURE_HINTS) or bool(original_refs)
        if wants_image:
            best = pick_best_diagram(output_folder, pages)
            if best:
                valid_refs = [best]

    new_body = body
    for ref in extract_image_refs(new_body):
        if ref not in valid_refs:
            new_body = new_body.replace(f"![]({ref})", "").strip()

    for ref in valid_refs:
        if ref not in extract_image_refs(new_body):
            new_body = f"{new_body}\n\n![]({ref})" if new_body.strip() else f"![]({ref})"

    images, flags = attach_images(
        new_body,
        output_folder,
        assets_dir,
        record.level,
        record.year,
        record.problem_number,
        document_slug=record.document_slug,
    )

    changed = (
        new_body != record.body_md
        or [img.filename for img in images] != [img.filename for img in record.images]
    )
    if not changed:
        return False

    record.body_md = new_body
    record.images = images
    record.flags = [
        f
        for f in record.flags
        if not f.startswith("missing_image:") and f != "expected_image_missing"
    ]
    record.flags.extend(flags)
    return True
