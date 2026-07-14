from __future__ import annotations

import re
import shutil
from pathlib import Path

from src.schema import ProblemImage

IMAGE_REF_RE = re.compile(r"!\[\]\(([^)]+)\)")
IMAGE_NAME_RE = re.compile(
    r"_page_(\d+)_(Picture|Figure)_(\d+)\.(jpeg|jpg|png)$",
    re.IGNORECASE,
)

FIGURE_HINTS = ("gambar", "lihat gambar", "lihat pada gambar", "seperti di bawah", "diagram", "grafik")
STUDENT_DRAW_RE = re.compile(
    r"\b(?:gambarkan|gambar\s+(?:diagram|lintasan|grafik)|buat\s+diagram|nyatakan\s+dalam\s+diagram)\b",
    re.IGNORECASE,
)


def body_expects_attached_figure(body_md: str) -> bool:
    lower = body_md.lower()
    if STUDENT_DRAW_RE.search(body_md):
        return False
    return any(hint in lower for hint in FIGURE_HINTS)


def parse_image_filename(filename: str) -> tuple[int | None, str | None]:
    match = IMAGE_NAME_RE.match(Path(filename).name)
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def normalize_image_ref(ref: str) -> str:
    """Strip bogus URL prefixes LLMs sometimes add to Marker image filenames."""
    clean = ref.strip().replace("\\", "/")
    if clean.startswith("https://"):
        clean = clean[8:]
    elif clean.startswith("http://"):
        clean = clean[7:]
    name = Path(clean).name
    return name if name.startswith("_page_") else clean


def sanitize_image_refs_in_body(body_md: str) -> str:
    """Rewrite ![](https://_page_*.jpeg) → ![](_page_*.jpeg) in problem bodies."""

    def repl(match: re.Match[str]) -> str:
        ref = match.group(1)
        normalized = normalize_image_ref(ref)
        return f"![]({normalized})" if normalized != ref else match.group(0)

    return IMAGE_REF_RE.sub(repl, body_md)


def extract_image_refs(body_md: str) -> list[str]:
    return [normalize_image_ref(ref) for ref in IMAGE_REF_RE.findall(body_md)]


def asset_dest_dir(
    assets_dir: Path,
    *,
    level: str | None,
    year: int | None,
    document_slug: str,
    problem_number: int,
) -> Path:
    if level and year is not None:
        return assets_dir / level / str(year) / f"{problem_number:02d}"
    safe_slug = re.sub(r"[^\w\-]+", "_", document_slug)[:80]
    return assets_dir / safe_slug / f"{problem_number:02d}"


def attach_images(
    body_md: str,
    output_folder: Path,
    assets_dir: Path,
    level: str | None,
    year: int | None,
    problem_number: int,
    *,
    document_slug: str,
) -> tuple[list[ProblemImage], list[str]]:
    """Resolve image refs, copy into assets_dir, return records and flags."""
    flags: list[str] = []
    images: list[ProblemImage] = []
    refs = extract_image_refs(body_md)

    dest_dir = asset_dest_dir(
        assets_dir,
        level=level,
        year=year,
        document_slug=document_slug,
        problem_number=problem_number,
    )
    dest_dir.mkdir(parents=True, exist_ok=True)

    for ref in refs:
        filename = normalize_image_ref(ref)
        src = (output_folder / filename).resolve()
        if not src.is_file():
            flags.append(f"missing_image:{ref}")
            continue
        page, kind = parse_image_filename(src.name)
        dest = dest_dir / src.name
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            shutil.copy2(src, dest)
        rel_path = dest.relative_to(assets_dir.parent)
        images.append(
            ProblemImage(
                filename=src.name,
                path=str(rel_path).replace("\\", "/"),
                page=page,
                kind=kind,
            )
        )

    if not images and body_expects_attached_figure(body_md):
        flags.append("expected_image_missing")

    return images, flags
