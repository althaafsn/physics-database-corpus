from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    """Portable layout for bronze (Marker) → silver → gold."""

    root: Path
    pdf_dir: Path
    bronze_dir: Path
    parsed_dir: Path

    @property
    def silver_dir(self) -> Path:
        return self.parsed_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.parsed_dir / "gold"

    @property
    def assets_dir(self) -> Path:
        return self.parsed_dir / "assets"

    @property
    def review_dir(self) -> Path:
        return self.parsed_dir / "review"

    @property
    def llm_cache_dir(self) -> Path:
        return self.parsed_dir / "llm_cache"

    @property
    def registry_path(self) -> Path:
        return self.parsed_dir / "ingest_registry.json"

    @property
    def silver_problems_path(self) -> Path:
        return self.silver_dir / "problems.jsonl"

    @property
    def gold_problems_path(self) -> Path:
        return self.gold_dir / "problems.jsonl"

    @property
    def catalog_problems_path(self) -> Path:
        return self.parsed_dir / "catalog" / "problems.jsonl"

    @property
    def catalog_manifest_path(self) -> Path:
        return self.parsed_dir / "catalog" / "manifest.json"

    @property
    def legacy_problems_path(self) -> Path:
        """Backward-compatible alias to silver output."""
        return self.parsed_dir / "problems.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.parsed_dir / "manifest.json"

    @property
    def run_history_path(self) -> Path:
        return self.parsed_dir / "run_history.jsonl"

    def ensure_dirs(self) -> None:
        for path in (
            self.pdf_dir,
            self.bronze_dir,
            self.parsed_dir,
            self.silver_dir,
            self.gold_dir,
            self.assets_dir,
            self.review_dir,
            self.llm_cache_dir,
            self.parsed_dir / "catalog",
        ):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def resolve(cls, root: Path | None = None) -> PipelinePaths:
        root = (root or Path.cwd()).resolve()
        if env_root := os.environ.get("PHYSICS_DB_ROOT"):
            root = Path(env_root).resolve()

        pdf_dir = Path(os.environ.get("PHYSICS_PDF_DIR", root / "all_pdf")).resolve()
        bronze_dir = Path(os.environ.get("PHYSICS_BRONZE_DIR", root / "output")).resolve()
        parsed_dir = Path(os.environ.get("PHYSICS_PARSED_DIR", root / "parsed")).resolve()
        return cls(root=root, pdf_dir=pdf_dir, bronze_dir=bronze_dir, parsed_dir=parsed_dir)

    def bronze_folder(self, slug: str) -> Path:
        return self.bronze_dir / slug

    def pdf_path_for_slug(self, slug: str) -> Path:
        return self.pdf_dir / f"{slug}.pdf"
