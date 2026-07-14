from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    slug: str
    source_pdf: str
    level: str | None = None
    year: int | None = None
    round: str | None = None
    variant: int | None = None
    title: str | None = None
    meta_source: str = "unknown"


class MetadataOverrides(BaseModel):
    level: str | None = None
    year: int | None = None
    round: str | None = None
    variant: int | None = None
    title: str | None = None


class SubPart(BaseModel):
    label: str
    text: str


class ProblemImage(BaseModel):
    filename: str
    path: str
    page: int | None = None
    kind: str | None = None


class ProblemSource(BaseModel):
    pdf: str
    md: str
    meta_json: str


class ValidationIssue(BaseModel):
    code: str
    message: str
    snippet: str | None = None


class ProblemRecord(BaseModel):
    id: str
    document_slug: str
    level: str | None = None
    year: int | None = None
    round: str | None = None
    variant: int | None = None
    problem_number: int
    title: str
    topic: str
    topic_confidence: float
    topic_scores: dict[str, float] = Field(default_factory=dict)
    subparts: list[SubPart] = Field(default_factory=list)
    body_md: str
    body_md_raw: str | None = None
    images: list[ProblemImage] = Field(default_factory=list)
    source: ProblemSource
    flags: list[str] = Field(default_factory=list)
    errors: list[ValidationIssue] = Field(default_factory=list)
    llm_repaired: bool = False
    llm_model: str | None = None
    title_en: str | None = None
    body_md_en: str | None = None
    subparts_en: list[SubPart] = Field(default_factory=list)
    llm_translated: bool = False
    llm_translate_model: str | None = None
    # Native language of this problem's primary body (body_md). Indonesian PDFs use
    # "id"; English-only PDFs use "en". Translated EN text lives in body_md_en.
    content_locale: str = "id"


class ParseError(BaseModel):
    folder: str
    error: str


class Manifest(BaseModel):
    documents_processed: int = 0
    problems_extracted: int = 0
    skipped_folders: list[str] = Field(default_factory=list)
    parse_errors: list[ParseError] = Field(default_factory=list)
    low_confidence_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)
