from __future__ import annotations

from pydantic import BaseModel, Field


class SolutionSource(BaseModel):
    pdf: str
    md: str | None = None


class SolutionStep(BaseModel):
    index: int
    body_md: str
    kind: str = "derivation"
    concepts: list[str] = Field(default_factory=list)


class SolutionRecord(BaseModel):
    """A worked solution aligned to a gold ProblemRecord.id.

    Kept in its own private JSONL (parsed/solutions/solutions.jsonl), never
    exported to public/data/* - see src/solutions/__init__.py.
    """

    problem_id: str
    document_slug: str
    level: str | None = None
    year: int | None = None
    solution_number: int
    body_md: str
    method: str = "typed"  # "typed" | "handwriting_vision"
    source: SolutionSource
    alignment_method: str = "exact"  # "exact" | "ambiguous" | "manual"
    alignment_confidence: float = 1.0
    flags: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    llm_model: str | None = None
    steps: list[SolutionStep] = Field(default_factory=list)
    formatting_confidence: float = 1.0
    parse_version: str = "solution-v1"

    @property
    def needs_review(self) -> bool:
        return (
            bool(self.errors)
            or "alignment_review_required" in self.flags
            or self.alignment_confidence < 0.9
            or any(flag.startswith("low_text_overlap:") for flag in self.flags)
            or self.formatting_confidence < 0.9
        )


class SkippedSolutionDoc(BaseModel):
    """A PDF in all_pdf/solutions/ that was not ingested, with the reason why."""

    pdf: str
    reason: str
    detail: str | None = None
