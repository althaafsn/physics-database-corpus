"""Extract solved_concepts (the actual technique used, per the taxonomy detail
ids in data/physics-tags-taxonomy.json) from worked-solution text.

This is strictly more accurate than tagging from the problem statement alone
(src/halliday/classify.py): two problems in the same topic can require very
different solution techniques, and the solution text reveals which one was
actually used. Used to build the prerequisite graph (src/graph/build_prerequisites.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.halliday.classify import classify_heuristic
from src.halliday.taxonomy import PhysicsTagTaxonomy, load_taxonomy
from src.schema import ProblemRecord


def _default_model() -> str:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in {"local", "ollama"} or os.environ.get("LOCAL_LLM_BASE_URL", "").strip():
        return os.environ.get("HALLIDAY_TAG_MODEL", "qwen2.5:3b")
    return os.environ.get("HALLIDAY_TAG_MODEL", "qwen3.6-35b")


@dataclass
class SolutionConcepts:
    problem_id: str
    solved_concepts: list[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = "heuristic_fallback"  # "llm_solution" | "heuristic_fallback"
    model: str | None = None

    def as_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "solved_concepts": self.solved_concepts,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "model": self.model,
        }


def _taxonomy_detail_listing(taxonomy: PhysicsTagTaxonomy) -> str:
    lines: list[str] = []
    for topic in taxonomy.topics:
        detail_ids = ", ".join(d.id for d in topic.details)
        lines.append(f"{topic.id}: {detail_ids}")
    return "\n".join(lines)


def _build_prompt(rec: ProblemRecord, solution_body_md: str, taxonomy: PhysicsTagTaxonomy) -> list[dict[str, str]]:
    system = (
        "You read a WORKED SOLUTION (not the problem statement) to an Indonesian "
        "physics olympiad problem and identify which specific technique(s) from a "
        "fixed taxonomy were actually applied to solve it.\n"
        "Return strict JSON only:\n"
        '{"solved_concepts": ["newtons-laws", "friction"], "confidence": 0.85}\n'
        "Rules:\n"
        "- solved_concepts: 1-4 kebab-case detail ids from the taxonomy below that "
        "the SOLUTION actually uses (e.g. a specific formula, principle, or method "
        "applied in the derivation) - not just topics the problem statement mentions.\n"
        "- Only choose ids that literally exist in the taxonomy list below.\n"
        "- confidence: 0-1, your confidence that these are the concepts genuinely "
        "required to follow this solution."
    )
    user = {
        "problem_id": rec.id,
        "problem_title": rec.title,
        "problem_topic": rec.topic,
        "solution_body_md": solution_body_md[:4000],
        "taxonomy": _taxonomy_detail_listing(taxonomy),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def extract_solution_concepts(
    rec: ProblemRecord,
    solution_body_md: str,
    *,
    model: str | None = None,
) -> SolutionConcepts:
    from src.llm.llm_client import ChatCompletionFailure, chat_completion_json

    if model is None:
        model = _default_model()

    taxonomy = load_taxonomy()
    valid_details = taxonomy.valid_detail_ids()

    completion = chat_completion_json(
        messages=_build_prompt(rec, solution_body_md, taxonomy),
        model=model,
        max_tokens=300,
        timeout_s=90.0,
    )

    fallback_tags = classify_heuristic(rec)
    if isinstance(completion, ChatCompletionFailure):
        return SolutionConcepts(
            problem_id=rec.id,
            solved_concepts=fallback_tags.details,
            confidence=fallback_tags.confidence * 0.5,
            method="heuristic_fallback",
        )

    try:
        data = json.loads(completion.content)
    except json.JSONDecodeError:
        return SolutionConcepts(
            problem_id=rec.id,
            solved_concepts=fallback_tags.details,
            confidence=fallback_tags.confidence * 0.5,
            method="heuristic_fallback",
        )

    concepts = [c for c in data.get("solved_concepts", []) if c in valid_details]
    if not concepts:
        return SolutionConcepts(
            problem_id=rec.id,
            solved_concepts=fallback_tags.details,
            confidence=fallback_tags.confidence * 0.5,
            method="heuristic_fallback",
        )

    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.7))))
    return SolutionConcepts(
        problem_id=rec.id,
        solved_concepts=concepts[:4],
        confidence=confidence,
        method="llm_solution",
        model=model,
    )
