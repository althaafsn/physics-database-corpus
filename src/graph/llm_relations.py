"""LLM-inferred prerequisite / related-problem edges within a topic.

Fast batch mode: one LLM call per topic lists plausible "study before" pairs
among problems in that topic. Merged with deterministic concept-subset edges
in parsed/graph/prerequisites.jsonl.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from src.graph.build_prerequisites import PrerequisiteEdge
from src.schema import ProblemRecord


def _default_model() -> str:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider == "openrouter":
        return os.environ.get("RELATIONS_MODEL", "google/gemma-3-27b-it:free")
    if provider in {"local", "ollama"}:
        return os.environ.get("HALLIDAY_TAG_MODEL", "qwen2.5:3b")
    return os.environ.get("RELATIONS_MODEL", "qwen3.6-35b")


@dataclass
class ProblemSummary:
    problem_id: str
    title: str
    level: str | None
    year: int | None
    concepts: list[str] = field(default_factory=list)
    has_solution: bool = False
    body_snippet: str = ""

    def as_dict(self) -> dict:
        return {
            "id": self.problem_id,
            "title": self.title,
            "level": self.level,
            "year": self.year,
            "concepts": self.concepts[:6],
            "has_solution": self.has_solution,
            "snippet": self.body_snippet[:240],
        }


def _build_topic_prompt(topic: str, summaries: list[ProblemSummary]) -> list[dict[str, str]]:
    system = (
        "You analyze Indonesian physics olympiad problems and infer study order links.\n"
        "Return strict JSON only:\n"
        '{"edges": [{"prerequisite_id": "OSK-2012-01", "target_id": "OSK-2013-05", '
        '"reason": "short phrase", "confidence": 0.7}]}\n'
        "Rules:\n"
        "- prerequisite_id should be a simpler/earlier problem a student can practice "
        "before target_id (same physics techniques, lower combined difficulty).\n"
        "- Only use ids from the provided list.\n"
        "- 0-12 edges total for the whole topic batch; prefer high-confidence links.\n"
        "- problems with has_solution=true are more trustworthy anchors.\n"
        "- confidence: 0-1."
    )
    user = {
        "topic": topic,
        "problems": [s.as_dict() for s in summaries[:40]],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def infer_topic_relations(
    topic: str,
    summaries: list[ProblemSummary],
    *,
    model: str | None = None,
) -> list[tuple[str, str, str, float]]:
    """Return [(prerequisite_id, target_id, reason, confidence), ...]."""
    if len(summaries) < 2:
        return []

    from src.llm.llm_client import ChatCompletionFailure, chat_completion_json

    if model is None:
        model = _default_model()

    completion = chat_completion_json(
        messages=_build_topic_prompt(topic, summaries),
        model=model,
        max_tokens=1200,
        timeout_s=120.0,
    )
    if isinstance(completion, ChatCompletionFailure):
        return []

    try:
        data = _parse_json_object(completion.content)
        if not data:
            return []
    except Exception:
        return []

    valid_ids = {s.problem_id for s in summaries}
    edges: list[tuple[str, str, str, float]] = []
    for item in data.get("edges", []):
        pre = item.get("prerequisite_id")
        tgt = item.get("target_id")
        if not pre or not tgt or pre == tgt:
            continue
        if pre not in valid_ids or tgt not in valid_ids:
            continue
        reason = str(item.get("reason", "related techniques")).strip()
        conf = max(0.0, min(1.0, float(item.get("confidence", 0.6))))
        edges.append((pre, tgt, reason, conf))
    return edges


def merge_llm_edges(
    graph_edges: dict[str, list[PrerequisiteEdge]],
    llm_edges: list[tuple[str, str, str, float]],
) -> dict[str, list[PrerequisiteEdge]]:
    """Add LLM edges into prerequisite lists (target <- prerequisite)."""
    for pre, tgt, reason, conf in llm_edges:
        existing = {e.id for e in graph_edges.get(tgt, [])}
        if pre in existing:
            continue
        graph_edges.setdefault(tgt, []).append(
            PrerequisiteEdge(
                id=pre,
                shared_concepts=(f"llm:{reason}",),
                overlap_ratio=conf,
            )
        )
        graph_edges.setdefault(pre, [])
    return graph_edges


def summaries_for_topic(
    topic: str,
    records: list[ProblemRecord],
    concepts_by_id: dict[str, list[str]],
    solution_ids: set[str],
) -> list[ProblemSummary]:
    out: list[ProblemSummary] = []
    for rec in records:
        if rec.topic != topic:
            continue
        out.append(
            ProblemSummary(
                problem_id=rec.id,
                title=rec.title,
                level=rec.level,
                year=rec.year,
                concepts=concepts_by_id.get(rec.id, []),
                has_solution=rec.id in solution_ids,
                body_snippet=rec.body_md.replace("\n", " ")[:300],
            )
        )
    out.sort(key=lambda s: (s.year or 0, s.problem_id))
    return out
