"""Derive a directed prerequisite graph from solved_concepts sets.

For each pair of problems in the same coarse topic, A is a prerequisite of B
when concepts(A) is a (near-)subset of concepts(B) and they share at least
one concept, and B requires strictly more - e.g. a problem needing only
"continuity-equation" is a natural prerequisite for one needing
"continuity-equation + bernoullis-equation + energy-conservation". This
sidesteps needing a subjective difficulty score: subset-of-concepts is itself
the ordering signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.schema import ProblemRecord

# A's concepts don't need to be a *literal* subset of B's - concept
# extraction from an LLM is imperfect, so we tolerate a small amount of
# non-overlap as long as most of A is contained in B.
NEAR_SUBSET_THRESHOLD = 0.75


@dataclass(frozen=True)
class PrerequisiteEdge:
    id: str
    shared_concepts: tuple[str, ...]
    overlap_ratio: float


@dataclass
class ProblemPrerequisites:
    problem_id: str
    prerequisites: list[PrerequisiteEdge] = field(default_factory=list)
    unlocks: list[PrerequisiteEdge] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "prerequisites": [
                {"id": e.id, "shared_concepts": list(e.shared_concepts), "overlap_ratio": round(e.overlap_ratio, 3)}
                for e in self.prerequisites
            ],
            "unlocks": [
                {"id": e.id, "shared_concepts": list(e.shared_concepts), "overlap_ratio": round(e.overlap_ratio, 3)}
                for e in self.unlocks
            ],
        }


def _is_prerequisite(concepts_a: set[str], concepts_b: set[str]) -> tuple[bool, float]:
    """True when A looks like a prerequisite of B (A simpler, mostly
    contained in B's required concepts)."""
    if not concepts_a or not concepts_b:
        return False, 0.0
    if concepts_a == concepts_b:
        return False, 0.0
    shared = concepts_a & concepts_b
    if not shared:
        return False, 0.0
    overlap_ratio = len(shared) / len(concepts_a)
    if overlap_ratio < NEAR_SUBSET_THRESHOLD:
        return False, overlap_ratio
    if len(concepts_b) <= len(concepts_a):
        return False, overlap_ratio  # B isn't more complex than A
    return True, overlap_ratio


def build_prerequisite_graph(
    records: list[ProblemRecord],
    concepts_by_id: dict[str, list[str]],
    *,
    top_k: int = 5,
) -> dict[str, ProblemPrerequisites]:
    by_topic: dict[str, list[ProblemRecord]] = {}
    for rec in records:
        if rec.id not in concepts_by_id:
            continue
        by_topic.setdefault(rec.topic, []).append(rec)

    raw_prereqs: dict[str, list[PrerequisiteEdge]] = {rid: [] for rid in concepts_by_id}
    raw_unlocks: dict[str, list[PrerequisiteEdge]] = {rid: [] for rid in concepts_by_id}

    for topic_records in by_topic.values():
        for rec_b in topic_records:
            concepts_b = set(concepts_by_id.get(rec_b.id, []))
            for rec_a in topic_records:
                if rec_a.id == rec_b.id:
                    continue
                concepts_a = set(concepts_by_id.get(rec_a.id, []))
                is_prereq, ratio = _is_prerequisite(concepts_a, concepts_b)
                if not is_prereq:
                    continue
                shared = tuple(sorted(concepts_a & concepts_b))
                raw_prereqs[rec_b.id].append(PrerequisiteEdge(id=rec_a.id, shared_concepts=shared, overlap_ratio=ratio))
                raw_unlocks[rec_a.id].append(PrerequisiteEdge(id=rec_b.id, shared_concepts=shared, overlap_ratio=ratio))

    result: dict[str, ProblemPrerequisites] = {}
    for problem_id in concepts_by_id:
        prereqs = sorted(raw_prereqs.get(problem_id, []), key=lambda e: (-e.overlap_ratio, e.id))[:top_k]
        unlocks = sorted(raw_unlocks.get(problem_id, []), key=lambda e: (-e.overlap_ratio, e.id))[:top_k]
        result[problem_id] = ProblemPrerequisites(problem_id=problem_id, prerequisites=prereqs, unlocks=unlocks)
    return result
