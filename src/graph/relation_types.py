# src/graph/relation_types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EDGE_TYPES = frozenset({"prerequisite", "similar", "variant", "harder"})
UNDIRECTED_TYPES = frozenset({"similar", "variant"})
EdgeType = Literal["prerequisite", "similar", "variant", "harder"]
OverrideAction = Literal["upsert", "tombstone"]


@dataclass(frozen=True)
class RelationEdge:
    from_id: str
    to_id: str
    type: str
    reason: str
    confidence: float
    source: str  # "llm" | "edited"
    model: str | None = None
    topic: str | None = None

    def key(self) -> tuple[str, str, str]:
        a, b = canonicalize_endpoints(self.from_id, self.to_id, self.type)
        return (a, b, self.type)


@dataclass(frozen=True)
class RelationOverride:
    from_id: str
    to_id: str
    type: str
    action: str
    reason: str | None = None

    def key(self) -> tuple[str, str, str]:
        a, b = canonicalize_endpoints(self.from_id, self.to_id, self.type)
        return (a, b, self.type)


def canonicalize_endpoints(from_id: str, to_id: str, edge_type: str) -> tuple[str, str]:
    if edge_type in UNDIRECTED_TYPES:
        return (from_id, to_id) if from_id <= to_id else (to_id, from_id)
    return (from_id, to_id)


def bucket_for_problem(problem_id: str, edges: list[RelationEdge]) -> dict[str, list[RelationEdge]]:
    out: dict[str, list[RelationEdge]] = {
        "builds_on": [],
        "similar": [],
        "variants": [],
        "harder": [],
        "easier": [],
    }
    for e in edges:
        if e.type == "prerequisite" and e.to_id == problem_id:
            out["builds_on"].append(e)
        elif e.type == "harder" and e.from_id == problem_id:
            out["harder"].append(e)
        elif e.type == "harder" and e.to_id == problem_id:
            out["easier"].append(e)
        elif e.type == "similar" and problem_id in (e.from_id, e.to_id):
            out["similar"].append(e)
        elif e.type == "variant" and problem_id in (e.from_id, e.to_id):
            out["variants"].append(e)
    return out
