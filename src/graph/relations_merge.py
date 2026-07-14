# src/graph/relations_merge.py
from __future__ import annotations

import math

from src.graph.relation_types import RelationEdge, RelationOverride, canonicalize_endpoints
from src.graph.relation_types import EDGE_TYPES


def _valid_edge(from_id: str, to_id: str, edge_type: str) -> bool:
    return bool(
        isinstance(from_id, str)
        and isinstance(to_id, str)
        and from_id.strip()
        and to_id.strip()
        and from_id != to_id
        and edge_type in EDGE_TYPES
    )


def _safe_confidence(value: float) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    return max(0.0, min(1.0, confidence))


def merge_relations(
    auto: list[RelationEdge],
    overrides: list[RelationOverride],
) -> list[RelationEdge]:
    by_key: dict[tuple[str, str, str], RelationEdge] = {}
    for edge in auto:
        if not _valid_edge(edge.from_id, edge.to_id, edge.type):
            continue
        confidence = _safe_confidence(edge.confidence)
        if confidence is None:
            continue
        a, b = canonicalize_endpoints(edge.from_id, edge.to_id, edge.type)
        by_key[edge.key()] = RelationEdge(
            from_id=a,
            to_id=b,
            type=edge.type,
            reason=edge.reason,
            confidence=confidence,
            source=edge.source,
            model=edge.model,
            topic=edge.topic,
        )

    for ov in overrides:
        if ov.action not in {"tombstone", "upsert"} or not _valid_edge(
            ov.from_id, ov.to_id, ov.type
        ):
            continue
        key = ov.key()
        if ov.action == "tombstone":
            by_key.pop(key, None)
            continue
        if ov.action == "upsert":
            a, b = canonicalize_endpoints(ov.from_id, ov.to_id, ov.type)
            by_key[key] = RelationEdge(
                from_id=a,
                to_id=b,
                type=ov.type,
                reason=ov.reason or "",
                confidence=1.0,
                source="edited",
            )
    return [by_key[key] for key in sorted(by_key)]
