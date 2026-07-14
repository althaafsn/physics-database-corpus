from __future__ import annotations

import json
import os
from typing import Iterable

from src.graph.relation_types import (
    EDGE_TYPES,
    RelationEdge,
    canonicalize_endpoints,
)
from src.schema import ProblemRecord

OS_LEVELS = frozenset({"OSK", "OSN", "OSP"})


def _default_model() -> str:
    return os.environ.get("RELATIONS_MODEL", "nvidia/nemotron-3-nano-30b-a3b:free")


def iter_batches(items: list, size: int):
    """Yield successive slices of *items* with at most *size* elements."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def filter_os_records(records: Iterable[ProblemRecord]) -> list[ProblemRecord]:
    return [r for r in records if (r.level or "") in OS_LEVELS]


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from model output."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    for marker in ('{"edges"', '{ "edges"'):
        idx = text.find(marker)
        if idx == -1:
            continue
        sub = text[idx:]
        end = sub.rfind("}")
        if end == -1:
            continue
        try:
            data = json.loads(sub[: end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


def parse_typed_edges_payload(
    data: dict,
    *,
    valid_ids: set[str],
    model: str | None = None,
    topic: str | None = None,
) -> list[RelationEdge]:
    edges: list[RelationEdge] = []
    seen: set[tuple[str, str, str]] = set()
    for item in data.get("edges") or []:
        frm = item.get("from_id") or item.get("prerequisite_id")
        to = item.get("to_id") or item.get("target_id")
        etype = str(item.get("type", "")).strip()
        if not frm or not to or frm == to:
            continue
        if frm not in valid_ids or to not in valid_ids:
            continue
        if etype not in EDGE_TYPES:
            continue
        reason = str(item.get("reason", "")).strip()
        try:
            conf = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            conf = 0.6
        conf = max(0.0, min(1.0, conf))
        a, b = canonicalize_endpoints(str(frm), str(to), etype)
        key = (a, b, etype)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            RelationEdge(
                from_id=a,
                to_id=b,
                type=etype,
                reason=reason or "related",
                confidence=conf,
                source="llm",
                model=model,
                topic=topic,
            )
        )
    return edges


def build_typed_topic_messages(
    topic: str,
    summaries: list[dict],
    *,
    candidate_hints: list[dict] | None = None,
) -> list[dict[str, str]]:
    system = (
        "You analyze Indonesian physics olympiad problems (OSK/OSN/OSP) and infer typed links.\n"
        "Output ONLY a single JSON object. No markdown fences, no analysis, no extra text.\n"
        "Schema:\n"
        '{"edges":[{"from_id":"OSK-…","to_id":"OSK-…","type":"prerequisite|similar|variant|harder",'
        '"reason":"short phrase","confidence":0.0}]}\n'
        "Rules:\n"
        "- Only use ids from the provided list. Never invent ids.\n"
        "- prerequisite: from is simpler; study before to.\n"
        "- similar: same idea / practice twin.\n"
        "- variant: near-repeat across years/exams.\n"
        "- harder: to is the next step up from from.\n"
        "- When 3+ problems are given, include at least some edges (prefer 3–12 high-confidence links).\n"
        "- 0-20 edges total; prefer high-confidence links.\n"
        "- confidence: 0-1."
    )
    user = {
        "topic": topic,
        "problems": summaries,
        "candidate_hints": candidate_hints or [],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def infer_typed_topic_relations(
    topic: str,
    summaries: list[dict],
    *,
    model: str | None = None,
    candidate_hints: list[dict] | None = None,
) -> tuple[list[RelationEdge], str | None]:
    if len(summaries) < 2:
        return [], None
    from src.llm.llm_client import ChatCompletionFailure, chat_completion_json

    if model is None:
        model = _default_model()
    completion = chat_completion_json(
        messages=build_typed_topic_messages(topic, summaries, candidate_hints=candidate_hints),
        model=model,
        max_tokens=2000,
        timeout_s=120.0,
    )
    if isinstance(completion, ChatCompletionFailure):
        return [], "completion_failure"
    data = extract_json_object(completion.content)
    if data is None:
        return [], "parse_failure"
    valid = {s["id"] for s in summaries}
    edges = parse_typed_edges_payload(data, valid_ids=valid, model=model, topic=topic)
    if not edges and not (data.get("edges") or []):
        return [], "empty_edges"
    return edges, None
