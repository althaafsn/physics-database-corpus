#!/usr/bin/env python3
"""Build typed problem relations -> parsed/graph/relations.jsonl.

Always uses LLM inference (src/graph/typed_relations.py) per topic on OSK/OSN/OSP
problems. Deterministic prerequisite candidates from build_prerequisite_graph are
passed as optional hints to the model and always merged as a baseline after LLM
chunks (LLM edges win on duplicate keys).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env_loader import load_local_env

load_local_env(ROOT)

from src.graph.build_prerequisites import build_prerequisite_graph
from src.graph.llm_relations import summaries_for_topic
from src.graph.relation_types import RelationEdge
from src.graph.typed_relations import (
    filter_os_records,
    infer_typed_topic_relations,
    iter_batches,
)
from src.paths import PipelinePaths
from src.record_store import load_jsonl
from src.solutions.store import load_solutions, solutions_by_problem_id, solutions_jsonl_path

BATCH_SIZE = 40


def _load_halliday_concepts(parsed_dir: Path) -> dict[str, list[str]]:
    path = parsed_dir / "halliday" / "tags.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        details = data.get("details") or []
        if details:
            out[data["problem_id"]] = list(details)
    return out


def _load_solution_concepts(parsed_dir: Path) -> dict[str, list[str]]:
    path = parsed_dir / "concepts" / "solution_concepts.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        concepts = data.get("solved_concepts") or []
        if concepts:
            out[data["problem_id"]] = concepts
    return out


def _deterministic_baseline_edges(
    topic: str,
    topic_ids: set[str],
    prereq_graph: dict,
    seen: set[tuple[str, str, str]],
) -> list[RelationEdge]:
    edges: list[RelationEdge] = []
    for target_id in sorted(topic_ids):
        graph_node = prereq_graph.get(target_id)
        if not graph_node:
            continue
        for pre in graph_node.prerequisites[:5]:
            if pre.id not in topic_ids:
                continue
            edge = RelationEdge(
                from_id=pre.id,
                to_id=target_id,
                type="prerequisite",
                reason="concept-subset prerequisite",
                confidence=pre.overlap_ratio,
                source="deterministic",
                model=None,
                topic=topic,
            )
            key = edge.key()
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
    return edges


def _candidate_hints_for_topic(
    os_ids: set[str],
    prereq_graph: dict,
) -> list[dict]:
    hints: list[dict] = []
    for target_id, graph_node in prereq_graph.items():
        if target_id not in os_ids:
            continue
        for pre in graph_node.prerequisites[:5]:
            if pre.id not in os_ids:
                continue
            hints.append(
                {
                    "from_id": pre.id,
                    "to_id": target_id,
                    "type": "prerequisite",
                }
            )
    return hints


def main() -> int:
    parser = argparse.ArgumentParser(description="Build typed OS* problem relations graph")
    parser.add_argument(
        "--topics",
        type=str,
        default=None,
        help="Comma-separated topics (default: all topics with OS* problems)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts without writing relations.jsonl",
    )
    args = parser.parse_args()

    paths = PipelinePaths.resolve(ROOT)
    concepts_by_id = _load_halliday_concepts(paths.parsed_dir)
    solution_concepts = _load_solution_concepts(paths.parsed_dir)
    concepts_by_id.update(solution_concepts)

    if not concepts_by_id:
        print("No concept tags found. Run extract_solution_concepts.py or tag_halliday.py first.")
        return 1

    records = load_jsonl(paths.gold_problems_path, lenient=True)
    os_records = filter_os_records(records)
    if not os_records:
        print("No OSK/OSN/OSP problems found in gold corpus.")
        return 1

    os_ids = {r.id for r in os_records}
    prereq_graph = build_prerequisite_graph(os_records, concepts_by_id)

    solution_ids: set[str] = set()
    sol_path = solutions_jsonl_path(paths.parsed_dir)
    if sol_path.is_file():
        solutions = [s for s in load_solutions(sol_path) if not s.needs_review]
        solution_ids = set(solutions_by_problem_id(solutions).keys())

    topics = sorted({r.topic for r in os_records})
    if args.topics:
        wanted = {t.strip() for t in args.topics.split(",") if t.strip()}
        topics = [t for t in topics if t in wanted]
        missing = wanted - set(topics)
        for t in sorted(missing):
            print(f"  skip unknown/empty topic: {t}", flush=True)

    seen: set[tuple[str, str, str]] = set()
    all_edges: list[RelationEdge] = []

    for topic in topics:
        topic_ids = {r.id for r in os_records if r.topic == topic}
        summaries = summaries_for_topic(topic, os_records, concepts_by_id, solution_ids)
        if len(summaries) < 2:
            print(f"  {topic}: skipped ({len(summaries)} OS* problems)", flush=True)
            continue

        summary_dicts = [s.as_dict() for s in summaries]
        hints = _candidate_hints_for_topic(topic_ids, prereq_graph)
        chunks = list(iter_batches(summary_dicts, BATCH_SIZE))
        topic_added = 0
        topic_raw = 0

        for chunk_idx, chunk in enumerate(chunks, start=1):
            chunk_ids = {s["id"] for s in chunk}
            chunk_hints = [
                h for h in hints if h["from_id"] in chunk_ids and h["to_id"] in chunk_ids
            ]
            edges, note = infer_typed_topic_relations(
                topic, chunk, candidate_hints=chunk_hints
            )

            added = 0
            for edge in edges:
                key = edge.key()
                if key in seen:
                    continue
                seen.add(key)
                all_edges.append(edge)
                added += 1

            topic_added += added
            topic_raw += len(edges)
            print(
                f"  {topic} chunk {chunk_idx}/{len(chunks)}: "
                f"{len(chunk)} problems, +{added} edges",
                flush=True,
            )
            if len(edges) == 0:
                detail = f" ({note})" if note else ""
                print(
                    f"  warning: {topic} chunk {chunk_idx}/{len(chunks)} "
                    f"returned 0 edges after LLM call{detail}",
                    flush=True,
                )

        baseline = _deterministic_baseline_edges(topic, topic_ids, prereq_graph, seen)
        if baseline:
            all_edges.extend(baseline)
            topic_added += len(baseline)
            print(
                f"  {topic}: +{len(baseline)} deterministic prerequisite edges (baseline)",
                flush=True,
            )

        print(
            f"  {topic}: {len(summaries)} problems, "
            f"{len(hints)} hints, +{topic_added} edges ({topic_raw} raw)",
            flush=True,
        )

    print(
        f"Total: {len(all_edges)} typed edges across {len(topics)} topic(s) "
        f"({len(os_records)} OS* problems)",
        flush=True,
    )

    if args.dry_run:
        print("Dry run: not writing relations.jsonl")
        return 0

    out_path = paths.parsed_dir / "graph" / "relations.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for edge in all_edges:
            f.write(
                json.dumps(
                    {
                        "from_id": edge.from_id,
                        "to_id": edge.to_id,
                        "type": edge.type,
                        "reason": edge.reason,
                        "confidence": round(edge.confidence, 3),
                        "source": edge.source,
                        "model": edge.model,
                        "topic": edge.topic,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {len(all_edges)} edges -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
