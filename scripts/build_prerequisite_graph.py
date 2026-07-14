#!/usr/bin/env python3
"""Build prerequisite graph -> parsed/graph/prerequisites.jsonl.

Concept sources (in priority order):
  1. parsed/concepts/solution_concepts.jsonl (from worked solutions)
  2. parsed/halliday/tags.jsonl detail ids (fallback for problems without solutions)

Optional --llm adds LLM-inferred edges per topic (src/graph/llm_relations.py).
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

from src.graph.build_prerequisites import PrerequisiteEdge, build_prerequisite_graph
from src.graph.llm_relations import infer_topic_relations, merge_llm_edges, summaries_for_topic
from src.paths import PipelinePaths
from src.record_store import load_jsonl
from src.solutions.store import load_solutions, solutions_by_problem_id, solutions_jsonl_path


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build problem prerequisite graph")
    parser.add_argument("--llm", action="store_true", help="Add LLM-inferred topic relations")
    parser.add_argument("--topics", type=str, default=None, help="Comma-separated topics for --llm only")
    args = parser.parse_args()

    paths = PipelinePaths.resolve(ROOT)
    concepts_by_id = _load_halliday_concepts(paths.parsed_dir)
    solution_concepts = _load_solution_concepts(paths.parsed_dir)
    concepts_by_id.update(solution_concepts)  # solution concepts override halliday

    if not concepts_by_id:
        print("No concept tags found. Run extract_solution_concepts.py or tag_halliday.py first.")
        return 1

    records = load_jsonl(paths.gold_problems_path, lenient=True)
    graph = build_prerequisite_graph(records, concepts_by_id)

    solution_ids: set[str] = set()
    sol_path = solutions_jsonl_path(paths.parsed_dir)
    if sol_path.is_file():
        solutions = [s for s in load_solutions(sol_path) if not s.needs_review]
        solution_ids = set(solutions_by_problem_id(solutions).keys())

    if args.llm:
        topics = sorted({r.topic for r in records})
        if args.topics:
            wanted = {t.strip() for t in args.topics.split(",") if t.strip()}
            topics = [t for t in topics if t in wanted]
        raw_edges: dict[str, list[PrerequisiteEdge]] = {
            pid: list(g.prerequisites) for pid, g in graph.items()
        }
        llm_total = 0
        for topic in topics:
            summaries = summaries_for_topic(topic, records, concepts_by_id, solution_ids)
            if len(summaries) < 2:
                continue
            edges = infer_topic_relations(topic, summaries)
            llm_total += len(edges)
            merge_llm_edges(raw_edges, edges)
            print(f"  LLM {topic}: +{len(edges)} edges", flush=True)
        # Rebuild graph objects with merged prereqs
        for pid, g in graph.items():
            g.prerequisites = sorted(
                raw_edges.get(pid, []),
                key=lambda e: (-e.overlap_ratio, e.id),
            )[:8]

    out_path = paths.parsed_dir / "graph" / "prerequisites.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for problem_id in sorted(graph):
            fh.write(json.dumps(graph[problem_id].as_dict(), ensure_ascii=False) + "\n")

    with_prereqs = sum(1 for g in graph.values() if g.prerequisites)
    with_unlocks = sum(1 for g in graph.values() if g.unlocks)
    print(f"Wrote {len(graph)} graph nodes -> {out_path}")
    print(
        f"  concepts: {len(solution_concepts)} from solutions, "
        f"{len(concepts_by_id) - len(solution_concepts)} halliday fallback"
    )
    print(f"  {with_prereqs} problems have >=1 prerequisite, {with_unlocks} unlock >=1 later problem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
