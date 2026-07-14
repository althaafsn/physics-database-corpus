#!/usr/bin/env python3
"""Extract solved_concepts from ingested worked solutions -> parsed/concepts/solution_concepts.jsonl.

Feeds each problem's *solution* text (not just the problem statement) through
the Halliday taxonomy LLM classifier so tags reflect the technique actually
used - see src/halliday/solution_concepts.py.
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

from src.halliday.solution_concepts import extract_solution_concepts
from src.llm.llm_client import _llm_provider
from src.paths import PipelinePaths
from src.record_store import load_jsonl
from src.solutions.store import load_solutions, solutions_by_problem_id, solutions_jsonl_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract solved_concepts from worked solutions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated problem ids")
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Also process solutions flagged alignment_review_required/errors (off by default)",
    )
    args = parser.parse_args()

    paths = PipelinePaths.resolve(ROOT)
    problems = load_jsonl(paths.gold_problems_path, lenient=True)
    problems_by_id = {p.id: p for p in problems}

    solutions_path = solutions_jsonl_path(paths.parsed_dir)
    solutions = load_solutions(solutions_path)
    if not args.include_review:
        solutions = [s for s in solutions if not s.needs_review]
    best = solutions_by_problem_id(solutions)

    targets = [(pid, sol) for pid, sol in best.items() if pid in problems_by_id]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        targets = [(pid, sol) for pid, sol in targets if pid in wanted]
    targets.sort(key=lambda item: item[0])
    if args.limit is not None:
        targets = targets[: args.limit]

    if not targets:
        print(f"No usable solutions found in {solutions_path}. Run scripts/ingest_solutions.py first.")
        return 1

    out_dir = paths.parsed_dir / "concepts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "solution_concepts.jsonl"

    llm_count = 0
    fallback_count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, (problem_id, solution) in enumerate(targets, start=1):
            rec = problems_by_id[problem_id]
            tags = extract_solution_concepts(rec, solution.body_md)
            if tags.method == "llm_solution":
                llm_count += 1
            else:
                fallback_count += 1
            fh.write(json.dumps(tags.as_dict(), ensure_ascii=False) + "\n")
            fh.flush()
            if i % 10 == 0 or i == len(targets):
                print(f"  extracted {i}/{len(targets)} ({llm_count} llm, {fallback_count} fallback)…", flush=True)

    print(f"Wrote {len(targets)} solution-concept records -> {out_path}")
    print(f"provider={_llm_provider()} llm={llm_count} heuristic_fallback={fallback_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
