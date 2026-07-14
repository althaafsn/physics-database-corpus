#!/usr/bin/env python3
"""Run the full solution pipeline: ingest -> concepts -> graph.

Usage:
  python3 scripts/run_solution_pipeline.py
  python3 scripts/run_solution_pipeline.py --skip-ingest --llm-graph
  python3 scripts/run_solution_pipeline.py --only-ingest osk-fisika-2013-solusi.pdf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    print(f"\n>> {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Solution ingest + tagging + graph pipeline")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--only-ingest", type=str, default=None, help="Comma-separated PDF filenames")
    parser.add_argument("--ingest-limit", type=int, default=None)
    parser.add_argument("--llm-graph", action="store_true", help="Add LLM topic relations to graph")
    parser.add_argument("--include-review", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_ingest:
        ingest_cmd = [py, "scripts/ingest_solutions.py"]
        if args.only_ingest:
            ingest_cmd += ["--only", args.only_ingest]
        if args.ingest_limit is not None:
            ingest_cmd += ["--limit", str(args.ingest_limit)]
        code = _run(ingest_cmd)
        if code != 0:
            return code

    code = _run([py, "scripts/extract_solution_concepts.py"] + (
        ["--include-review"] if args.include_review else []
    ))
    if code != 0:
        return code

    graph_cmd = [py, "scripts/build_prerequisite_graph.py"]
    if args.llm_graph:
        graph_cmd.append("--llm")
    code = _run(graph_cmd)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
