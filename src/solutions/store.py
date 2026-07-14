from __future__ import annotations

import json
from pathlib import Path

from src.solutions.schema import SolutionRecord


def solutions_jsonl_path(parsed_dir: Path) -> Path:
    return parsed_dir / "solutions" / "solutions.jsonl"


def solutions_bronze_dir(root: Path) -> Path:
    return root / "output_solutions"


def load_solutions(path: Path) -> list[SolutionRecord]:
    if not path.is_file():
        return []
    records: list[SolutionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(SolutionRecord(**json.loads(line)))
    return records


def save_solutions(path: Path, records: list[SolutionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def solutions_by_problem_id(records: list[SolutionRecord]) -> dict[str, SolutionRecord]:
    """Latest/best record wins per problem_id (higher alignment_confidence,
    then fewer errors)."""
    best: dict[str, SolutionRecord] = {}
    for rec in records:
        current = best.get(rec.problem_id)
        if current is None:
            best[rec.problem_id] = rec
            continue
        current_score = (current.alignment_confidence, -len(current.errors))
        new_score = (rec.alignment_confidence, -len(rec.errors))
        if new_score > current_score:
            best[rec.problem_id] = rec
    return best


def solution_status_by_problem_id(records: list[SolutionRecord]) -> dict[str, str]:
    """One of "verified" | "needs_review" per problem_id with at least one
    ingested solution segment (problems absent from the dict have none).

    "needs_review" wins over "verified" for a given problem_id if ANY
    ingested segment for it still needs review (better to under- than
    over-claim verified status in the editor)."""
    status: dict[str, str] = {}
    for rec in records:
        current = status.get(rec.problem_id)
        if current == "needs_review":
            continue
        status[rec.problem_id] = "needs_review" if rec.needs_review else "verified"
    return status
