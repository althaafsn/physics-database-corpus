#!/usr/bin/env python3
"""Validate the public JSONL corpus using only the Python standard library."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")


def read_jsonl(name: str) -> list[dict]:
    path = DATASET / name
    rows: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{number}: expected a JSON object")
        rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: dataset is empty")
    return rows


def assert_relative(value: object, context: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or value.startswith(("/", "~")) or WINDOWS_ABSOLUTE.match(value):
        raise SystemExit(f"{context}: machine-specific path is not allowed: {value!r}")


def validate_problems(rows: list[dict]) -> None:
    ids = [row.get("id") for row in rows]
    if any(not isinstance(problem_id, str) or not problem_id for problem_id in ids):
        raise SystemExit("problems.jsonl: every record needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise SystemExit("problems.jsonl: duplicate problem id")
    if ids != sorted(ids):
        raise SystemExit("problems.jsonl: records must be sorted by id")
    for row in rows:
        problem_id = row["id"]
        for key, value in (row.get("source") or {}).items():
            assert_relative(value, f"{problem_id}.source.{key}")
        for image in row.get("images") or []:
            assert_relative(image.get("path"), f"{problem_id}.images.path")


def validate_solutions(rows: list[dict]) -> None:
    for row in rows:
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise SystemExit("solutions.jsonl: every record needs a problem_id")
        for key, value in (row.get("source") or {}).items():
            assert_relative(value, f"{problem_id}.source.{key}")


def validate_relations(rows: list[dict]) -> None:
    for row in rows:
        for key in ("from_id", "to_id", "type"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise SystemExit(f"relations.jsonl: every record needs {key}")


def main() -> None:
    datasets = {
        "problems": read_jsonl("problems.jsonl"),
        "solutions": read_jsonl("solutions.jsonl"),
        "relations": read_jsonl("relations.jsonl"),
    }
    validate_problems(datasets["problems"])
    validate_solutions(datasets["solutions"])
    validate_relations(datasets["relations"])
    print("OK " + " ".join(f"{name}={len(rows)}" for name, rows in datasets.items()))


if __name__ == "__main__":
    main()
