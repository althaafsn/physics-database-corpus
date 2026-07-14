#!/usr/bin/env python3
"""Export path-sanitized JSONL corpus files for GitHub collaboration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _filename(value: str | None) -> str | None:
    if not value:
        return value
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _sanitize_source(source: dict | None) -> dict:
    return {key: _filename(value) for key, value in (source or {}).items()}


def sanitize_problem(row: dict) -> dict:
    clean = dict(row)
    clean["source"] = _sanitize_source(row.get("source"))
    clean["images"] = [
        dict(image) | {"path": _filename(image.get("path"))}
        for image in row.get("images", [])
    ]
    return clean


def sanitize_solution(row: dict) -> dict:
    return dict(row) | {"source": _sanitize_source(row.get("source"))}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def export_dataset(parsed_dir: Path, output_dir: Path) -> dict[str, int]:
    problems = sorted(
        (sanitize_problem(row) for row in _read_jsonl(parsed_dir / "gold" / "problems.jsonl")),
        key=lambda row: row.get("id", ""),
    )
    solutions = sorted(
        (sanitize_solution(row) for row in _read_jsonl(parsed_dir / "solutions" / "solutions.jsonl")),
        key=lambda row: (row.get("problem_id", ""), row.get("solution_number", 0)),
    )
    relations = sorted(
        _read_jsonl(parsed_dir / "graph" / "relations.jsonl"),
        key=lambda row: (row.get("from_id", ""), row.get("to_id", ""), row.get("type", "")),
    )
    datasets = {"problems": problems, "solutions": solutions, "relations": relations}
    for name, rows in datasets.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)
    return {name: len(rows) for name, rows in datasets.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-dir", type=Path, default=ROOT / "parsed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.parsed_dir, args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
