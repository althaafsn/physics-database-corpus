from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.schema import ProblemRecord


def problem_id_aliases(records: list[ProblemRecord]) -> tuple[set[str], dict[str, str]]:
    """Return current ids and aliases emitted by older variant-aware runs."""
    valid = {record.id for record in records}
    aliases: dict[str, str] = {}
    for record in records:
        if record.level and record.year and record.variant is not None:
            old_id = f"{record.level}-{record.year}-v{record.variant}-{record.problem_number:02d}"
            if old_id != record.id:
                aliases[old_id] = record.id
    return valid, aliases


def canonical_problem_id(problem_id: str, valid: set[str], aliases: dict[str, str]) -> str:
    return problem_id if problem_id in valid else aliases.get(problem_id, problem_id)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp.replace(path)


def normalize_graph_files(parsed_dir: Path, records: list[ProblemRecord]) -> dict[str, int]:
    """Migrate variant aliases and remove edges to non-public problem ids."""
    valid, aliases = problem_id_aliases(records)
    result = {"prerequisite_nodes": 0, "prerequisite_edges": 0, "relation_edges": 0}

    prerequisites_path = parsed_dir / "graph" / "prerequisites.jsonl"
    if prerequisites_path.is_file():
        nodes: dict[str, dict[str, dict[str, Any]]] = {}
        for line in prerequisites_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            owner = canonical_problem_id(str(row.get("problem_id", "")), valid, aliases)
            if owner not in valid:
                continue
            node = nodes.setdefault(owner, {"prerequisites": {}, "unlocks": {}})
            for field in ("prerequisites", "unlocks"):
                for edge in row.get(field, []):
                    target = canonical_problem_id(str(edge.get("id", "")), valid, aliases)
                    if target not in valid or target == owner:
                        continue
                    candidate = {**edge, "id": target}
                    previous = node[field].get(target)
                    if previous is None or float(candidate.get("overlap_ratio", 0)) > float(
                        previous.get("overlap_ratio", 0)
                    ):
                        node[field][target] = candidate
        rows = []
        for owner in sorted(nodes):
            node = nodes[owner]
            row = {"problem_id": owner}
            for field in ("prerequisites", "unlocks"):
                edges = sorted(
                    node[field].values(),
                    key=lambda edge: (-float(edge.get("overlap_ratio", 0)), edge["id"]),
                )
                row[field] = edges
                result["prerequisite_edges"] += len(edges)
            rows.append(row)
        _write_jsonl(prerequisites_path, rows)
        result["prerequisite_nodes"] = len(rows)

    relations_path = parsed_dir / "graph" / "relations.jsonl"
    if relations_path.is_file():
        edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        for line in relations_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            from_id = canonical_problem_id(str(row.get("from_id", "")), valid, aliases)
            to_id = canonical_problem_id(str(row.get("to_id", "")), valid, aliases)
            edge_type = str(row.get("type", ""))
            if from_id not in valid or to_id not in valid or from_id == to_id:
                continue
            normalized = {**row, "from_id": from_id, "to_id": to_id}
            key = (from_id, to_id, edge_type)
            previous = edges.get(key)
            if previous is None or float(normalized.get("confidence", 0)) > float(
                previous.get("confidence", 0)
            ):
                edges[key] = normalized
        _write_jsonl(relations_path, list(edges.values()))
        result["relation_edges"] = len(edges)

    return result
