#!/usr/bin/env python3
"""Build deterministic similar-problem index from physics tags + TF-IDF."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.halliday.classify import ProblemTags
from src.halliday.similarity import build_similarity_index
from src.paths import PipelinePaths
from src.record_store import load_jsonl


def _load_tags(data: dict) -> ProblemTags:
    topics = data.get("topics") or data.get("chapters", [])
    details = data.get("details") or data.get("sections", [])
    return ProblemTags(
        problem_id=data["problem_id"],
        topics=topics,
        details=details,
        disciplines=data.get("disciplines", []),
        confidence=float(data.get("confidence", 0.5)),
        method=data.get("method", "unknown"),
        model=data.get("model"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build similar-problems index")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    paths = PipelinePaths.resolve(ROOT)
    catalog_path = paths.catalog_problems_path
    tags_path = paths.parsed_dir / "halliday" / "tags.jsonl"

    if not catalog_path.is_file():
        print(f"Missing {catalog_path}", file=sys.stderr)
        return 1
    if not tags_path.is_file():
        print(f"Missing {tags_path}. Run scripts/tag_halliday.py first.", file=sys.stderr)
        return 1

    records = load_jsonl(catalog_path, lenient=True)
    tags_by_id: dict[str, ProblemTags] = {}
    for line in tags_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tags = _load_tags(json.loads(line))
        tags_by_id[tags.problem_id] = tags

    missing = [r.id for r in records if r.id not in tags_by_id]
    if missing:
        print(f"Warning: {len(missing)} problems lack tags (skipped in index)")

    index = build_similarity_index(
        [r for r in records if r.id in tags_by_id],
        tags_by_id,
        top_k=args.top_k,
    )

    out_dir = paths.parsed_dir / "halliday"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 2,
        "method": "tfidf_cosine + topic/detail_tag_jaccard",
        "top_k": args.top_k,
        "problem_count": len(index),
    }
    (out_dir / "similarity-manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    neighbors_path = out_dir / "similarity.jsonl"
    with neighbors_path.open("w", encoding="utf-8") as fh:
        for pid, neighbors in sorted(index.items()):
            fh.write(
                json.dumps(
                    {"id": pid, "similar": [n.as_dict() for n in neighbors]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote similarity for {len(index)} problems → {neighbors_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
