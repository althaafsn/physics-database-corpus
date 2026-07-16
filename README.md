# Bilingual Physics Problem Corpus

[![Validate corpus](https://github.com/althaafsn/physics-database-corpus/actions/workflows/validate.yml/badge.svg)](https://github.com/althaafsn/physics-database-corpus/actions/workflows/validate.yml)
[![Lab Fisika](https://img.shields.io/badge/app-labfisika.com-0ea5e9)](https://labfisika.com)

A public, data-only export of parsed olympiad physics problems, worked
solutions, and problem-roadmap relations from [Lab Fisika](https://labfisika.com).
It is ready to stream as JSONL for search, evaluation, training preparation,
or retrieval-augmented generation.

## Files

| File | Rows | Use |
| --- | ---: | --- |
| [`data/problems.jsonl`](data/problems.jsonl) | 667 | Canonical parsed problems, original text, and accepted English translations |
| [`data/solutions.jsonl`](data/solutions.jsonl) | 506 | Parsed worked solutions with explicit language and quality metadata |
| [`data/relations.jsonl`](data/relations.jsonl) | 2,134 | Prerequisite, similarity, variant, and difficulty edges |
| [`data/rag.id.jsonl`](data/rag.id.jsonl) | 644 | Indonesian problems joined to solutions and graph neighbors |
| [`data/rag.en.jsonl`](data/rag.en.jsonl) | 667 | English problems joined to solutions and graph neighbors |
| [`data/manifest.json`](data/manifest.json) | — | Machine-readable schema version and counts |

English problem rows contain native or accepted English text. Indonesian
problem rows contain Indonesian text. Linked solutions keep their real
`solution_locale`, so an Indonesian solution is never silently labelled as
English.

## Use it

```bash
git clone https://github.com/althaafsn/physics-database-corpus.git
cd physics-database-corpus
```

```python
import json

with open("data/rag.en.jsonl", encoding="utf-8") as source:
    records = [json.loads(line) for line in source]

record = records[0]
document = "\n\n".join(
    [record["title"], record["problem_md"]]
    + [solution["body_md"] for solution in record["solutions"]]
)
print(record["id"], document[:500])
```

Each RAG row already contains a stable problem ID, locale, Markdown problem
text, subparts, metadata, available solutions, incoming relations, and outgoing
relations. No PDF parsing or joins are required.

See [DATA_CARD.md](DATA_CARD.md) for schemas, provenance, quality limitations,
rights, and intended use. Corrections are welcome through
[pull requests](CONTRIBUTING.md) or the
[rendered Lab Fisika editor](https://labfisika.com/admin/problems).

## Important

This is parsed educational source material, not a fully human-verified answer
key. OCR, alignment, formatting, and source-rights caveats remain. Review
records and applicable source terms before publishing derived products or using
the material commercially.
