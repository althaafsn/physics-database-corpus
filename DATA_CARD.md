# Data Card

## Summary

This repository is the public data export behind
[Lab Fisika](https://labfisika.com), a physics olympiad problem browser and AI
tutor. It contains structured problem statements, worked solutions, and graph
relations extracted from educational source documents.

Current release:

- 667 canonical problems
- 644 Indonesian problem views
- 667 English problem views
- 506 worked solutions
- 2,134 problem-to-problem relations

## Formats and schemas

All large files use UTF-8 JSON Lines: one JSON object per line.

### `problems.jsonl`

Canonical parser records. Important fields include:

- `id`: stable problem identifier
- `content_locale`: native problem language, `id` or `en`
- `title`, `body_md`, `subparts`: native text
- `title_en`, `body_md_en`, `subparts_en`: accepted English translation when
  the native problem is Indonesian
- `topic`, `level`, `year`, `round`: retrieval metadata
- `flags`, `errors`, confidence fields: machine-readable quality signals
- `source`, `images`: path-sanitized source and figure metadata

### `solutions.jsonl`

Parsed solution records joined by `problem_id`. `solution_locale` states the
solution text language. Alignment, formatting, method, flags, errors, and
step-level fields describe parser output and quality.

### `relations.jsonl`

Directed edges with `from_id`, `to_id`, `type`, `confidence`, `reason`, and
`source`. Relation types include prerequisites, similarity, variants, and
difficulty progression.

### `rag.id.jsonl` and `rag.en.jsonl`

Ready-to-index views with:

- localized `title`, `problem_md`, and `subparts`
- stable metadata and source fields
- `has_solution` and a `solutions` list
- `relations_in` and `relations_out`

The locale applies to problem text. Every solution independently declares
`solution_locale`; the English RAG view may therefore link an Indonesian
worked solution without misrepresenting its language.

## Provenance and processing

Records are deterministic exports from the private Lab Fisika ingestion
workspace. Processing includes document conversion, segmentation, cleanup,
language detection, validation, solution alignment, accepted English
translation, and relation generation. Machine-specific paths, raw PDFs,
credentials, caches, and generated image binaries are excluded.

## Quality and limitations

- Parsed text can contain OCR, equation, Unicode, page-header, or segmentation
  errors.
- A solution can be aligned imperfectly or contain incomplete derivations.
- Graph relations include deterministic and model-assisted edges; inspect
  `source` and `confidence`.
- Figure metadata can be present without the corresponding image binary.
- English problem coverage is complete for this export; English solution
  coverage is not.
- Records with flags or errors are retained so downstream users can filter
  rather than receive silently discarded data.

This dataset is unsuitable as an unquestioned answer key or sole evaluator of
student correctness.

## Intended uses

- Physics search and retrieval experiments
- RAG indexing and tutor prototypes
- Curriculum and prerequisite-graph research
- Translation, OCR, parsing, and alignment evaluation
- Training-data preparation after task-specific review and filtering

## Rights and attribution

Repository-authored schemas, metadata, documentation, and software
transformations are provided under the repository's MIT license to the extent
the maintainer owns them. Source problem statements, diagrams, and worked
solutions may remain protected by their original authors or publishers and are
not relicensed by this repository. Users are responsible for checking source
terms and applicable law for their intended use.

Suggested attribution:

> Althaaf. *Bilingual Physics Problem Corpus*. 2026.
> https://github.com/althaafsn/physics-database-corpus
