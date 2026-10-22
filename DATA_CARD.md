# Data Card

## Summary

This repository is the public data export behind
[Lab Fisika](https://labfisika.com), a physics olympiad problem browser and AI
tutor. It contains structured problem statements, worked solutions, and graph
relations extracted from educational source documents.

Current release (22 October 2026, schema version 2):

- 572 current reader-catalog problems
- 564 Indonesian and 572 English problem views
- 314 filtered parsed text solutions, marked `parsed_unverified`
- 28 reviewed source-image solution metadata entries
- 1,812 problem-to-problem relations

The earlier raw-gold export included more records. This release follows the
current reader catalog and excludes orphan solutions, parser-review records,
unsafe text, and known source-label conflicts. Original inputs are preserved in
the source workspace; exclusion is not a correction or mathematical certification.
Counts and exclusion reasons are in `data/manifest.json`.

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

### `solution.images.jsonl`

Source-crop metadata joined by `problem_id`, including author, source PDF
basename/hash, review note, page/bounds, source solution number, and existing
LabFisika image URLs. `source_crops_reviewed` means source identity and faithful
cropping were checked. It does not mean every mathematical derivation is certified.
Raw PDFs and image binaries are excluded. Newly extracted unreviewed drafts are
not public in this release.

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
- Parser-review solutions and known mismatches are excluded; the manifest records
  reasons. Remaining text is still parsed data, not fully reviewed mathematics.
- Known promotional lines are removed from text, including original-text fields,
  translations, parts, and steps. Native-text checks cannot prove that scanned
  source content is clean or that every diagram/equation was extracted correctly.

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
