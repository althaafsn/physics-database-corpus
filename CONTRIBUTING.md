# Contributing

Corrections are welcome through either the live editor at [labfisika.com/admin/problems](https://labfisika.com/admin/problems) or a GitHub pull request.

## Problem and translation corrections

In the public corpus repository, edit the matching JSON object in `dataset/problems.jsonl`. Indonesian source text lives in `title` and `body_md`; English translations live in `title_en`, `body_md_en`, and `subparts_en`. Preserve LaTeX delimiters, problem IDs, units, variables, subpart labels, and image references. A maintainer imports accepted data changes into the canonical corpus and regenerates the export.

Then regenerate and validate:

```bash
python3 scripts/validate_dataset.py
```

## Pipeline changes

Parsing and validation code lives in `src/`; command-line entry points live in `scripts/`; focused regression tests live in `tests/`. Include one small failing-then-passing fixture for parser changes.

## Pull request rules

- Explain the source of the correction and name the affected problem IDs.
- Do not commit raw PDFs, screenshots, model weights, caches, database files, private keys, or API credentials.
- Put screenshots in the pull-request description rather than the repository.
- Do not remove validation flags unless the underlying issue is actually fixed.
- Keep each JSONL record on one line and run `python3 scripts/validate_dataset.py`.
