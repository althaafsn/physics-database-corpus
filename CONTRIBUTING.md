# Contributing

Corrections are welcome through the
[rendered editor](https://labfisika.com/admin/problems) or a GitHub pull
request.

## Data corrections

- Edit the matching one-line record in `data/problems.jsonl`,
  `data/solutions.jsonl`, or `data/relations.jsonl`.
- Preserve problem IDs, LaTeX delimiters, units, variables, subpart labels, and
  source metadata.
- Explain the evidence for the correction and list every affected problem ID.
- Do not hand-edit `data/rag.*.jsonl` or `data/manifest.json`; maintainers
  regenerate those files from accepted canonical changes before merge.
- Put screenshots in the pull-request description, not in this repository.

## Do not commit

Raw PDFs, screenshots, generated figures, model weights, caches, databases,
private paths, API keys, or credentials are intentionally excluded.

GitHub Actions parses every JSONL line, checks counts and relation endpoints,
and rejects common private-path or secret patterns.
