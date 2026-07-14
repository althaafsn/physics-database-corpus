# Public corpus dataset

These deterministic JSONL files make the parsed physics corpus reviewable through ordinary GitHub pull requests:

- `problems.jsonl` — canonical parsed problem records, including Indonesian text, optional `title_en`/`body_md_en` translations, validation flags, and image metadata.
- `solutions.jsonl` — parsed worked solutions with alignment and formatting quality fields.
- `relations.jsonl` — prerequisite, similarity, variant, and difficulty edges.

Regenerate them from the pipeline outputs:

```bash
npm run export:dataset
```

The exporter removes machine-specific absolute paths. Raw PDFs, generated images, API keys, and model caches are intentionally excluded. A solution carrying `errors`, review flags, or low confidence is parsed data, not a claim that it is verified.
