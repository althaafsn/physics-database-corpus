# Physics Database Corpus and Parsing Pipeline

This repository contains the public, reviewable output and Python pipeline behind [Lab Fisika](https://labfisika.com):

- `dataset/problems.jsonl` — 667 parsed olympiad physics problems
- `dataset/solutions.jsonl` — 506 parsed worked solutions
- `dataset/relations.jsonl` — 2,134 roadmap edges connecting prerequisites, variants, similar problems, and difficulty progression
- `src/` — parsing, cleanup, validation, translation, solution alignment, and graph-building modules
- `scripts/` — command-line entry points for ingestion, translation, auditing, and graph generation

The dataset includes both Indonesian source fields and optional English translation fields. It deliberately excludes raw PDFs, screenshots, generated image assets, credentials, model caches, and machine-specific paths.

## Contribute a correction

For a small correction, use the [live rendered editor](https://labfisika.com/admin/problems) or edit the matching one-line JSON record and open a pull request. Please explain the source of the correction and affected problem IDs. See [CONTRIBUTING.md](CONTRIBUTING.md).

Validate dataset edits locally:

```bash
python3 scripts/validate_dataset.py
```

## Run the pipeline

Python 3.11+ is recommended.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-web.txt
python scripts/ingest.py --help
```

PDF layout conversion uses Marker and can be installed separately with `requirements-marker.txt`. AWS queue support is optional and listed in `requirements-ingest-aws.txt`. AI-assisted repair, translation, and graph generation use an OpenAI-compatible endpoint configured through environment variables; no key is stored here.

The exported records are parser output and may still carry validation flags or imperfect OCR. A record's presence is not a claim that it has been human-verified.

## License

Pipeline code is licensed under [MIT](LICENSE). Source problem statements and solutions remain subject to their original owners' terms; this repository provides structured metadata and transformations for research and education.
