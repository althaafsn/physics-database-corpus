"""S3 corpus sync for queue-based ingest (control plane + GPU worker)."""
from __future__ import annotations

import os
from pathlib import Path

from src.paths import PipelinePaths
from src.record_store import load_jsonl, merge_records, save_jsonl, source_pdf_key


def corpus_bucket() -> str:
    bucket = os.environ.get("CORPUS_S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("CORPUS_S3_BUCKET is not set")
    return bucket


def s3_client():
    import boto3

    return boto3.client("s3")


def upload_file(local_path: Path, s3_key: str) -> None:
    s3_client().upload_file(str(local_path), corpus_bucket(), s3_key)


def download_file(s3_key: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client().download_file(corpus_bucket(), s3_key, str(local_path))


def upload_tree(local_root: Path, s3_prefix: str) -> int:
    if not local_root.is_dir():
        return 0
    count = 0
    prefix = s3_prefix.strip("/")
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(local_root).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        upload_file(path, key)
        count += 1
    return count


def download_prefix(s3_prefix: str, local_root: Path) -> int:
    """Download all objects under prefix into local_root (preserve relative paths)."""
    client = s3_client()
    bucket = corpus_bucket()
    prefix = s3_prefix.strip("/")
    if prefix:
        prefix = prefix + "/"
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :] if prefix else key
            dest = local_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            count += 1
    return count


def incoming_job_prefix(job_id: str) -> str:
    return f"parsed/incoming/{job_id}"


def merge_incoming_job(paths: PipelinePaths, *, job_id: str, slug: str) -> dict[str, int]:
    """Merge worker-uploaded gold/silver for one job into the API host corpus."""
    import tempfile

    paths.ensure_dirs()
    with tempfile.TemporaryDirectory(prefix="corpus_merge_") as tmp:
        incoming = Path(tmp) / "incoming"
        downloaded = download_prefix(incoming_job_prefix(job_id), incoming)
        if downloaded == 0:
            return {"downloaded": 0, "gold_merged": 0, "silver_merged": 0}

        stats = {"downloaded": downloaded, "gold_merged": 0, "silver_merged": 0}

        incoming_gold = incoming / "gold" / "problems.jsonl"
        if incoming_gold.is_file():
            new_gold = load_jsonl(incoming_gold, lenient=True)
            existing_gold = load_jsonl(paths.gold_problems_path, lenient=True)
            replace_pdfs = {source_pdf_key(r) for r in new_gold}
            merged = merge_records(
                existing_gold,
                new_gold,
                replace_source_pdfs=replace_pdfs,
                replace_slugs={slug},
            )
            save_jsonl(paths.gold_problems_path, merged)
            save_jsonl(paths.silver_problems_path, merged)
            stats["gold_merged"] = len(new_gold)

        incoming_assets = incoming / "assets"
        if incoming_assets.is_dir():
            for src in incoming_assets.rglob("*"):
                if src.is_file():
                    rel = src.relative_to(incoming_assets)
                    dest = paths.assets_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(src.read_bytes())

        bronze_prefix = f"bronze/{slug}"
        download_prefix(bronze_prefix, paths.bronze_dir)

        raw_prefix = f"raw/pdfs/{job_id}"
        download_prefix(raw_prefix, paths.pdf_dir)

    return stats


def sync_mart_to_s3(paths: PipelinePaths, *, s3_prefix: str = "mart/public/data") -> int:
    """Upload public/data export tree for optional CloudFront pull."""
    mart = paths.root / "public" / "data"
    if not mart.is_dir():
        return 0
    return upload_tree(mart, s3_prefix)
