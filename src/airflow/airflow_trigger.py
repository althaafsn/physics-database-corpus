"""Trigger Airflow DAG runs via REST API (after SQS enqueue)."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


def airflow_trigger_enabled() -> bool:
    return bool(os.environ.get("AIRFLOW_API_URL", "").strip())


def trigger_ingest_dag(*, reason: str = "upload", job_id: str | None = None) -> str | None:
    """Return dag_run_id if triggered, else None when Airflow is not configured."""
    base = os.environ.get("AIRFLOW_API_URL", "").strip().rstrip("/")
    if not base:
        return None

    dag_id = os.environ.get("AIRFLOW_INGEST_DAG_ID", "ingest_gpu_batch")
    user = os.environ.get("AIRFLOW_API_USER", "admin")
    password = os.environ.get("AIRFLOW_API_PASSWORD", "")
    conf: dict[str, Any] = {"reason": reason}
    if job_id:
        conf["job_id"] = job_id

    payload = json.dumps({"conf": conf}).encode("utf-8")
    url = f"{base}/api/v1/dags/{dag_id}/dagRuns"
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    token = f"{user}:{password}".encode()
    import base64

    request.add_header("Authorization", "Basic " + base64.b64encode(token).decode())

    try:
        # Keep short: upload already wrote S3+SQS; a slow Airflow must not fail the HTTP response.
        with urllib.request.urlopen(request, timeout=8) as response:
            body = json.loads(response.read().decode())
            return str(body.get("dag_run_id", ""))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Airflow trigger failed ({exc.code}): {detail}") from exc
    except TimeoutError as exc:
        raise RuntimeError("Airflow trigger timed out") from exc
    except OSError as exc:
        raise RuntimeError(f"Airflow trigger connection failed: {exc}") from exc
