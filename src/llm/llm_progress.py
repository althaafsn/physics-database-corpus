from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from src.llm.llm_client import DEFAULT_BASE_URL, netra_provider_info

RepairStatus = Literal[
    "succeeded",
    "rejected",
    "api_error",
    "parse_error",
    "truncated",
    "cached",
]


class RepairProgressStore:
    """Tracks per-record LLM repair outcomes and Netra usage metrics."""

    def __init__(self, path: Path, *, model: str) -> None:
        self.path = path
        self.model = model
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self._empty()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return self._empty()
        if data.get("model") != self.model:
            return self._empty()
        data.setdefault("records", {})
        data.setdefault("stats", self._compute_stats(data["records"]))
        data.setdefault("usage_totals", self._compute_usage_totals(data["records"]))
        return data

    def _empty(self) -> dict[str, Any]:
        info = netra_provider_info()
        return {
            "model": self.model,
            "provider": info["provider"],
            "base_url": info["base_url"],
            "updated_at": None,
            "records": {},
            "stats": self._compute_stats({}),
            "usage_totals": self._compute_usage_totals({}),
        }

    @staticmethod
    def _compute_stats(records: dict[str, Any]) -> dict[str, int]:
        stats = {
            "attempted": len(records),
            "succeeded": 0,
            "rejected": 0,
            "api_error": 0,
            "parse_error": 0,
            "truncated": 0,
            "cached": 0,
        }
        for entry in records.values():
            status = entry.get("status")
            if status in stats:
                stats[status] += 1
        return stats

    @staticmethod
    def _compute_usage_totals(records: dict[str, Any]) -> dict[str, Any]:
        totals = {
            "api_calls": 0,
            "cached_hits": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_s": 0.0,
            "wall_latency_s": 0.0,
            "avg_completion_tokens_per_s": None,
            "avg_total_tokens_per_s": None,
            "avg_latency_s": None,
        }
        completion_rates: list[float] = []
        total_rates: list[float] = []
        latencies: list[float] = []

        for entry in records.values():
            if entry.get("from_cache"):
                totals["cached_hits"] += 1
                continue
            usage = entry.get("usage")
            if not isinstance(usage, dict):
                continue
            totals["api_calls"] += 1
            totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
            totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
            totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            latency = float(usage.get("latency_s", 0) or 0)
            wall = float(usage.get("wall_latency_s", latency) or latency)
            totals["latency_s"] += latency
            totals["wall_latency_s"] += wall
            latencies.append(latency)
            if usage.get("completion_tokens_per_s") is not None:
                completion_rates.append(float(usage["completion_tokens_per_s"]))
            if usage.get("total_tokens_per_s") is not None:
                total_rates.append(float(usage["total_tokens_per_s"]))

        if completion_rates:
            totals["avg_completion_tokens_per_s"] = round(
                sum(completion_rates) / len(completion_rates), 2
            )
        if total_rates:
            totals["avg_total_tokens_per_s"] = round(sum(total_rates) / len(total_rates), 2)
        if latencies:
            totals["avg_latency_s"] = round(sum(latencies) / len(latencies), 2)

        totals["latency_s"] = round(totals["latency_s"], 2)
        totals["wall_latency_s"] = round(totals["wall_latency_s"], 2)
        info = netra_provider_info()
        totals["provider"] = info["provider"]
        totals["base_url"] = info["base_url"]
        return totals

    def get(self, record_id: str) -> dict[str, Any] | None:
        entry = self._data["records"].get(record_id)
        return entry if isinstance(entry, dict) else None

    def mark(
        self,
        record_id: str,
        *,
        cache_key: str,
        status: RepairStatus,
        error: str | None = None,
        from_cache: bool = False,
        usage: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "cache_key": cache_key,
            "status": status,
            "from_cache": from_cache,
            "error": error,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if usage is not None:
            entry["usage"] = usage
        self._data["records"][record_id] = entry
        self._data["updated_at"] = datetime.now(UTC).isoformat()
        self._data["stats"] = self._compute_stats(self._data["records"])
        self._data["usage_totals"] = self._compute_usage_totals(self._data["records"])
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._data["stats"])

    @property
    def usage_totals(self) -> dict[str, Any]:
        return dict(self._data.get("usage_totals", {}))

    def reset(self) -> None:
        self._data = self._empty()
        self.save()


def format_usage_summary(usage_totals: dict[str, Any], *, model: str) -> str:
    provider = usage_totals.get("provider", "netra")
    base_url = usage_totals.get("base_url", DEFAULT_BASE_URL)
    lines = [
        f"Netra LLM usage ({model} @ {provider})",
        f"  Endpoint: {base_url}",
        f"  API calls: {usage_totals.get('api_calls', 0)}"
        f" | cache hits: {usage_totals.get('cached_hits', 0)}",
        (
            "  Tokens: "
            f"{usage_totals.get('total_tokens', 0):,} total "
            f"({usage_totals.get('prompt_tokens', 0):,} prompt + "
            f"{usage_totals.get('completion_tokens', 0):,} completion)"
        ),
    ]
    avg_gen = usage_totals.get("avg_completion_tokens_per_s")
    avg_total = usage_totals.get("avg_total_tokens_per_s")
    if avg_gen is not None or avg_total is not None:
        lines.append(
            "  Throughput: "
            f"{avg_gen or 'n/a'} gen tok/s avg"
            f" | {avg_total or 'n/a'} total tok/s avg"
        )
    lines.append(
        "  Latency: "
        f"{usage_totals.get('latency_s', 0):.1f}s model time"
        f" | {usage_totals.get('wall_latency_s', 0):.1f}s wall"
        f" | {usage_totals.get('avg_latency_s', 'n/a')}s avg/call"
    )
    return "\n".join(lines)
