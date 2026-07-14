from __future__ import annotations

from typing import Any, Callable

LogFn = Callable[[str], None]


def default_log(message: str) -> None:
    print(message, flush=True)


def log_repair_event(
    log: LogFn | None,
    *,
    phase: str,
    record_id: str,
    index: int | None = None,
    total: int | None = None,
    **details: Any,
) -> None:
    if log is None:
        return
    prefix_parts = ["[LLM"]
    if index is not None and total is not None:
        prefix_parts.append(f" {index}/{total}")
    prefix_parts.append(f"] {record_id}")
    prefix = "".join(prefix_parts)

    detail_parts: list[str] = [phase]
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, list):
            detail_parts.append(f"{key}={','.join(str(v) for v in value)}")
        else:
            detail_parts.append(f"{key}={value}")
    log(f"{prefix} | {' | '.join(detail_parts)}")
