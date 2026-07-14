"""Load repo-root local.env into os.environ (setdefault only)."""
from __future__ import annotations

import os
from pathlib import Path


def load_local_env(root: Path | None = None) -> None:
    base = root or Path(__file__).resolve().parents[1]
    env_path = base / "local.env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
