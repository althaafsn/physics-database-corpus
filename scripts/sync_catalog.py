#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog import sync_catalog
from src.env_loader import load_local_env
from src.paths import PipelinePaths


def main() -> int:
    load_local_env()
    paths = PipelinePaths.resolve()
    meta = sync_catalog(paths)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
