"""Offline figure captions for bronze diagrams (image-only vision calls)."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from src.llm.llm_client import get_client

DEFAULT_CAPTION_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
FALLBACK_CAPTION_MODELS = (
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
)

CAPTION_PROMPT = """Describe this physics olympiad figure for a tutor.
ONLY what is visibly drawn. Be precise about objects, labels, axes, angles,
arrows/forces, springs, walls, tracks, and printed symbols.
Do NOT solve the problem. Do NOT invent labels that are not visible.
If unclear, say "unclear: ...".
Write 4-8 short bullet points in English."""


@dataclass(frozen=True)
class FigureCaption:
    filename: str
    caption: str
    model: str
    status: str  # ok | empty | error | skipped


def caption_model() -> str:
    return os.environ.get("PHYSICS_FIGURE_CAPTION_MODEL", DEFAULT_CAPTION_MODEL).strip()


def caption_model_fallbacks() -> list[str]:
    raw = os.environ.get("PHYSICS_FIGURE_CAPTION_MODEL_FALLBACK", ",".join(FALLBACK_CAPTION_MODELS))
    out: list[str] = []
    for part in raw.split(","):
        m = part.strip()
        if m and m not in out:
            out.append(m)
    return out


def _image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _cache_key(path: Path, model: str) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"{path.stem}_{model.replace('/', '_')}_{digest}"


def captions_cache_dir(cache_root: Path) -> Path:
    return cache_root / "figure_captions"


def load_cached_caption(cache_dir: Path, key: str) -> str | None:
    path = cache_dir / f"{key}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    caption = data.get("caption")
    return caption if isinstance(caption, str) and caption.strip() else None


def save_cached_caption(cache_dir: Path, key: str, caption: str, *, model: str, filename: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.json"
    path.write_text(
        json.dumps(
            {"filename": filename, "caption": caption, "model": model},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _provider_ready() -> bool:
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if provider in {"local", "ollama"}:
        return True
    if provider == "netra":
        return bool(os.environ.get("NETRA_API_KEY", "").strip())
    return bool(os.environ.get("OPENROUTER_API_KEY", "").strip())


def caption_image(
    path: Path,
    *,
    model: str | None = None,
    timeout_s: float | None = None,
) -> FigureCaption:
    """Caption one image. Image-only prompt (no problem text)."""
    filename = path.name
    if not path.is_file():
        return FigureCaption(filename=filename, caption="", model="", status="error")
    if not _provider_ready():
        return FigureCaption(filename=filename, caption="", model="", status="skipped")

    models = [model] if model else [caption_model(), *caption_model_fallbacks()]
    # dedupe
    seen: set[str] = set()
    chain: list[str] = []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            chain.append(m)

    timeout = timeout_s if timeout_s is not None else float(
        os.environ.get("PHYSICS_FIGURE_CAPTION_TIMEOUT_S", "90")
    )
    last_err = ""
    for mid in chain:
        try:
            client = get_client(timeout_s=timeout)
            response = client.chat.completions.create(
                model=mid,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(path)}},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=int(os.environ.get("PHYSICS_FIGURE_CAPTION_MAX_TOKENS", "500")),
            )
            content = ""
            if response.choices:
                msg = response.choices[0].message
                content = (getattr(msg, "content", None) or "").strip()
            if content:
                return FigureCaption(filename=filename, caption=content, model=mid, status="ok")
            last_err = "empty"
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if "429" in last_err.lower() or "rate" in last_err.lower():
                time.sleep(2)
            continue
    status = "empty" if last_err == "empty" else "error"
    return FigureCaption(filename=filename, caption="", model=chain[0] if chain else "", status=status)


def caption_images(
    paths: list[Path],
    *,
    cache_root: Path | None = None,
    sleep_s: float | None = None,
    workers: int | None = None,
) -> list[FigureCaption]:
    """Caption many images. Uses a thread pool for wall-clock speed on free VL."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_dir = captions_cache_dir(cache_root) if cache_root is not None else None
    if sleep_s is None:
        sleep_s = float(os.environ.get("PHYSICS_FIGURE_CAPTION_SLEEP_S", "0.2"))
    if workers is None:
        workers = int(os.environ.get("PHYSICS_FIGURE_CAPTION_WORKERS", "4"))

    pending: list[Path] = []
    results_by_name: dict[str, FigureCaption] = {}
    model = caption_model()

    for path in paths:
        key = _cache_key(path, model)
        if cache_dir is not None:
            cached = load_cached_caption(cache_dir, key)
            if cached:
                results_by_name[path.name] = FigureCaption(
                    filename=path.name, caption=cached, model=model, status="ok"
                )
                continue
        pending.append(path)

    def _one(path: Path) -> FigureCaption:
        result = caption_image(path)
        if result.status == "ok" and cache_dir is not None:
            save_cached_caption(
                cache_dir,
                _cache_key(path, result.model or model),
                result.caption,
                model=result.model,
                filename=result.filename,
            )
        if sleep_s > 0:
            time.sleep(sleep_s)
        return result

    if pending:
        workers = max(1, min(workers, len(pending)))
        if workers == 1:
            for path in pending:
                results_by_name[path.name] = _one(path)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_one, path): path for path in pending}
                for fut in as_completed(futures):
                    path = futures[fut]
                    try:
                        results_by_name[path.name] = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        results_by_name[path.name] = FigureCaption(
                            filename=path.name, caption="", model="", status="error"
                        )

    return [results_by_name[p.name] for p in paths if p.name in results_by_name]


def write_slug_captions(parsed_dir: Path, slug: str, captions: list[FigureCaption]) -> Path:
    """Persist captions for a bronze slug under parsed/figure_captions/."""
    out_dir = parsed_dir / "figure_captions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for cap in captions:
            fh.write(
                json.dumps(
                    {
                        "filename": cap.filename,
                        "caption": cap.caption,
                        "model": cap.model,
                        "status": cap.status,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def load_slug_captions(parsed_dir: Path, slug: str) -> dict[str, str]:
    path = parsed_dir / "figure_captions" / f"{slug}.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if data.get("status") == "ok" and data.get("caption"):
            out[str(data["filename"])] = str(data["caption"])
    return out
