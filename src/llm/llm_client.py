from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.repair.repair_log import LogFn

DEFAULT_BASE_URL = "https://api.netraruntime.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:11434/v1"
# General OpenRouter default (batch jobs, relations). Tutor uses faster models below.
DEFAULT_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
# AI tutor: Gemma is more reliable for short Socratic replies; Nemotron fallbacks when limited.
DEFAULT_TUTOR_OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
FALLBACK_TUTOR_OPENROUTER_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free,nvidia/nemotron-3-ultra-550b-a55b:free"
# Document translation (ingest pipeline): strongest free model primary, with fallbacks.
DEFAULT_TRANSLATE_OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
FALLBACK_TRANSLATE_OPENROUTER_MODELS = (
    "nousresearch/hermes-3-llama-3.1-405b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
)
DEFAULT_MODEL = "qwen3.6-35b"
DEFAULT_LOCAL_MODEL = "qwen2.5:3b"
DEFAULT_TRANSLATE_LOCAL_MODEL = "qwen2.5:7b-instruct"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_S = 2.0
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_REPAIR_MAX_TOKENS = 8192


@dataclass(frozen=True)
class LLMCallMetrics:
    model: str
    provider: str
    base_url: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    wall_latency_s: float
    completion_tokens_per_s: float | None
    total_tokens_per_s: float | None
    attempts: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_s, 3),
            "wall_latency_s": round(self.wall_latency_s, 3),
            "completion_tokens_per_s": (
                round(self.completion_tokens_per_s, 2)
                if self.completion_tokens_per_s is not None
                else None
            ),
            "total_tokens_per_s": (
                round(self.total_tokens_per_s, 2) if self.total_tokens_per_s is not None else None
            ),
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    metrics: LLMCallMetrics
    finish_reason: str | None = None
    truncated: bool = False


@dataclass(frozen=True)
class ChatCompletionFailure:
    reason: str
    detail: str
    metrics: LLMCallMetrics | None = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _llm_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit in {"local", "ollama"}:
        return "local"
    if explicit == "openrouter":
        return "openrouter"
    if explicit == "netra":
        return "netra"
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    if os.environ.get("LOCAL_LLM_BASE_URL", "").strip():
        return "local"
    return "netra"


def is_openrouter_free_model(model: str) -> bool:
    """OpenRouter free routes are identified by the ``:free`` suffix."""
    return model.strip().endswith(":free")


def ensure_openrouter_free_model(model: str) -> str:
    """Reject paid OpenRouter models — only ``:free`` routes are allowed.

    Used by the AI tutor model resolver. No-op for local/Netra providers.
    Raises ``ValueError`` if the active provider is OpenRouter and ``model``
    is not a free-tier id.
    """
    cleaned = model.strip()
    if _llm_provider() != "openrouter":
        return cleaned
    if not cleaned:
        raise ValueError("OpenRouter model id is empty")
    if not is_openrouter_free_model(cleaned):
        raise ValueError(
            f"OpenRouter model must be free-tier (end with :free); got {cleaned!r}"
        )
    return cleaned


def _local_base_url() -> str:
    return os.environ.get("LOCAL_LLM_BASE_URL", DEFAULT_LOCAL_BASE_URL).strip()


def get_client(*, timeout_s: float | None = None) -> OpenAI:
    if timeout_s is None:
        timeout_s = _env_float("NETRA_TIMEOUT_S", DEFAULT_TIMEOUT_S)
    provider = _llm_provider()
    if provider == "local":
        return OpenAI(
            base_url=_local_base_url(),
            api_key=os.environ.get("LOCAL_LLM_API_KEY", "ollama"),
            timeout=timeout_s,
            max_retries=0,
        )
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY environment variable is not set "
                "(set LLM_PROVIDER=openrouter in admin/server/.env)"
            )
        # OpenRouter asks for site attribution headers (optional but improves routing).
        site = os.environ.get("OPENROUTER_SITE_URL", "https://labfisika.com").strip()
        title = os.environ.get("OPENROUTER_APP_NAME", "Bank Soal Fisika AI Tutor").strip()
        return OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
            api_key=api_key,
            timeout=timeout_s,
            max_retries=0,
            default_headers={
                "HTTP-Referer": site,
                "X-Title": title,
            },
        )
    api_key = os.environ.get("NETRA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NETRA_API_KEY environment variable is not set "
            "(or set LLM_PROVIDER=openrouter / LLM_PROVIDER=local)"
        )
    return OpenAI(
        base_url=os.environ.get("NETRA_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
        timeout=timeout_s,
        max_retries=0,
    )


def provider_info() -> dict[str, str]:
    provider = _llm_provider()
    if provider == "local":
        return {"provider": "local", "base_url": _local_base_url()}
    if provider == "openrouter":
        return {
            "provider": "openrouter",
            "base_url": os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL),
        }
    return {
        "provider": "netra",
        "base_url": os.environ.get("NETRA_BASE_URL", DEFAULT_BASE_URL),
    }


def netra_provider_info() -> dict[str, str]:
    """Backward-compatible alias."""
    return provider_info()


def resolve_translate_model() -> str:
    """Pick the translation model for the ingest pipeline.

    Priority:
      1. ``INGEST_TRANSLATE_MODEL`` env override (explicit).
      2. Provider-appropriate default:
         - openrouter → strongest free model (Nemotron 3 Ultra 550B :free)
         - local      → a capable local model (qwen2.5:7b-instruct by default)
         - netra      → DEFAULT_MODEL (paid Netra endpoint)
    """
    explicit = os.environ.get("INGEST_TRANSLATE_MODEL", "").strip()
    if explicit:
        return explicit
    provider = _llm_provider()
    if provider == "openrouter":
        return DEFAULT_TRANSLATE_OPENROUTER_MODEL
    if provider == "local":
        return os.environ.get("LLM_TRANSLATE_MODEL", DEFAULT_TRANSLATE_LOCAL_MODEL)
    return DEFAULT_MODEL


def translate_models() -> list[str]:
    """Primary translate model + deduped fallbacks (for robustness)."""
    primary = resolve_translate_model()
    extras = [
        m.strip()
        for m in os.environ.get(
            "INGEST_TRANSLATE_MODEL_FALLBACK", ",".join(FALLBACK_TRANSLATE_OPENROUTER_MODELS)
        ).split(",")
        if m.strip()
    ]
    seen: set[str] = set()
    out: list[str] = []
    for model in (primary, *extras):
        if model not in seen:
            seen.add(model)
            out.append(model)
    return out


_STRAY_JSON_ESCAPE_RE = re.compile(r"(?<!\\)\\(?=[bfrt])")


def _sanitize_json_escapes(text: str) -> str:
    """Escape lone backslashes that start a LaTeX command (\\theta, \\frac,
    \\rho, \\tau, \\text, \\beta, ...) but happen to spell a *legal* JSON
    control escape (\\b \\f \\r \\t). ``json.loads`` silently accepts those and
    turns them into real backspace/formfeed/CR/tab characters, quietly
    destroying the LaTeX command instead of raising a parse error. \\n, \\",
    \\\\, \\/ and \\uXXXX are left untouched since those are almost always
    intentional (real newlines, quotes, escaped backslashes, unicode)."""
    return _STRAY_JSON_ESCAPE_RE.sub(r"\\\\", text)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return _sanitize_json_escapes(content.strip())
    for attr in ("reasoning_content", "text"):
        val = getattr(message, attr, None)
        if isinstance(val, str) and val.strip():
            return _sanitize_json_escapes(val.strip())
    return ""


def _extract_usage(
    response: Any,
    *,
    model: str,
    latency_s: float,
    wall_latency_s: float,
    attempts: int,
) -> LLMCallMetrics:
    usage = getattr(response, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    completion_tps = (
        completion_tokens / latency_s if latency_s > 0 and completion_tokens > 0 else None
    )
    total_tps = total_tokens / latency_s if latency_s > 0 and total_tokens > 0 else None
    info = provider_info()
    return LLMCallMetrics(
        model=model,
        provider=info["provider"],
        base_url=info["base_url"],
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_s=latency_s,
        wall_latency_s=wall_latency_s,
        completion_tokens_per_s=completion_tps,
        total_tokens_per_s=total_tps,
        attempts=attempts,
    )


def chat_completion_json(
    *,
    messages: list[dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_s: float = DEFAULT_RETRY_DELAY_S,
    timeout_s: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    log: LogFn | None = None,
) -> ChatCompletionResult | ChatCompletionFailure:
    """Call Netra chat API; return content + metrics or structured failure."""
    if timeout_s is None:
        timeout_s = _env_float("NETRA_TIMEOUT_S", DEFAULT_TIMEOUT_S)
    if max_tokens is None:
        max_tokens = _env_int("NETRA_MAX_TOKENS", DEFAULT_MAX_TOKENS)

    client = get_client(timeout_s=timeout_s)
    provider = _llm_provider()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider == "netra":
        kwargs["top_p"] = 1
        kwargs["extra_body"] = {
            "top_k": 40,
            "min_p": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    elif provider == "openrouter" and reasoning_effort:
        kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
    wall_start = time.perf_counter()
    attempts = 0
    last_error: Exception | None = None
    last_metrics: LLMCallMetrics | None = None

    for attempt in range(max_retries):
        attempts += 1
        if log:
            label = "Ollama" if provider == "local" else "Netra"
            log(
                f"  → {label} API attempt {attempt + 1}/{max_retries} "
                f"(timeout={timeout_s:.0f}s, max_tokens={max_tokens})"
            )
        try:
            call_start = time.perf_counter()
            used_json_mode = True
            try:
                response = client.chat.completions.create(
                    **kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as json_exc:
                if getattr(json_exc, "status_code", None) not in {400, 422}:
                    raise
                used_json_mode = False
                if log:
                    log(f"  → json_object mode unavailable ({json_exc.__class__.__name__}), retrying plain")
                response = client.chat.completions.create(**kwargs)
            latency_s = time.perf_counter() - call_start

            if not response.choices:
                last_error = RuntimeError("LLM returned no choices")
            else:
                finish_reason = getattr(response.choices[0], "finish_reason", None)
                text = _message_text(response.choices[0].message)
                last_metrics = _extract_usage(
                    response,
                    model=model,
                    latency_s=latency_s,
                    wall_latency_s=time.perf_counter() - wall_start,
                    attempts=attempts,
                )
                if text:
                    truncated = finish_reason == "length"
                    if truncated and log:
                        log(
                            f"  ⚠ response truncated at max_tokens={max_tokens} "
                            f"({last_metrics.completion_tokens} completion tokens)"
                        )
                    if log:
                        label = "Ollama" if provider == "local" else "Netra"
                        log(
                            f"  ← {label} responded in {latency_s:.1f}s "
                            f"({last_metrics.completion_tokens:,} completion tok, "
                            f"mode={'json' if used_json_mode else 'plain'})"
                        )
                    return ChatCompletionResult(
                        content=text,
                        metrics=last_metrics,
                        finish_reason=finish_reason,
                        truncated=truncated,
                    )
                last_error = RuntimeError("LLM returned empty response")
        except Exception as exc:
            last_error = exc
            if log:
                log(f"  ✗ API error: {exc.__class__.__name__}: {exc}")

        if attempt + 1 < max_retries:
            delay = retry_delay_s * (attempt + 1)
            body = getattr(last_error, "body", None)
            headers = body.get("metadata", {}).get("headers", {}) if isinstance(body, dict) else {}
            reset_ms = headers.get("X-RateLimit-Reset")
            retry_after = headers.get("Retry-After")
            if getattr(last_error, "status_code", None) == 429:
                if reset_ms:
                    delay = min(60.0, max(delay, float(reset_ms) / 1000 - time.time()))
                if retry_after:
                    delay = min(60.0, max(delay, float(retry_after)))
            if log:
                log(f"  … retrying in {delay:.0f}s")
            time.sleep(delay)

    detail = f"{last_error.__class__.__name__}: {last_error}" if last_error else "unknown error"
    return ChatCompletionFailure(reason="api_error", detail=detail, metrics=last_metrics)


def format_metrics_line(metrics: LLMCallMetrics | None, *, cached: bool = False) -> str:
    if cached:
        return "cached | 0 tok | 0.0s"
    if metrics is None:
        return "no metrics"
    parts = [
        f"{metrics.completion_tokens_per_s:.1f} gen tok/s"
        if metrics.completion_tokens_per_s is not None
        else "n/a gen tok/s",
        f"{metrics.total_tokens:,} tok",
        f"{metrics.latency_s:.1f}s",
    ]
    if metrics.attempts > 1:
        parts.append(f"{metrics.attempts} attempts")
    return " | ".join(parts)
