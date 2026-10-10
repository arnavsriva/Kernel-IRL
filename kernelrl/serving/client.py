"""Async OpenAI-compatible client for the SGLang server (sampling params per Qwen3 guidance)."""
from __future__ import annotations

import asyncio
import json
import time
import urllib.request
from dataclasses import asdict, dataclass

import re

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, BadRequestError, InternalServerError

_CTX_RE = re.compile(r"maximum context length of (\d+) tokens.*?(\d+) tokens from the input", re.S)
MIN_COMPLETION_TOKENS = 512


@dataclass
class Completion:
    content: str
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    latency_s: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Generator:
    def __init__(self, base_url: str, model: str, sampling: dict, *, thinking: bool, concurrency: int = 64,
                 timeout_s: float = 1800, retries: int = 3):
        self.client = AsyncOpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s, max_retries=0)
        self.model, self.sampling, self.thinking = model, dict(sampling), thinking
        self.sem = asyncio.Semaphore(concurrency)
        self.retries = retries

    async def complete(self, messages: list[dict], seed: int) -> Completion:
        s = self.sampling
        extra = {"top_k": s.get("top_k", 20), "min_p": s.get("min_p", 0.0), "separate_reasoning": True,
                 "chat_template_kwargs": {"enable_thinking": self.thinking}}
        max_tokens = s["max_tokens"]
        async with self.sem:
            for attempt in range(self.retries + 1):
                t0 = time.perf_counter()
                try:
                    r = await self.client.chat.completions.create(
                        model=self.model, messages=messages, temperature=s["temperature"], top_p=s["top_p"],
                        max_tokens=max_tokens, presence_penalty=s.get("presence_penalty", 0.0), seed=seed,
                        extra_body=extra)
                    ch = r.choices[0]
                    u = r.usage
                    details = getattr(u, "prompt_tokens_details", None)
                    return Completion(
                        content=ch.message.content or "",
                        reasoning=getattr(ch.message, "reasoning_content", None),
                        finish_reason=ch.finish_reason, prompt_tokens=u.prompt_tokens,
                        completion_tokens=u.completion_tokens,
                        cached_tokens=getattr(details, "cached_tokens", None) if details else None,
                        latency_s=time.perf_counter() - t0)
                except BadRequestError as e:
                    # Long turn-2 histories can exceed the context window with the configured budget.
                    # Shrink the completion budget to what fits instead of failing the whole run.
                    m = _CTX_RE.search(str(e))
                    if m and attempt < self.retries:
                        fit = int(m.group(1)) - int(m.group(2)) - 16
                        if fit >= MIN_COMPLETION_TOKENS and fit < max_tokens:
                            max_tokens = fit
                            continue
                    return Completion("", None, None, 0, 0, None, time.perf_counter() - t0,
                                      error=f"{type(e).__name__}: {str(e)[:500]}")
                except (APIConnectionError, APITimeoutError, InternalServerError) as e:
                    if attempt >= self.retries - 1:
                        return Completion("", None, None, 0, 0, None, time.perf_counter() - t0,
                                          error=f"{type(e).__name__}: {e}")
                    await asyncio.sleep(2 ** attempt)
                except APIStatusError as e:  # any other server-side rejection: record, don't crash the run
                    return Completion("", None, None, 0, 0, None, time.perf_counter() - t0,
                                      error=f"{type(e).__name__}: {str(e)[:500]}")
        raise AssertionError("unreachable")


def server_info(base_url: str) -> dict:
    """GET /server_info (sglang version + server args) for the result fingerprint."""
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/server_info", timeout=10) as r:
            info = json.loads(r.read())
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    keep = ("version", "model_path", "revision", "disable_radix_cache", "mem_fraction_static", "context_length",
            "random_seed", "reasoning_parser", "enable_metrics", "tp_size", "dtype", "max_running_requests")
    return {k: info.get(k) for k in keep}


def flush_cache(base_url: str) -> bool:
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{root}/flush_cache", method="POST"), timeout=30) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False
