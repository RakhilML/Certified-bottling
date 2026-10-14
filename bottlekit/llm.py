"""Provider-agnostic async LLM client for any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, OpenAI,
or a caching proxy).

Features: per-model and global concurrency limits (so callers respect server rules even without a proxy),
deterministic instance selection over model replicas (content-hashed: reproducible and cache-friendly), retries
with backoff (sleeping OUTSIDE the concurrency slots), optional on-disk response cache (keyed on the request
content, not on which replica served it; empty replies are never cached), explicit error surfacing
(ChatResult.ok / .error instead of silent empty strings or exceptions), and a cost ledger per phase
(tokens and USD from per-model prices) with snapshots for per-run accounting. `embed` calls an OpenAI-compatible
/v1/embeddings endpoint (dense student features; costed under phase 'embed').
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import yaml

DEFAULT_BASE_URL = "http://127.0.0.1:8765/v1"   # fallback when a models file sets no base_url (part of the expert
                                                # fingerprint: changing it changes the pinned test fingerprints)
DEFAULT_MAX_CONCURRENCY = 2                     # per-model concurrent requests (safe for local servers)
DEFAULT_GLOBAL_CONCURRENCY = 8                  # LLMClient: concurrent requests over all models
DEFAULT_TIMEOUT_S = 900
DEFAULT_RETRIES = 4
NON_RETRYABLE_STATUS = (400, 401, 402, 403, 404)   # HTTP statuses that are not retried
BACKOFF_CAP_S = 60.0                            # retry backoff: min(cap, 2**attempt + U(0, 1)) seconds
MAP_LIMIT = 64                                  # LLMClient.map: concurrent coroutines in flight
EMBED_BATCH_SIZE = 64                           # texts per embeddings request
EMBED_CHARS_PER_TOKEN = 4                       # embedding cost estimate when the server reports no usage
NORM_FLOOR = 1e-12                              # L2 normalisation floor of embeddings


def _backoff(attempt: int, cap: float = BACKOFF_CAP_S) -> float:
    return min(cap, 2 ** attempt + random.random())


@dataclass
class ModelSpec:
    alias: str
    ids: list[str]                       # one or more loaded instances (replicas of the same model)
    base_url: str = DEFAULT_BASE_URL
    api_key_env: str | None = None       # env var holding the key (None: no auth header)
    extra_body: dict = field(default_factory=dict)   # e.g. {"reasoning_effort": "none"}
    system_prefix: str | None = None     # e.g. "/no_think" for qwen3
    price_in: float = 0.0                # USD per 1M prompt tokens
    price_out: float = 0.0               # USD per 1M completion tokens
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_tokens: int | None = None        # single-label reply budget (reasoning models need more); None: derived
    # optional (the defaults keep the standard behaviour)
    batch_extra_tokens: int = 0          # added once to every BATCHED reply budget (a reasoning cheap model, e.g. ~300)
    expected_reply_tokens: int | None = None   # cost estimate of one single reply's extra tokens (None: 0.5*max_tokens)

    def usd(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.price_in + completion_tokens * self.price_out) / 1e6


@dataclass
class ChatResult:
    text: str
    ok: bool
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usd: float = 0.0
    cached: bool = False
    error: str | None = None


class EmbeddingError(RuntimeError):
    """An embeddings request failed after retries (Bottler falls back to TF-IDF features, flagged)."""


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults = cfg.get("defaults", {})
    out = {}
    for alias, m in cfg["models"].items():
        spec = {**defaults, **m}
        ids = spec.pop("ids", None) or [spec.pop("id")]
        spec.pop("id", None)
        out[alias] = ModelSpec(alias=alias, ids=list(ids), **spec)
    return out


class _Ledger:
    def __init__(self):
        self.rows: dict[str, dict] = {}

    def add(self, phase: str, r: ChatResult):
        d = self.rows.setdefault(phase, {"calls": 0, "cached": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0})
        d["calls"] += 1
        d["cached"] += int(r.cached)
        d["errors"] += int(not r.ok)
        d["prompt_tokens"] += r.prompt_tokens
        d["completion_tokens"] += r.completion_tokens
        d["usd"] += r.usd

    def total_usd(self) -> float:
        return sum(d["usd"] for d in self.rows.values())

    def as_dict(self) -> dict:
        return {k: dict(v) for k, v in self.rows.items()}

    def snapshot(self) -> dict:
        return copy.deepcopy(self.rows)

    def since(self, snap: dict) -> dict:
        """Ledger rows accumulated after `snap` (per-run accounting on a shared client)."""
        out = {}
        for ph, d in self.rows.items():
            s = snap.get(ph, {})
            diff = {k: v - s.get(k, 0) for k, v in d.items()}
            if diff["calls"]:
                out[ph] = diff
        return out


def _replica(spec: ModelSpec, content_key: str) -> str:
    return spec.ids[int(content_key[:8], 16) % len(spec.ids)] if len(spec.ids) > 1 else spec.ids[0]


class LLMClient:
    def __init__(self, models: dict[str, ModelSpec], global_concurrency: int = DEFAULT_GLOBAL_CONCURRENCY,
                 cache_path: str | Path | None = None, timeout: float = DEFAULT_TIMEOUT_S, retries: int = DEFAULT_RETRIES,
                 *, phase_header: str | None = "X-Exp", cache_header: str | None = "X-Cache",
                 backoff_cap: float = BACKOFF_CAP_S, non_retryable=NON_RETRYABLE_STATUS, map_limit: int = MAP_LIMIT):
        """phase_header / cache_header: optional proxy headers (the ledger phase is sent as `phase_header`; a reply
        header `cache_header: hit` marks a proxy cache hit); None disables either. backoff_cap: longest retry sleep.
        non_retryable: HTTP statuses returned at once. map_limit: default fan-out of map()."""
        self.models = models
        self.global_sem = asyncio.Semaphore(global_concurrency)
        self.model_sems = {a: asyncio.Semaphore(m.max_concurrency) for a, m in models.items()}
        self.retries = retries
        self.phase_header, self.cache_header = phase_header, cache_header
        self.backoff_cap, self.non_retryable, self.map_limit = float(backoff_cap), tuple(non_retryable), int(map_limit)
        self.http = httpx.AsyncClient(timeout=timeout)
        self.ledger = _Ledger()
        self.db = None
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(str(cache_path), check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
            self.db.commit()

    async def close(self):
        await self.http.aclose()
        if self.db:
            self.db.commit()
            self.db.close()

    def _headers(self, spec: ModelSpec, phase: str) -> dict:
        h = {"Content-Type": "application/json"}
        ph = getattr(self, "phase_header", "X-Exp")
        if ph:
            h[ph] = phase
        if spec.api_key_env and os.environ.get(spec.api_key_env):
            h["Authorization"] = f"Bearer {os.environ[spec.api_key_env]}"
        return h

    def _cache_get(self, key):
        if not self.db:
            return None
        row = self.db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key, resp):
        if self.db:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(resp)))
            self.db.commit()

    async def chat(self, alias: str, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0,
                   phase: str = "default", seed: int | None = None) -> ChatResult:
        """Request failures never raise: they come back as ChatResult(ok=False, error=...). An alias that is not in the
        models file raises KeyError (a configuration error, not a request failure)."""
        spec = self.models.get(alias)
        if spec is None:
            raise KeyError(f"model alias {alias!r} is not in the models file (known: {sorted(self.models)})")
        try:
            msgs = list(messages)
            if spec.system_prefix:
                if msgs and msgs[0]["role"] == "system":
                    msgs[0] = {"role": "system", "content": spec.system_prefix + "\n" + msgs[0]["content"]}
                else:
                    msgs = [{"role": "system", "content": spec.system_prefix}] + msgs
            content = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature, **spec.extra_body}
            if seed is not None:
                content["seed"] = seed
            key = hashlib.sha256(json.dumps([spec.base_url, sorted(spec.ids), content], sort_keys=True).encode()).hexdigest()
            model_id = _replica(spec, key)
            body = {"model": model_id, **content}
        except Exception as e:  # malformed messages etc.
            r = ChatResult(text="", ok=False, model=alias, error=f"{type(e).__name__}: {e}")
            self.ledger.add(phase, r)
            return r
        if (hit := self._cache_get(key)) is not None:
            r = self._to_result(spec, model_id, hit, cached=True)
            self.ledger.add(phase, r)
            return r
        last_err = None
        for attempt in range(self.retries):
            try:
                async with self.model_sems[alias], self.global_sem:
                    resp = await self.http.post(f"{spec.base_url}/chat/completions", json=body, headers=self._headers(spec, phase))
                if resp.status_code == 200:
                    d = resp.json()
                    r = self._to_result(spec, model_id, d, cached=self._gateway_hit(resp))
                    if r.ok and r.text:
                        self._cache_put(key, d)
                    self.ledger.add(phase, r)
                    return r
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code in getattr(self, "non_retryable", NON_RETRYABLE_STATUS):
                    break  # not retryable
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as e:
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(_backoff(attempt, getattr(self, "backoff_cap", BACKOFF_CAP_S)))   # outside the concurrency slots
        r = ChatResult(text="", ok=False, model=model_id, error=last_err)
        self.ledger.add(phase, r)
        return r

    def _gateway_hit(self, resp) -> bool:
        ch = getattr(self, "cache_header", "X-Cache")
        return bool(ch) and resp.headers.get(ch) == "hit"

    @staticmethod
    def _to_result(spec: ModelSpec, model_id: str, d, cached: bool) -> ChatResult:
        if not isinstance(d, dict):
            return ChatResult(text="", ok=False, model=model_id, error=f"non-object response: {str(d)[:120]}")
        choices = d.get("choices") or [{}]
        msg = (choices[0] if isinstance(choices[0], dict) else {}).get("message") or {}
        u = d.get("usage") or {}
        pt, ct = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
        return ChatResult(text=msg.get("content") or "", ok=True, model=model_id, prompt_tokens=pt, completion_tokens=ct,
                          usd=spec.usd(pt, ct), cached=cached)

    # ------------------------------------------------------------------------------------------------ embeddings
    async def embed(self, alias: str, texts: list[str], phase: str = "embed", batch_size: int = EMBED_BATCH_SIZE,
                    normalize: bool = True) -> np.ndarray:
        """(len(texts), d) float32 embeddings from an OpenAI-compatible /v1/embeddings endpoint, one row per text,
        L2-normalised by default (the row scale of the TF-IDF blocks they are concatenated with). Cached per text when
        a cache is configured; each request is costed in the ledger under `phase` (prompt tokens x price_in; cache hits
        are costed at an estimate, as for chat). Raises EmbeddingError if a request still fails after retries."""
        spec = self.models.get(alias)
        if spec is None:
            raise KeyError(f"model alias {alias!r} is not in the models file (known: {sorted(self.models)})")
        texts = [str(t) for t in texts]
        keys = [hashlib.sha256(json.dumps(["embed", spec.base_url, sorted(spec.ids), spec.extra_body, t],
                                          sort_keys=True).encode()).hexdigest() for t in texts]
        rows: list = [None] * len(texts)
        miss = []
        for i, k in enumerate(keys):
            hit = self._cache_get(k)
            if hit is not None:
                rows[i] = hit
            else:
                miss.append(i)
        n_hit = len(texts) - len(miss)
        if n_hit:
            est = sum(len(texts[i]) for i in range(len(texts)) if rows[i] is not None) // EMBED_CHARS_PER_TOKEN
            self.ledger.add(phase, ChatResult(text="", ok=True, model=alias, prompt_tokens=est, usd=spec.usd(est, 0),
                                              cached=True))
        chunks = [miss[a:a + batch_size] for a in range(0, len(miss), batch_size)]
        outs = await asyncio.gather(*(self._embed_request(spec, alias, [texts[i] for i in ch], phase) for ch in chunks))
        fresh = [(i, v) for ch, vecs in zip(chunks, outs) for i, v in zip(ch, vecs)]
        for i, v in fresh:
            rows[i] = v
        if self.db and fresh:   # one transaction for the whole batch
            self.db.executemany("INSERT OR REPLACE INTO cache VALUES (?,?)", [(keys[i], json.dumps(v)) for i, v in fresh])
            self.db.commit()
        if not texts:
            return np.zeros((0, 0), np.float32)
        E = np.asarray(rows, np.float32)
        if normalize:
            E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), NORM_FLOOR)
        return E

    async def _embed_request(self, spec: ModelSpec, alias: str, batch: list[str], phase: str) -> list:
        content = {"input": batch, **spec.extra_body}
        key = hashlib.sha256(json.dumps([spec.base_url, sorted(spec.ids), content], sort_keys=True).encode()).hexdigest()
        model_id = _replica(spec, key)
        body = {"model": model_id, **content}
        last_err = None
        for attempt in range(self.retries):
            try:
                async with self.model_sems[alias], self.global_sem:
                    resp = await self.http.post(f"{spec.base_url}/embeddings", json=body, headers=self._headers(spec, phase))
                if resp.status_code == 200:
                    d = resp.json()
                    data = d.get("data") if isinstance(d, dict) else None
                    if not isinstance(data, list) or len(data) != len(batch):
                        raise ValueError(f"expected {len(batch)} embeddings, got {type(data).__name__} "
                                         f"{len(data) if isinstance(data, list) else ''}")
                    data = sorted(data, key=lambda r: r.get("index", 0)) if all("index" in r for r in data) else data
                    vecs = [list(map(float, r["embedding"])) for r in data]
                    if len({len(v) for v in vecs}) != 1 or not vecs[0]:
                        raise ValueError("embeddings of unequal or zero width")
                    u = d.get("usage") or {}
                    pt = (u.get("prompt_tokens") or u.get("total_tokens")
                          or sum(len(t) for t in batch) // EMBED_CHARS_PER_TOKEN)
                    self.ledger.add(phase, ChatResult(text="", ok=True, model=model_id, prompt_tokens=pt,
                                                      usd=spec.usd(pt, 0), cached=self._gateway_hit(resp)))
                    return vecs
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code in getattr(self, "non_retryable", NON_RETRYABLE_STATUS):
                    break  # not retryable
            except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError, TypeError, AttributeError) as e:
                last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(_backoff(attempt, getattr(self, "backoff_cap", BACKOFF_CAP_S)))   # outside the concurrency slots
        self.ledger.add(phase, ChatResult(text="", ok=False, model=model_id, error=last_err))
        raise EmbeddingError(f"embeddings from {alias!r} failed: {last_err}")

    async def map(self, coros, limit: int | None = None):
        """Bounded fan-out over many coroutines (keeps memory flat for big workloads). Accepts coroutines or
        zero-argument coroutine factories (factories avoid materialising all coroutines up front).
        limit: None = self.map_limit (64 by default)."""
        sem = asyncio.Semaphore(getattr(self, "map_limit", MAP_LIMIT) if limit is None else limit)

        async def run(c):
            async with sem:
                return await (c() if callable(c) else c)
        return await asyncio.gather(*(run(c) for c in coros))
