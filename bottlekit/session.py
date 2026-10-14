"""BottleSession: one stream object over Bottler.run / Bottler.pour (no change to either).

    s = BottleSession(client, spec, cfg, state_path="stream.json")      # min_batch='auto', votes='auto'
    h = await s.submit(texts)            # a Submission handle; certified once its coalesced batch is flushed
    await s.flush(force=True)            # certify whatever is still pending (end of stream / latency bound)
    h.answers, h.provenance, h.certificate   # the certificate is the certified BATCH's, not the submission's
    s.report()                           # per-batch rows + stream totals (cost, expected errors, sum of delta)
    s = BottleSession.resume("stream.json", client, spec, cfg)   # after a crash: exact state, then `await s.flush()`

What it adds on top of the harness:
  1. COALESCING. Submissions go to a pending buffer. When the buffer holds >= min_batch items, the WHOLE buffer is
     certified as one batch: run() while there is no bottle, else pour(bottle, votes=votes). min_batch='auto' is the
     smallest N whose planned single-look calibration share n(N)/N is <= `tax` (0.30), a function of (eps, delta,
     design_c, tune, method) only (`min_certified_size`; 355 at eps=.05, 190 at .1, 875 at .02). Batch composition
     and flush times depend only on submission sizes, min_batch and explicit flush calls: never on a label, score,
     certificate or forecast. A submission is never split across certified batches. min_batch=0 certifies every
     submission on its own (exactly the manual run/pour chain). THE CERTIFIED UNIT IS THE FLUSHED BATCH: a
     100-item submission inside a 400-item batch has no guarantee of its own (its own share can be higher).
     GROWING FLUSHES (batch_growth=g, opt-in): the flush target grows with the bottle's PLANNED size,
     T = min(max_batch, max(min_batch, ceil(g * M_plan))), M_plan = |initial artifact| + the sizes of all committed
     batches (never the bottle's actual memory, which unusable expert answers and dedupe shrink: label-dependent).
     g=0 (default) gives fixed min_batch flushes; min_batch=0 ignores g; max_batch='auto' = ceil((design_c/s)^2/eps) with
     s = max_batch_share (default 0.10; 3,125 at eps .05) caps the wait. The schedule is still a function of submission sizes and earlier batch SIZES only, so
     each batch is an ordinary run()/pour() and the stream ledger's condition holds. Bigger flushes pay a smaller
     calibration share but mean longer waits; the certified unit stays the flushed batch.
  2. LINEAGE GUARD. Every calibration stream actually drawn (certificate['cal_stream'], and per-chunk keys under
     certificate['chunks'] if a chunked pour returns them) is recorded. Before each batch the session predicts the key
     (run: (seed, 1); pour: (seed, 1, version+1, |memory|)) and raises ValueError BEFORE ANY LLM CALL if that key (or
     its derived seed) was already used: a swapped or stale artifact, a stale checkpoint, a reused journal. If the
     prediction and the key the harness actually used disagree, the session raises (a bug, never ignored).
     A version-0 artifact= (hand-built, or from a run outside this session) gets a deterministic, persisted seed bump
     with a warning (the README's recipe), recorded in the state and never repeated on resume.
  3. CRASH-SAFE CHECKPOINTS (state_path). The state is ONE JSON file of plain types (pending buffer with exact emb
     bytes, submissions, per-batch answers/certificates/ledgers, used keys, cfg hash, expert fingerprint, seed state,
     the bottle as BottleArtifact.save bytes; no pickled sklearn or client object), written to a temp file in the same
     directory and os.replace'd. It is written at construction, after each submit, when a batch is reserved and after
     each certified batch. An append-only lineage journal (<state>.lineage.jsonl) records reserved and committed keys
     outside the state, so a stale state file copied back is refused. A batch interrupted by a crash is re-certified
     with EXACTLY the reserved composition and the same calibration key (identical draw: no certificate shopping);
     a checkpointed batch is never re-run.
  4. REPORT. One row per certified batch (N, kind, submissions, bottled share, expert calls, USD by ledger phase,
     measured expert-all USD, cost_frac, eps_certified, the worst-case expected-error bound, delta) and stream totals.
     By linearity: E[#answers != expert among bottled answers] <= sum of the per-batch worst-case bounds, and
     E[#failing batches] <= sum of delta. NO statement about the probability that no batch fails is made (that would
     need a union bound: run each batch at delta/m).
  5. STREAM LEDGER (stream_ledger=True, report-only). After each certified batch its label-free design (n, N, k) per
     certified population (harness.stream_design: one per chunk of a chunked pour; every pre-specified look of a staged
     design) is added to a certify.StreamLaw: the exact convolution of the per-batch worst-case laws. report() then
     states totals['stream_risk']: P[share of ALL final answers != expert > eps' + F/N] <= tail(eps') for every eps'
     at once, eps_at_delta, delta_stream = P[stream share > eps + F/N]. By the batch_tail proof applied across batches
     (conditional stochastic dominance; fresh calibration draw per batch), valid when the flush sizes and the
     reporting horizon are fixed independently of the outputs (a fixed-horizon claim at each read, not a time-uniform
     one). A batch without a label-free design (memo-restricted, budget-capped, chunk plan mismatch, or with
     chunking='auto'/'cold' after an output without a usable answer changed the bottle's memory size, or a pour that
     passed the 'cold' gate under votes='auto', whose split followed the measured forecast) withdraws the stream
     claim (status 'none'): dropping it would let a label-dependent event pick the covered batches. Nothing feeds back
     into any k, n, threshold, votes or answer; only the designs are checkpointed and the law is rebuilt on resume.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import hashlib
import json
import math
import numbers
import os
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .certify import (DEFAULT_MIN_TAX, NONE, StreamLaw, min_certified_size,  # noqa: F401  (re-exported: session API)
                      single_look_design, spend, tuned_looks)
from .harness import (EXPERT_PHASE_PREFIX, BottleArtifact, Bottler, _plain, cal_stream_key, check_emb_values,
                      format_learning_curve, learning_curve, stream_design, stream_report,
                      stream_sentence)
from .labeling import RESEED_TAG, sub_seed
from .tasks import spec_to_plain

STATE_FORMAT = "bottlekit.session"
STATE_VERSION = 1
DEFAULT_TAX = DEFAULT_MIN_TAX   # min_batch='auto': smallest N with planned calibration share n(N)/N <= this
# RESEED_TAG (0x5E55): sub_seed key of the version-0 reseed, from labeling's stream registry (1 cal, 2 cross-fit,
# 7 forecast, 11 contexts, 13 partition)
VOTES = ("auto", True, False)
COST_FRAC_BASIS = "estimate: spent USD / (measured expert USD per call x items)"
DEFAULT_MAX_BATCH_SHARE = 0.10   # default of BottleSession(max_batch_share=): max_batch='auto' is the size where the
                                 # untuned single look labels this share of the batch (a latency/tax trade-off)
CEIL_TOL = 1e-9                  # float tolerance of the ceil()s below (0.1 * 3000 -> 300, not 301)


# ------------------------------------------------------------------------------------------------ label-free sizing
# min_certified_size lives in certify.py (one exact helper shared with chunked pours, which use it through
# certify.min_batch_for_tax); it is imported above and re-exported under the same name.
def max_batch_auto(eps: float, design_c: float, share: float = DEFAULT_MAX_BATCH_SHARE) -> int:
    """Label-free latency cap of growing session flushes (max_batch='auto'): ceil((design_c / share)^2 / eps), the
    batch size where the untuned single look n = design_c * sqrt(N / eps) labels `share` of the batch (3,125 at
    eps .05, 1,563 at .1, 7,813 at .02 with design_c 1.25 and share .10). Beyond it the remaining calibration
    saving is small. `share` is the session argument max_batch_share."""
    if not (eps > 0 and design_c > 0 and 0 < share < 1):
        raise ValueError("max_batch_auto needs eps > 0, design_c > 0 and 0 < share < 1")
    return int(math.ceil((float(design_c) / float(share)) ** 2 / float(eps) - CEIL_TOL))


def growth_target(min_batch: int, batch_growth: float, max_batch: int | None, planned_memory: int) -> int:
    """The flush target T of a growing session: min(max_batch, max(min_batch, ceil(g * M_plan))), and exactly
    min_batch when g == 0 or min_batch == 0 (per-submission certification keeps its meaning). min_batch always wins
    over the cap. A function of (min_batch, g, max_batch, M_plan) only; M_plan is a sum of batch SIZES."""
    mb = int(min_batch)
    if mb == 0 or not batch_growth > 0:
        return mb
    grown = int(math.ceil(float(batch_growth) * int(planned_memory) - CEIL_TOL))
    if max_batch is not None:
        grown = min(grown, int(max_batch))
    return max(mb, grown)


# ------------------------------------------------------------------------------------------------ plain-type helpers
def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    if isinstance(o, bytes):
        return base64.b64encode(o).decode("ascii")
    return str(o)


def _norm(o):
    """The plain-JSON form of o (numpy -> Python, tuples -> lists, non-str keys -> str). The session keeps every
    record in this form in memory too, so a resumed session holds exactly what an uninterrupted one holds."""
    return json.loads(json.dumps(_plain(o), default=_json_default))


def _emb_to_plain(e: np.ndarray | None):
    if e is None:
        return None
    e = np.ascontiguousarray(e)
    return {"b64": base64.b64encode(e.tobytes()).decode("ascii"), "dtype": e.dtype.str, "shape": list(e.shape)}


def _emb_from_plain(d) -> np.ndarray | None:
    if d is None:
        return None
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to a temp file in the SAME directory, fsync it, then os.replace it over `path` (atomic on POSIX
    and Windows): a crash leaves either the old file or the new one, never a torn one."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


REPORT_ONLY_FIELDS = ("health", "health_level", "history_cap", "health_max_rows", "health_low_percentile")
# BottleConfig fields that change no certified output (the health report): never part of config_hash, so
# changing the health report's settings never blocks a resume


def _at_default(f, v) -> bool:
    if f.default is not dataclasses.MISSING:
        return v == f.default
    if f.default_factory is not dataclasses.MISSING:
        return v == f.default_factory()
    return False


def config_hash(cfg) -> str:
    """Hash of every BottleConfig field (a callable embedder is recorded only as present/absent; embedder_id names
    it). resume() refuses a configuration with another hash. Report-only fields (REPORT_ONLY_FIELDS) are never
    hashed. Optional fields (metadata 'since') are hashed only when they differ from their default, so a
    configuration that leaves them at their defaults keeps its hash and its checkpoints resume."""
    d = {}
    for f in dataclasses.fields(cfg):
        if f.name in REPORT_ONLY_FIELDS:
            continue
        v = getattr(cfg, f.name)
        if f.metadata.get("since") and _at_default(f, v):
            continue
        if f.name == "embedder":
            v = None if v is None else "callable"
        d[f.name] = v
    return hashlib.sha256(json.dumps(_plain(d), sort_keys=True, default=_json_default).encode()).hexdigest()[:32]


def _model_identity(client, alias) -> str | None:
    if alias is None:
        return None
    ms = client.models.get(alias)
    ident = {"alias": alias, "ids": sorted(str(x) for x in (getattr(ms, "ids", None) or [])),
             "base_url": getattr(ms, "base_url", None), "extra_body": getattr(ms, "extra_body", None),
             "system_prefix": getattr(ms, "system_prefix", None)}
    return hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:32]


def _artifact_digest(art: BottleArtifact) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([art.version, art.texts, art.labels, list(art.src)], default=str).encode())
    if art.emb is not None:
        h.update(np.ascontiguousarray(art.emb).tobytes())
    return h.hexdigest()


def cert_streams(cert: dict) -> list[tuple]:
    """Every calibration-stream key a certificate records, in order: its own 'cal_stream' and, recursively, those of
    certificate['chunks'] (a chunked pour certifies several chunks with several keys). Duplicates are kept so the
    caller can detect a repeat inside one batch."""
    if not isinstance(cert, dict):
        return []
    out = []
    chunk_keys = [k for ch in (cert.get("chunks") or []) for k in cert_streams(ch)]
    cs = cert.get("cal_stream")
    if cs is not None and len(cs) and isinstance(cs[0], (list, tuple)):
        # a chunked pour's certificate lists one key PER CHUNK on top (harness format), the same keys as its
        # 'chunks': read them once (reading both would look like a repeated stream)
        tops = [tuple(int(x) for x in k) for k in cs if k is not None]
        if not chunk_keys or tops == chunk_keys:
            return tops if not chunk_keys else chunk_keys
        return tops + chunk_keys            # inconsistent record: the caller sees the repeat and raises
    if cs is not None:
        top = tuple(int(x) for x in cs)
        out.append(top)
        if chunk_keys and chunk_keys[0] == top:      # a chunked certificate may repeat its first chunk's key on top
            chunk_keys = chunk_keys[1:]
    return out + chunk_keys


def _key_seed(key: tuple) -> int:
    """The calibration seed a key derives (what actually decides the permutation): sub_seed(seed, *rest)."""
    return int(sub_seed(int(key[0]), *[int(x) for x in key[1:]]))


def _expected_errors(cert: dict):
    """Worst-case EXPECTED number of bottled answers != expert among the items the expert did not label, from the
    certificate's exact risk profile (risk['expected_eps_bound'] x the population it was computed for). None when the
    certificate makes no expectation claim (every batch with an expert budget, binding or not; excluding only
    the binding ones would select the summed batches on a label-dependent event)."""
    if cert.get("chunks"):
        parts = [_expected_errors(c) for c in cert["chunks"]]
        return None if any(p is None for p in parts) else float(sum(parts))
    if cert.get("no_calibration_needed"):
        return float(cert.get("N_population") or 0)   # everything certified without labels: worst case all of R
    risk = cert.get("risk") or {}
    b = risk.get("expected_eps_bound")
    if b is None:
        return None
    pop = risk.get("population_N")
    if pop is None:
        pop = cert.get("N_population", cert.get("N"))
    return float(b) * float(pop)


def _cert_delta(cert: dict, default: float) -> float:
    for k in ("delta_batch", "delta"):
        if cert.get(k) is not None:
            return float(cert[k])
    return float(default)


# ------------------------------------------------------------------------------------------------ handles
@dataclass(eq=False)
class Submission:
    """Handle of one submit() call. `answers`/`provenance` are None until the coalesced batch that contains it is
    certified. `certificate` is that BATCH's certificate: the guarantee bounds the share of the batch's answers that
    differ from the expert, not this submission's own share (see `guarantee_unit`)."""
    id: int
    n: int
    batch: int | None = None
    offset: int | None = None
    _session: object = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        return self.n == 0 or self.batch is not None

    def _rec(self):
        return None if self.batch is None else self._session._batches[self.batch]

    @property
    def answers(self):
        r = self._rec()
        if r is None:
            return [] if self.n == 0 else None
        return r["answers"][self.offset:self.offset + self.n]

    @property
    def provenance(self):
        r = self._rec()
        if r is None:
            return [] if self.n == 0 else None
        return r["provenance"][self.offset:self.offset + self.n]

    @property
    def certificate(self) -> dict | None:
        r = self._rec()
        return None if r is None else r["certificate"]

    @property
    def batch_N(self) -> int | None:
        r = self._rec()
        return None if r is None else int(r["N"])

    def coverage(self) -> float | None:
        p = self.provenance
        return None if p is None else (sum(x == "bottle" for x in p) / max(1, len(p)))

    @property
    def guarantee_unit(self) -> str:
        r = self._rec()
        if self.n == 0:
            return "empty submission: nothing to certify"
        if r is None:
            return "pending: waiting to be coalesced into a certified batch (call flush(force=True) to certify now)"
        c = r["certificate"]
        return (f"certified unit = batch #{self.batch} ({r['N']} items, submissions {r['submissions']}): "
                f"P[share of the BATCH's final answers != expert > {c.get('eps_certified', c.get('eps'))}] <= "
                f"{_cert_delta(c, float('nan'))}. This submission ({self.n} items) has no guarantee of its own; its "
                "own share can be higher than the batch's.")


# ------------------------------------------------------------------------------------------------ session
class BottleSession:
    """A certified stream: coalesce submissions, certify each flushed batch with run()/pour(), guard the calibration
    lineage, checkpoint crash-safely and report stream economics and risk. See the module docstring."""

    def __init__(self, client, spec, cfg, state_path: str | Path | None = None, artifact: BottleArtifact | None = None,
                 min_batch: int | str = "auto", votes: str | bool = "auto", tax: float = DEFAULT_TAX,
                 reseed_v0: bool = True, prior_streams=None, stream_ledger: bool = True,
                 batch_growth: float = 0.0, max_batch: int | str | None = "auto",
                 max_batch_share: float = DEFAULT_MAX_BATCH_SHARE):
        if not cfg.export:
            raise ValueError("BottleSession needs cfg.export=True: each batch's bottle is the next batch's scorer")
        if votes not in VOTES or isinstance(votes, int) and not isinstance(votes, bool):
            raise ValueError(f"votes must be one of {VOTES}")
        if artifact is None and cfg.cheap is None:
            raise ValueError("a session without an initial artifact starts with run(), which needs cfg.cheap")
        if min_batch != "auto" and not (isinstance(min_batch, int) and not isinstance(min_batch, bool)
                                        and min_batch >= 0):
            raise ValueError("min_batch must be 'auto' or an int >= 0")
        self._init_common(client, spec, cfg)
        self.tax = float(tax)
        self.min_batch_spec = min_batch
        self.min_batch = (min_certified_size(float(cfg.eps), float(cfg.delta), float(cfg.design_c), bool(cfg.tune),
                                             cfg.certificate, self.tax) if min_batch == "auto" else int(min_batch))
        self.votes = votes
        self._init_growth(batch_growth, max_batch, 0 if artifact is None else len(artifact.texts), max_batch_share)
        self.stream_ledger = bool(stream_ledger)
        self._sid = uuid.uuid4().hex
        self._gen = 0
        self._next_id = 0
        self._reseed = None
        eff_seed = int(cfg.seed)
        if artifact is not None and int(artifact.version) == 0 and reseed_v0:
            eff_seed, self._reseed = self._reseed_for(cfg, artifact)
            warnings.warn(f"BottleSession: the initial artifact has version 0 (hand-built or from a run outside this "
                          f"session); its calibration lineage is unknown, so the session pours with a deterministic "
                          f"new seed {eff_seed} instead of cfg.seed={cfg.seed} (recorded in the state)", stacklevel=2)
        self._set_cfg(eff_seed)
        self.artifact = artifact
        self._art_cache = None
        for k in prior_streams or []:     # calibration streams of batches outside this session (user-supplied)
            k = tuple(int(x) for x in k)
            self._used[k] = -1
            self._used_seeds[_key_seed(k)] = k
        self.state_path = None if state_path is None else Path(state_path)
        self.lineage_path = None
        if self.state_path is not None:
            self.lineage_path = Path(str(self.state_path) + ".lineage.jsonl").resolve()
            if self.state_path.exists():
                raise ValueError(f"{self.state_path} exists: use BottleSession.resume() (or a new state_path)")
            if self.lineage_path.exists():
                raise ValueError(f"a lineage journal already exists at {self.lineage_path}: resume that session, or "
                                 "choose a new state_path (its calibration streams may already be used)")
            self._checkpoint()

    # ------------------------------------------------------------------------------------------ construction helpers
    def _init_common(self, client, spec, cfg):
        self.client, self.spec, self.cfg_user = client, spec, cfg
        self._pending: list[dict] = []           # FIFO: {"id", "texts", "emb"}
        self._subs: dict[int, Submission] = {}
        self._batches: list[dict] = []
        self._used: dict[tuple, int] = {}        # committed calibration keys -> batch index (-1: prior streams)
        self._used_seeds: dict[int, tuple] = {}
        self._reserved: dict[tuple, dict] = {}   # reserved, not committed: key -> {"batch", "digest"}
        self._inflight: dict | None = None       # the batch being certified (frozen composition)
        self._dense_inputs: bool | None = None
        self._emb_dim: int | None = None
        self._lock_obj = None
        self._lock_loop = None
        self.stream_ledger = True
        self._stream: list[dict] = []           # one compact ledger entry per certified batch (see _stream_entry)
        self._stream_law = StreamLaw()          # running law of the included designs (rebuilt on resume)
        self._stream_note: str | None = None
        self.batch_growth, self.max_batch_spec, self.max_batch, self._m_init = 0.0, None, None, 0   # growth
        self.max_batch_share = DEFAULT_MAX_BATCH_SHARE

    # ------------------------------------------------------------------------------------------ growing flushes
    def _init_growth(self, batch_growth, max_batch, m_init: int, share: float = DEFAULT_MAX_BATCH_SHARE) -> None:
        """Validate and set batch_growth / max_batch. m_init = |initial artifact| (0 without one): fixed before the
        session's first batch and persisted, never re-read from a bottle. share = max_batch_share: max_batch='auto'
        is the size where the untuned single look labels this share of the batch (max_batch_auto)."""
        if isinstance(share, bool) or not isinstance(share, numbers.Real) or not 0 < float(share) < 1:
            raise ValueError("max_batch_share must be a number in (0, 1)")
        if isinstance(batch_growth, bool) or not isinstance(batch_growth, numbers.Real) \
                or not math.isfinite(float(batch_growth)) or float(batch_growth) < 0:
            raise ValueError("batch_growth must be a finite number >= 0 (0 = fixed min_batch flushes)")
        if max_batch == "auto":
            value = max(int(self.min_batch), max_batch_auto(float(self.cfg_user.eps), float(self.cfg_user.design_c),
                                                            float(share)))
        elif max_batch is None:
            value = None
        elif isinstance(max_batch, int) and not isinstance(max_batch, bool):
            if max_batch < int(self.min_batch):
                raise ValueError(f"max_batch ({max_batch}) must be >= min_batch ({self.min_batch})")
            value = int(max_batch)
        else:
            raise ValueError("max_batch must be 'auto', None (no cap) or an int >= min_batch")
        self.batch_growth, self.max_batch_spec, self.max_batch = float(batch_growth), max_batch, value
        self.max_batch_share = float(share)
        self._m_init = int(m_init)

    @property
    def planned_memory(self) -> int:
        """M_plan = |initial artifact| + the sizes N of all committed batches. Deliberately NOT len(artifact.texts):
        the actual memory drops unusable expert answers and dedupes texts, so it depends on labels."""
        return int(self._m_init) + sum(int(r["N"]) for r in self._batches)

    def flush_target(self) -> int:
        """The label-free size the pending buffer must reach to be flushed: min_batch when batch_growth == 0,
        else min(max_batch, max(min_batch, ceil(batch_growth * planned_memory))) (see growth_target)."""
        return growth_target(self.min_batch, self.batch_growth, self.max_batch, self.planned_memory)

    def _growth_state(self) -> dict:
        return {"v": 1, "batch_growth": self.batch_growth, "max_batch": {"spec": self.max_batch_spec,
                                                                         "value": self.max_batch,
                                                                         "share": self.max_batch_share},
                "m_init": self._m_init}

    def _growth_load(self, g: dict | None) -> None:
        """Restore from a checkpoint; a checkpoint with no 'growth' field resumes as g = 0."""
        if not g:
            self.batch_growth, self.max_batch_spec, self.max_batch, self._m_init = 0.0, None, None, 0
            self.max_batch_share = DEFAULT_MAX_BATCH_SHARE
            return
        mb = g.get("max_batch") or {}
        self.batch_growth = float(g.get("batch_growth", 0.0))
        self.max_batch_spec = mb.get("spec")
        self.max_batch = None if mb.get("value") is None else int(mb["value"])     # the persisted value is used as is
        self.max_batch_share = float(mb.get("share", DEFAULT_MAX_BATCH_SHARE))
        self._m_init = int(g.get("m_init", 0))

    @staticmethod
    def _growth_fields(rec: dict) -> dict:
        """Per-batch schedule fields (new report keys): flush_target (None: not recorded) and
        items_waited = items of the batch that arrived after its first submission (the first submission's wait)."""
        subs = rec.get("subs") or []
        first = int(subs[0][2]) if subs else 0
        return {"flush_target": rec.get("flush_target"), "items_waited": int(rec["N"]) - first}

    def _growth_report(self, rows: list, totals: dict) -> None:
        """Adds the flush schedule to report(): new keys only."""
        for rec, row in zip(self._batches, rows):
            row.update(self._growth_fields(rec))
        waited = [r["items_waited"] for r in rows]
        totals.update({"batch_growth": self.batch_growth, "max_batch": self.max_batch,
                       "max_batch_share": self.max_batch_share,
                       "flush_schedule": {"m_init": self._m_init, "planned_memory": self.planned_memory,
                                          "next_flush_target": self.flush_target(),
                                          "flush_targets": [r.get("flush_target") for r in self._batches],
                                          "batch_sizes": [int(r["N"]) for r in self._batches],
                                          "max_items_waited": max(waited) if waited else 0,
                                          "basis": "label-free: submission sizes, min_batch, batch_growth, max_batch "
                                                   "and the sizes of earlier batches (planned memory)"}})

    def _set_cfg(self, seed: int):
        self.seed = int(seed)
        self.cfg = self.cfg_user if self.seed == int(self.cfg_user.seed) else dataclasses.replace(self.cfg_user,
                                                                                                    seed=self.seed)
        self.bottler = Bottler(self.client, self.spec, self.cfg)

    @staticmethod
    def _reseed_for(cfg, artifact) -> tuple[int, dict]:
        """Deterministic seed bump for a version-0 artifact: sub_seed(cfg.seed, RESEED_TAG, artifact digest), never
        equal to cfg.seed modulo 2^32 (sub_seed masks seeds to 32 bits)."""
        dig = _artifact_digest(artifact)
        d32 = int(dig[:8], 16)
        user = int(cfg.seed)
        j, new = 0, None
        while new is None or (new & 0xFFFFFFFF) == (user & 0xFFFFFFFF):
            new = int(sub_seed(user, RESEED_TAG, d32, j))
            j += 1
        return new, {"from": user, "to": new, "artifact_digest": dig, "tries": j,
                     "reason": "initial artifact= of version 0: unknown calibration lineage"}

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock_obj is None or self._lock_loop is not loop:
            self._lock_obj, self._lock_loop = asyncio.Lock(), loop
        return self._lock_obj

    # ------------------------------------------------------------------------------------------ public views
    @property
    def pending_items(self) -> int:
        return sum(len(p["texts"]) for p in self._pending)

    @property
    def pending_submissions(self) -> list[int]:
        return [p["id"] for p in self._pending]

    @property
    def submissions(self) -> list[Submission]:
        return [self._subs[i] for i in sorted(self._subs)]

    def handle(self, sid: int) -> Submission:
        return self._subs[int(sid)]

    @property
    def batches(self) -> list[dict]:
        """Per certified batch (plain types): answers, provenance, certificate, ledger, diagnostics, kind, N,
        submissions, cal streams, bottle version/memory before and after."""
        return self._batches

    @property
    def used_streams(self) -> list[tuple]:
        return sorted(k for k, b in self._used.items())

    # ------------------------------------------------------------------------------------------ submit / flush
    async def submit(self, texts, emb=None) -> Submission:
        """Queue one submission (never split across batches). If the pending buffer then holds >= flush_target()
        items (min_batch, or the batch_growth target), the whole buffer is certified as one batch before this
        returns. The handle is filled when its batch is.
        Failure semantics: an error BEFORE the checkpoint (bad input, failed write) means the submission was not
        accepted. An error raised AFTER the checkpoint (while certifying the batch) means the submission IS queued
        and durable: the exception carries its handle as `exc.submission`. Do not resubmit it; call flush() to
        retry the frozen batch (same items, same calibration key)."""
        texts = [str(t) for t in texts]
        if emb is not None:
            emb = np.asarray(emb)
            if emb.ndim != 2 or emb.shape[0] != len(texts):
                raise ValueError(f"emb has shape {emb.shape}; expected ({len(texts)}, d): one row per text")
            check_emb_values(emb)                # before queueing: the batch's bottle stores them as float16
        async with self._lock():
            if texts:
                dense = emb is not None
                if self._dense_inputs is not None and dense != self._dense_inputs:
                    raise ValueError("every submission of a session must come with emb= or none must (a coalesced "
                                     "batch needs embeddings for all of its items or for none)")
                if dense and self._emb_dim is not None and emb.shape[1] != self._emb_dim:
                    raise ValueError(f"emb width {emb.shape[1]} != {self._emb_dim} (earlier submissions)")
                self._dense_inputs = dense
                if dense:
                    self._emb_dim = int(emb.shape[1])
            sid = self._next_id
            prev_dense = (self._dense_inputs, self._emb_dim)
            self._next_id += 1
            h = Submission(sid, len(texts), _session=self)
            self._subs[sid] = h
            if texts:
                self._pending.append({"id": sid, "texts": texts, "emb": emb})
            try:
                self._checkpoint()               # the submission is durable once submit() returns
            except BaseException:
                # not durable -> not accepted: undo, so memory and the last checkpoint agree
                self._pending = [p for p in self._pending if p["id"] != sid]
                del self._subs[sid]
                self._next_id = sid
                self._dense_inputs, self._emb_dim = prev_dense
                raise
            try:
                await self._flush_locked(force=False)
            except BaseException as e:
                try:
                    e.submission = h             # queued and durable: retry with flush(), never resubmit
                except Exception:  # noqa: BLE001 - some exception types refuse new attributes
                    pass
                raise
            return h

    async def flush(self, force: bool = False) -> list[dict]:
        """Certify a batch reserved before a crash (exact same composition and calibration key), then the pending
        buffer if it holds >= flush_target() items, or any pending items if force=True. Returns the report rows of the
        batches certified by this call."""
        async with self._lock():
            return await self._flush_locked(force)

    async def _flush_locked(self, force: bool) -> list[dict]:
        rows = []
        if self._inflight is not None:
            rows.append(await self._certify(list(self._inflight["ids"]), self._inflight.get("trigger"),
                                            self._inflight.get("flush_target")))
        n = self.pending_items
        T = self.flush_target()          # min_batch or the growth target; computed after any in-flight commit
        if n > 0 and (force or n >= T):
            # 'forced': flush(force=True) below the flush target (the stream ledger states the condition such flushes
            # need)
            rows.append(await self._certify([p["id"] for p in self._pending], "auto" if n >= T else "forced", T))
        return rows

    # ------------------------------------------------------------------------------------------ certify one batch
    def _predict_key(self) -> tuple:
        # run(): (seed, 1); pour(): (seed, 1, version + 1, |memory|) -- the harness's own cal_stream_key
        return cal_stream_key(self.seed, self.artifact)

    def _batch_inputs(self, ids):
        idset = set(ids)
        subs = [p for p in self._pending if p["id"] in idset]
        if [p["id"] for p in subs] != list(ids):
            raise RuntimeError(f"batch composition {ids} is not a prefix of the pending buffer (session state corrupt)")
        texts = [t for p in subs for t in p["texts"]]
        if self._dense_inputs:
            embs = [p["emb"] for p in subs]
            emb = embs[0] if len(embs) == 1 else np.concatenate(embs, axis=0)
        else:
            emb = None
        return subs, texts, emb

    def _digest(self, t, ids, texts, emb, key) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([t, list(ids), list(key), texts, self.votes], default=str).encode())
        if emb is not None:
            e = np.ascontiguousarray(emb)
            h.update(e.dtype.str.encode() + str(e.shape).encode() + e.tobytes())
        return h.hexdigest()

    def _check_key(self, key: tuple, t: int, digest: str):
        """The lineage guard: refuse (ValueError, before any LLM call) a calibration key, or a key deriving the same
        seed, that a committed batch already used, or that is reserved for another batch composition."""
        s = _key_seed(key)
        if key in self._used or s in self._used_seeds:
            b = self._used.get(key, self._used.get(self._used_seeds.get(s), None))
            src = "a batch outside this session (prior_streams)" if b == -1 else f"certified batch #{b}"
            raise ValueError(f"lineage guard: calibration stream {list(key)} was already used by {src}. The bottle "
                             "(version, memory size) does not match this session's lineage: a stale checkpoint or a "
                             "swapped artifact. Re-certifying on a used stream would calibrate on memorized expert "
                             "answers (the certificate would not be valid); nothing was called")
        r = self._reserved.get(key)
        if r is None:
            r = next((v for k, v in self._reserved.items() if _key_seed(k) == s), None)
        if r is not None and (r["batch"] != t or r["digest"] != digest):
            raise ValueError(f"lineage guard: calibration stream {list(key)} is reserved for batch #{r['batch']} with "
                             "another composition (a crash retry must certify exactly the reserved items); nothing "
                             "was called")

    async def _certify(self, ids, trigger: str | None = None, target: int | None = None) -> dict:
        t = len(self._batches)
        subs, texts, emb = self._batch_inputs(ids)
        key = self._predict_key()
        digest = self._digest(t, ids, texts, emb, key)
        self._check_key(key, t, digest)
        if self._inflight is None:
            # reserve: freeze this composition with this key (state first, then the journal), before any LLM call
            self._inflight = {"batch": t, "ids": list(ids), "key": list(key), "digest": digest, "trigger": trigger,
                              "flush_target": None if target is None else int(target)}
            self._reserved[key] = {"batch": t, "digest": digest}
            try:
                self._checkpoint()
                self._journal({"ev": "reserve", "batch": t, "key": list(key), "s_cal": _key_seed(key),
                               "digest": digest})
            except BaseException:
                self._inflight = None
                self._reserved.pop(key, None)
                raise
        elif self._inflight["digest"] != digest or tuple(self._inflight["key"]) != key:
            raise RuntimeError("in-flight batch does not match the reserved composition (session state corrupt)")
        art_in = self.artifact
        if art_in is None:
            res = await self.bottler.run(texts, emb=emb)
        else:
            res = await self.bottler.pour(art_in, texts, votes=self.votes, emb=emb)
        cert = res.certificate
        actual = cert_streams(cert)
        if actual:
            if actual[0] != key:
                raise RuntimeError(f"lineage bug: predicted calibration stream {list(key)} but the harness used "
                                   f"{list(actual[0])}")
            if len(set(actual)) != len(actual) or len({_key_seed(k) for k in actual}) != len(actual):
                raise RuntimeError(f"lineage bug: one batch used a calibration stream twice: {actual}")
            for k in actual:
                if k in self._used or _key_seed(k) in self._used_seeds:
                    raise RuntimeError(f"lineage bug: the harness used an already-used stream {list(k)}")
        elif not cert.get("no_calibration_needed"):
            raise RuntimeError("lineage bug: a certified batch recorded no calibration stream")
        if res.artifact is None:
            raise RuntimeError("the batch returned no bottle (cfg.export must stay True)")
        mode = (res.diagnostics or {}).get("mode")
        kind = "run" if art_in is None else {"pour_warm": "pour_warm", "pour": "pour_cold",
                                             "pour_memo": "pour_memo"}.get(mode, str(mode))
        offs, o = [], 0
        for p in subs:
            offs.append([p["id"], o, len(p["texts"])])
            o += len(p["texts"])
        rec = _norm({"batch": t, "kind": kind, "N": len(texts), "submissions": [p["id"] for p in subs], "subs": offs,
                     "answers": res.answers, "provenance": res.provenance, "certificate": cert,
                     "ledger": res.ledger, "diagnostics": res.diagnostics,
                     "streams": [list(k) for k in actual], "seed": self.seed,
                     "version_in": None if art_in is None else int(art_in.version),
                     "memory_in": 0 if art_in is None else len(art_in.texts),
                     "version_out": int(res.artifact.version), "memory_out": len(res.artifact.texts),
                     "votes_policy": self.votes if art_in is not None else None})
        rec["flush_target"] = None if target is None else int(target)                 # label-free schedule
        sentry = self._stream_entry(rec, trigger) if self.stream_ledger else None   # report-only, never raises
        # commit (in memory, then the state file, then the journal: a crash in between leaves a consistent state)
        self._batches.append(rec)
        if sentry is not None:
            self._stream_add(sentry)
        self.artifact = res.artifact
        self._art_cache = None
        for sid, off, _n in offs:
            self._subs[sid].batch, self._subs[sid].offset = t, off
        done = set(ids)
        self._pending = [p for p in self._pending if p["id"] not in done]
        for k in actual:
            self._used[k] = t
            self._used_seeds[_key_seed(k)] = k
        self._reserved.pop(key, None)
        self._inflight = None
        self._checkpoint()
        self._journal({"ev": "commit", "batch": t, "keys": [list(k) for k in actual],
                       **({"stream": sentry} if sentry is not None else {})})
        return {**self._row(rec), **self._growth_fields(rec)}

    # ------------------------------------------------------------------------------------------ stream ledger
    @staticmethod
    def _n_none_out(rec: dict) -> int:
        return int(sum(a is None or a == NONE for a in rec.get("answers") or []))

    def _stream_entry(self, rec: dict, trigger, prior: list | None = None) -> dict:
        """Compact ledger entry of one certified batch: b (index), N, F (expert-unusable), t (trigger: 'auto', 'forced'
        or None = unknown), x (outputs without a usable answer), ok and d (designs: [[[n, k], ...], N_pop] per
        certified population) or r (why the batch has no label-free design). Report-only: a failure is recorded as
        ok=False, never raised."""
        prior = self._stream if prior is None else prior
        try:
            cert, N = rec["certificate"], int(rec["N"])
            e = {"b": int(rec["batch"]), "N": N, "F": int(cert.get("n_expert_unusable") or 0), "t": trigger,
                 "x": self._n_none_out(rec)}
            sd = stream_design(cert, N)
            ch = (rec.get("diagnostics") or {}).get("chunking") or {}
            # chunking='cold': a smaller memory only makes the N >= ratio x memory gate easier to pass, so a pour that
            # FAILED the gate would have failed with the larger memory too; only a pour that passed it can depend on
            # outputs without a usable answer
            gated = self.cfg.chunking == "auto" or (self.cfg.chunking == "cold"
                                                    and ch.get("decision") in ("warm", "cold"))
            if (sd.get("ok") and gated and rec.get("kind") != "run"
                    and any(int(p.get("x") or 0) for p in prior)):
                sd = {"ok": False, "reason": f"chunking='{self.cfg.chunking}': this pour's chunk plan read a bottle "
                                             "memory size that earlier outputs without a usable answer had reduced "
                                             "(label-dependent)"}
            if (sd.get("ok") and ch.get("rule") == "cold" and ch.get("decision") in ("warm", "cold")
                    and self.votes == "auto"):
                # the pour passed the 'cold' gate, so whether it was split followed the
                # votes='auto' decision, which reads the measured forecast of earlier batches: the design is
                # label-dependent, which the convolution proof does not allow
                sd = {"ok": False, "reason": "chunking='cold' with votes='auto': this pour passed the cold gate "
                                             "(N >= ratio x memory), so whether it was split followed the measured "
                                             "warm/cold forecast of earlier batches (label-dependent design)"}
            if sd.get("ok"):
                e.update({"ok": True, "d": sd["designs"]})
            else:
                e.update({"ok": False, "r": str(sd.get("reason"))})
            return e
        except Exception as ex:  # noqa: BLE001 - the ledger is report-only
            return {"b": int(rec.get("batch", -1)), "N": int(rec.get("N", 0)), "F": 0, "t": trigger, "x": 0,
                    "ok": False, "r": f"ledger error: {type(ex).__name__}: {ex}"[:300]}

    def _stream_add(self, e: dict) -> None:
        """Append an entry and, while every entry so far is ok, convolve its designs into the running law."""
        self._stream.append(e)
        if not e.get("ok") or not all(p.get("ok") for p in self._stream[:-1]):
            return
        try:
            for looks, Nj in e["d"]:
                self._stream_law.add_design(looks, Nj)
        except Exception as ex:  # noqa: BLE001
            e.update({"ok": False, "r": f"ledger error: {type(ex).__name__}: {ex}"[:300]})
            e.pop("d", None)

    def _stream_rebuild(self, entries: list) -> None:
        self._stream, self._stream_law = [], StreamLaw()
        for e in entries:
            self._stream_add(dict(e))

    def stream_risk(self) -> dict:
        """totals['stream_risk'] of report() (see the module docstring, item 5)."""
        if not self.stream_ledger:
            return {"status": "off", "reason": "stream_ledger=False"}
        cfg = self.cfg
        if len(self._stream) != len(self._batches):
            return {"status": "none", "reason": f"ledger holds {len(self._stream)} entries for {len(self._batches)} "
                                                "batches", "batches": len(self._batches)}
        entries = [{"batch": e["b"], "N": e["N"], "F": e["F"], "ok": bool(e.get("ok")),
                    **({"designs": e["d"]} if e.get("ok") else {"reason": e.get("r")})} for e in self._stream]
        forced = [e["b"] for e in self._stream if e.get("t") == "forced"]
        unknown = [e["b"] for e in self._stream if e.get("t") not in ("auto", "forced")]
        notes = []
        if forced:
            below = ("their flush target (min_batch, or the batch_growth target)"
                     if self.batch_growth > 0 and self.min_batch > 0 else "min_batch")    # g = 0: plain wording
            notes.append(f"{len(forced)} batch(es) {forced[:10]} were certified by flush(force=True) below {below}: "
                         "the claim needs those flush times to have been fixed in advance (not chosen after seeing "
                         "outputs)")
        if unknown:
            notes.append(f"{len(unknown)} batch(es) have no recorded flush trigger (resumed from a checkpoint "
                         "without a stream ledger, or a crash retry of one)")
        if self._stream_note:
            notes.append(self._stream_note)
        law = self._stream_law if all(e["ok"] for e in entries) else None
        wp = []
        if cfg.expert_budget is not None:
            wp.append("expert_budget is set")
        if cfg.memo:
            wp.append("memo=True")
        if cfg.chunking in ("auto", "cold"):
            wp.append(f"chunking='{cfg.chunking}'" + (" with votes='auto'" if cfg.chunking == "cold"
                                                      and self.votes == "auto" else ""))
        sr = stream_report(entries, float(cfg.eps), float(cfg.delta), law=law, notes=notes, withdrawal_possible=wp)
        sr["forced_flushes"] = len(forced)
        return sr

    # ------------------------------------------------------------------------------------------ report
    @staticmethod
    def _expert_usd_calls(led: dict) -> tuple[float, int]:
        ex = [v for k, v in (led or {}).items() if k.startswith(EXPERT_PHASE_PREFIX)]
        return float(sum(float(v.get("usd", 0.0)) for v in ex)), int(sum(int(v.get("calls", 0)) for v in ex))

    def _session_expert_unit(self) -> float | None:
        usd = calls = 0
        for r in self._batches:
            u, c = self._expert_usd_calls(r["ledger"])
            usd, calls = usd + u, calls + c
        return usd / calls if calls else None

    def _row(self, rec: dict, unit: float | None = None) -> dict:
        cert, led, N = rec["certificate"], rec["ledger"] or {}, int(rec["N"])
        unit = self._session_expert_unit() if unit is None else unit
        usd_phase = {k: float(v.get("usd", 0.0)) for k, v in led.items()}
        usd = float(sum(usd_phase.values()))
        e_usd, e_calls = self._expert_usd_calls(led)
        epc = e_usd / e_calls if e_calls else unit
        ex_all = None if epc is None else epc * N
        prov = rec["provenance"]
        ee = _expected_errors(cert)
        nb = int(sum(p == "bottle" for p in prov))
        return {"batch": rec["batch"], "kind": rec["kind"], "N": N, "submissions": list(rec["submissions"]),
                "bottled_items": nb, "bottled": nb / max(1, N),
                "memo_share": sum(p == "memo" for p in prov) / max(1, N),
                "expert_calls": e_calls, "usd_by_phase": usd_phase, "usd": usd,
                "expert_usd_per_call": epc, "expert_all_usd": ex_all,
                "cost_frac": (usd / ex_all) if ex_all else None, "cost_frac_basis": COST_FRAC_BASIS,
                "certified": bool(cert.get("certified", False)), "n_cal": cert.get("n_cal"),
                "eps": cert.get("eps"), "eps_certified": cert.get("eps_certified"),
                "delta": _cert_delta(cert, self.cfg.delta), "expected_errors_bound": ee,
                "expected_eps_bound": None if ee is None else ee / max(1, N),
                "n_expert_unusable": int(cert.get("n_expert_unusable") or 0),
                "cal_streams": rec["streams"], "version_out": rec["version_out"], "memory_out": rec["memory_out"],
                "votes_basis": ((rec.get("diagnostics") or {}).get("votes_decision") or {}).get("basis")}

    def report(self) -> dict:
        """Per-batch rows and stream totals. Statements are per certified batch, summed by linearity of expectation:
        E[#bottled answers != expert] <= sum of the per-batch worst-case bounds, E[#failing batches] <= sum of
        delta. No claim about the probability that no batch fails.
        cost_frac is an ESTIMATE: spent USD / (measured expert USD per call x items). The expert is called on the
        harder (often longer) items, so the per-call price can differ from the price of asking the expert about
        every item; in a stream replay the estimate was on average 4.5 points above the exact cost."""
        unit = self._session_expert_unit()
        rows = [self._row(r, unit) for r in self._batches]
        N = sum(r["N"] for r in rows)
        usd = sum(r["usd"] for r in rows)
        by_phase: dict = {}
        for r in rows:
            for k, v in r["usd_by_phase"].items():
                by_phase[k] = by_phase.get(k, 0.0) + v
        ex_all = None if unit is None else unit * N
        ees = [r["expected_errors_bound"] for r in rows]
        known = [e for e in ees if e is not None]
        F = sum(r["n_expert_unusable"] for r in rows)
        bottled = sum(r["bottled_items"] for r in rows)
        totals = {"batches": len(rows), "items": N, "submissions": len(self._subs),
                  "pending_items": self.pending_items, "pending_submissions": len(self._pending),
                  "bottled_items": int(bottled), "bottled_share": bottled / N if N else None,
                  "expert_calls": sum(r["expert_calls"] for r in rows), "usd": usd, "usd_by_phase": by_phase,
                  "expert_usd_per_call": unit, "expert_all_usd": ex_all,
                  "cost_frac": (usd / ex_all) if ex_all else None, "cost_frac_basis": COST_FRAC_BASIS,
                  "expected_errors_bound": float(sum(known)) if known else 0.0,
                  "expected_share_bound": (float(sum(known)) / N) if N else None,
                  "batches_without_expected_bound": sum(e is None for e in ees),
                  "n_expert_unusable": int(F),
                  "expected_failing_batches_bound": float(sum(r["delta"] for r in rows)),
                  "min_batch": self.min_batch, "votes": self.votes, "seed": self.seed,
                  "reseeded": self._reseed is not None}
        self._growth_report(rows, totals)       # flush_target / items_waited per row, the schedule in totals
        sr = self.stream_risk()
        totals["stream_risk"] = sr
        nb = totals["batches_without_expected_bound"]
        if self.cfg.expert_budget is not None and nb:
            # with an expert budget no batch makes an expectation claim, binding or not (label-free exclusion)
            exp_part = (f"of expectation over the {len(rows)} batches: no expected-error bound is claimed for the {nb} "
                        "batch(es) under the expert budget (with a budget no batch claims one, binding or not; each "
                        "batch's certificate['risk']['eps_at_delta'][d] holds at level d)"
                        + (f"; E[#bottled answers != expert] <= {totals['expected_errors_bound']:.2f} over the "
                           f"{len(rows) - nb} other batch(es)" if len(rows) - nb else "")
                        + f"; plus {F} observed expert-unusable answers")
        else:
            exp_part = (f"of expectation over the {len(rows)} batches: E[#bottled answers != expert] <= "
                        f"{totals['expected_errors_bound']:.2f} (sum of the per-batch exact worst-case bounds"
                        + (f"; {nb} budget-capped batch(es) make no expectation claim and are excluded" if nb else "")
                        + f"), plus {F} observed expert-unusable answers")
        statement = (
            "Certified unit = each flushed batch (submissions are coalesced; a submission has no guarantee of its own). "
            "Each batch b: P[share of batch b's final answers != expert > eps_certified_b] <= delta_b. By linearity "
            + exp_part + f"; E[#failing batches] <= {totals['expected_failing_batches_bound']:.2f} "
            "(sum of delta). No statement is made about the probability that no batch fails (that needs a union "
            "bound: certify each batch at delta/m).")
        extra = stream_sentence(sr) if sr.get("status") in ("exact", "none") else ""
        if extra:
            statement += " " + extra
        return {"rows": rows, "totals": totals, "statement": statement, "health": self.health_report()}

    # ------------------------------------------------------------------------------------------ health (report-only)
    def learning_curve(self) -> list[dict]:
        """The current bottle's learning curve (BottleArtifact.history: one row per certified batch of its lineage,
        including batches certified before this session); [] before the first batch."""
        return learning_curve(self.artifact) if self.artifact is not None else []

    def health_report(self) -> dict:
        """report()['health'] (report-only, never used by the session): the learning-curve rows and table of the
        current bottle, and one compact health row per certified batch of this session (agreement with the expert on
        the calibration sample with its interval, the bottled-set precision lower bound, label-free drift)."""
        try:
            batches = []
            for rec in self._batches:
                h = (rec.get("diagnostics") or {}).get("health") or {}
                ag = h.get("agree") or {}

                def pick(k, ag=ag):
                    e = ag.get(k) or {}
                    return {"rate": e.get("rate"), "ci": e.get("ci"), "ci_finite": e.get("ci_finite")} if e else None
                d = h.get("drift") or {}
                batches.append({"batch": rec.get("batch"), "kind": rec.get("kind"), "N": rec.get("N"),
                                "unbiased": h.get("unbiased"), "n_usable": h.get("n_usable"),
                                "agree_final": pick("final"), "agree_bottle": pick("bottle"),
                                "agree_vote": pick("vote"), "agree_student": pick("student"),
                                "precision_lower": (h.get("precision") or {}).get("precision_lower"),
                                "drift": {k: d.get(k) for k in ("conf_mean", "conf_mean_prev", "label_tv", "oov_mass",
                                                                "cos_mean")} if d and not d.get("error") else None,
                                "summary": h.get("summary"), "error": h.get("error")})
            lc = self.learning_curve()
            return {"report_only": True, "batches": batches, "learning_curve": lc,
                    "learning_curve_table": format_learning_curve(lc) if lc else ""}
        except Exception as e:  # noqa: BLE001 - report-only
            return {"report_only": True, "error": f"{type(e).__name__}: {e}"[:300]}

    # ------------------------------------------------------------------------------------------ persistence
    def _artifact_plain(self):
        if self.artifact is None:
            return None
        if self._art_cache is not None and self._art_cache[0] is self.artifact:
            return self._art_cache[1]
        tmp = (self.state_path.with_name(f"{self.state_path.name}.bottle-tmp-{os.getpid()}")
               if self.state_path is not None else None)
        if tmp is None:
            return None
        try:
            self.artifact.save(tmp)                       # the existing bottle format (plain types only)
            blob = tmp.read_bytes()
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass
        d = {"b64": base64.b64encode(blob).decode("ascii"), "sha256": hashlib.sha256(blob).hexdigest(),
             "version": int(self.artifact.version), "memory": len(self.artifact.texts)}
        self._art_cache = (self.artifact, d)
        return d

    def _state(self) -> dict:
        return {"format": STATE_FORMAT, "v": STATE_VERSION, "session": self._sid, "generation": self._gen,
                "cfg_hash": config_hash(self.cfg_user), "fingerprint": self.bottler.fingerprint(),
                "cheap_identity": _model_identity(self.client, self.cfg_user.cheap),
                "spec": spec_to_plain(self.spec),
                "seed": {"user": int(self.cfg_user.seed), "effective": self.seed, "reseed": self._reseed},
                "min_batch": {"spec": self.min_batch_spec, "value": self.min_batch, "tax": self.tax},
                "growth": self._growth_state(),
                "votes": self.votes, "next_id": self._next_id,
                "pending": [{"id": p["id"], "texts": p["texts"], "emb": _emb_to_plain(p["emb"])}
                            for p in self._pending],
                "submissions": [[s.id, s.n, s.batch, s.offset] for s in self.submissions],
                "batches": self._batches,
                "used": [[list(k), b] for k, b in self._used.items()],
                "reserved": [[list(k), v] for k, v in self._reserved.items()],
                "inflight": self._inflight, "dense_inputs": self._dense_inputs, "emb_dim": self._emb_dim,
                "artifact": self._artifact_plain(),
                "lineage_path": None if self.lineage_path is None else str(self.lineage_path),
                # stream ledger: the per-batch designs only (the law is rebuilt from them on resume)
                "stream": {"v": 1, "on": self.stream_ledger, "entries": self._stream},
                # the client's cumulative cost ledger at this checkpoint: per-batch ledgers are differences of
                # cumulative float sums, so a resumed client restarted from these rows reproduces them bit for bit
                "client_ledger": getattr(getattr(self.client, "ledger", None), "rows", None)}

    def _checkpoint(self) -> None:
        if self.state_path is None:
            return
        self._gen += 1
        try:
            _atomic_write(self.state_path, json.dumps(self._state(), default=_json_default).encode("utf-8"))
        except BaseException:
            self._gen -= 1
            raise

    def _journal(self, ev: dict) -> None:
        """Append one event to the lineage journal (append-only, fsync'd; a torn last line is ignored on read)."""
        if self.lineage_path is None:
            return
        line = json.dumps({"session": self._sid, "generation": self._gen, **ev}, default=_json_default)
        with open(self.lineage_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _read_journal(path: Path, sid: str) -> list[dict]:
        out = []
        if path is None or not Path(path).exists():
            return out
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue                       # torn last line after a crash
            if ev.get("session") == sid:
                out.append(ev)
        return out

    @classmethod
    def resume(cls, state_path, client, spec, cfg, restore_ledger: bool = True) -> "BottleSession":
        """Restore a session exactly (pending order, submissions, batches, used keys, seed state, bottle with its
        memory and forecast). Refuses a different configuration (cfg hash), expert request (fingerprint) or a stale
        checkpoint (the lineage journal records batches committed after it). A batch interrupted by a crash is
        re-certified with exactly its reserved items and calibration key by the next `await s.flush()`.
        restore_ledger: if the client's cost ledger is empty, it is set to the checkpoint's cumulative rows, so client
        totals continue the stream and per-batch USD (and the measured costs a forecast records) are bit-identical to
        an uninterrupted stream. Calls made by a crashed batch before the crash are not in the checkpoint (they were
        never recorded); with LLMClient(cache_path=...) its retry is served from the cache."""
        state_path = Path(state_path)
        st = json.loads(state_path.read_text(encoding="utf-8"))
        if st.get("format") != STATE_FORMAT or int(st.get("v", 0)) != STATE_VERSION:
            raise ValueError(f"{state_path} is not a bottlekit session state (format v{STATE_VERSION})")
        if config_hash(cfg) != st["cfg_hash"]:
            raise ValueError("resume refused: BottleConfig differs from the checkpointed session (cfg hash mismatch)")
        if cfg.expert_budget is not None:
            # budget batches certified by bottlekit <= 0.9 ran both parts at delta (no "delta_split"); mixing
            # them with delta/2-split batches would make the stream report silently inconsistent
            for rec in st.get("batches", []):
                c = rec.get("certificate") or {}
                calibrated = int(c.get("n_cal") or 0) > 0 and not c.get("no_calibration_needed")
                if calibrated and "delta_split" not in c:
                    raise ValueError("resume refused: this session has expert-budget batches certified by bottlekit "
                                     "<= 0.9 (both parts at delta, not covered; see README 'Expert budget'); "
                                     "finish it with that version or start a new session")
        s = cls.__new__(cls)
        s._init_common(client, spec, cfg)
        s._set_cfg(int(st["seed"]["effective"]))
        if s.bottler.fingerprint() != st["fingerprint"]:
            raise ValueError("resume refused: the expert request fingerprint differs (task instruction, labels, "
                             "max_chars or expert model identity changed)")
        if _model_identity(client, cfg.cheap) != st.get("cheap_identity"):
            warnings.warn("resume: the cheap model identity differs from the checkpoint; certificates stay valid but "
                          "the stream is no longer reproducible bit for bit", stacklevel=2)
        s._sid, s._gen = st["session"], int(st["generation"])
        s._reseed = st["seed"]["reseed"]
        s.tax = float(st["min_batch"]["tax"])
        s.min_batch_spec, s.min_batch = st["min_batch"]["spec"], int(st["min_batch"]["value"])
        s._growth_load(st.get("growth"))     # planned memory is recomputed from the batch sizes, never the bottle
        s.votes = st["votes"]
        s._next_id = int(st["next_id"])
        s._pending = [{"id": int(p["id"]), "texts": list(p["texts"]), "emb": _emb_from_plain(p["emb"])}
                      for p in st["pending"]]
        s._batches = st["batches"]
        for sid, n, b, off in st["submissions"]:
            s._subs[int(sid)] = Submission(int(sid), int(n), b, off, _session=s)
        for k, b in st["used"]:
            k = tuple(int(x) for x in k)
            s._used[k] = int(b)
            s._used_seeds[_key_seed(k)] = k
        for k, v in st.get("reserved") or []:
            s._reserved[tuple(int(x) for x in k)] = dict(v)
        s._inflight = st["inflight"]
        s._dense_inputs, s._emb_dim = st["dense_inputs"], st["emb_dim"]
        sl = st.get("stream")
        if sl is not None:
            s.stream_ledger = bool(sl.get("on", True))
            if s.stream_ledger:
                s._stream_rebuild(sl.get("entries") or [])
        else:   # a checkpoint written before the stream ledger: rebuild it from the batch certificates
            s.stream_ledger = True
            for rec in s._batches:
                s._stream_add(s._stream_entry(rec, None))
            s._stream_note = ("stream ledger rebuilt from the checkpointed batch certificates (checkpoint written "
                              "without a ledger)")
        s.state_path = state_path
        a = st["artifact"]
        s.artifact = None
        s._art_cache = None
        if a is not None:
            blob = base64.b64decode(a["b64"])
            if hashlib.sha256(blob).hexdigest() != a["sha256"]:
                raise ValueError("resume refused: the checkpointed bottle is corrupt (sha256 mismatch)")
            tmp = state_path.with_name(f"{state_path.name}.bottle-tmp-{os.getpid()}")
            try:
                tmp.write_bytes(blob)
                s.artifact = BottleArtifact.load(tmp)
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if int(s.artifact.version) != int(a["version"]) or len(s.artifact.texts) != int(a["memory"]):
                raise ValueError("resume refused: the checkpointed bottle does not match its record")
            s._art_cache = (s.artifact, a)
        # lineage journal: the recorded one (the session's own, even if this state file is a copy elsewhere) and the
        # one next to this state file; committed batches beyond this checkpoint mean the state is stale
        rec_path = Path(st["lineage_path"]) if st.get("lineage_path") else None
        here = Path(str(state_path) + ".lineage.jsonl").resolve()
        paths = []
        for p in (rec_path, here):
            if p is not None and p.exists() and all(p.resolve() != q.resolve() for q in paths):
                paths.append(p)
        if rec_path is not None and not rec_path.exists():
            warnings.warn(f"resume: lineage journal {rec_path} not found; stale-checkpoint detection relies on the "
                          "state file alone", stacklevel=2)
        s.lineage_path = (rec_path if rec_path is not None and rec_path.exists() else here)
        n_b = len(s._batches)
        for p in paths:
            for ev in cls._read_journal(p, s._sid):
                if ev.get("ev") == "commit":
                    if int(ev["batch"]) >= n_b:
                        raise ValueError(f"resume refused: stale checkpoint. The lineage journal {p} records batch "
                                         f"#{ev['batch']} as certified, but this state holds only {n_b} batches; "
                                         "continuing would re-certify already-certified items on used calibration "
                                         "streams")
                    for k in ev.get("keys") or []:
                        k = tuple(int(x) for x in k)
                        s._used.setdefault(k, int(ev["batch"]))
                        s._used_seeds.setdefault(_key_seed(k), k)
            committed = {int(ev["batch"]) for ev in cls._read_journal(p, s._sid) if ev.get("ev") == "commit"}
            for ev in cls._read_journal(p, s._sid):
                if ev.get("ev") == "reserve" and int(ev["batch"]) not in committed and int(ev["batch"]) >= n_b:
                    k = tuple(int(x) for x in ev["key"])
                    s._reserved.setdefault(k, {"batch": int(ev["batch"]), "digest": ev["digest"]})
        # every check passed: only now touch the caller's client (cost ledger continuity)
        led, saved = getattr(client, "ledger", None), st.get("client_ledger")
        if restore_ledger and saved and led is not None and hasattr(led, "rows"):
            if not led.rows:
                led.rows = {k: dict(v) for k, v in saved.items()}
            else:
                warnings.warn("resume: the client's cost ledger is not empty, so it was not restored; per-batch USD "
                              "then matches an uninterrupted stream only up to float rounding", stacklevel=2)
        return s
