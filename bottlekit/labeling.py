"""Labeling: single-item, randomized batched, warm-up, and multi-context voting.

Evidence-driven defaults (measured on development tasks):
  * random batch order, never similarity-sorted (similar neighbours cause label copying, 2-4.5x error clustering);
  * batch size ~20-25; weak labelers have a COLD START (first ~quarter of a batch is 10-15 pts worse);
  * position-balanced contexts: across the voting contexts no item sits in the cold-start zone more than once;
  * position-aware weighted voting: the reliability of each batch position is estimated WITHOUT labels from
    leave-one-out agreement with the other contexts, and votes are weighted by it. The weighted agreement is a
    continuous uncertainty score (fixes the coarseness of integer vote counts for certification).
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .tasks import TaskSpec, batch_max_tokens, batch_messages, parse_batch, parse_label, single_messages

# Random-stream registry: sub_seed keys of every random stream (VALUES ARE FIXED: changing one changes every draw).
CAL_STREAM_KEY = 1           # calibration draw: (seed, 1) for run(), (seed, 1, version + 1, |memory|) for a pour
CROSSFIT_STREAM_KEY = 2      # run()'s cross-fit folds: (seed, 2)
FORECAST_STREAM_KEY = 7      # forecast cross-fits: (seed, 7[, bottle version])
CONTEXT_STREAM_KEY = 11      # unbalanced voting orders: (seed, 11, context)
PARTITION_STREAM_KEY = 13    # chunked pour partition: (seed, 13, version + 1, |memory|)
HEALTH_STREAM_KEY = 19       # health report's drift subsample (report-only): (seed, 19, ...)
RESEED_TAG = 0x5E55          # BottleSession's version-0 reseed: (seed, 0x5E55, digest, j)
STREAM_KEYS = {"calibration": CAL_STREAM_KEY, "crossfit": CROSSFIT_STREAM_KEY, "forecast": FORECAST_STREAM_KEY,
               "contexts": CONTEXT_STREAM_KEY, "partition": PARTITION_STREAM_KEY, "health": HEALTH_STREAM_KEY,
               "reseed": RESEED_TAG}
# Raw seed offsets (not sub_seed keys; they only touch CHEAP-label streams): warm-up pool rng = seed + 7, context c's
# batched labelling seed = seed + 101*c.
WARMUP_SEED_OFFSET = 7
CONTEXT_SEED_STRIDE = 101
# Labelling constants (measured with a 4B model on agnews: the cold-start zone is the first quarter of a batch).
EARLY_FRAC = 0.25            # cold-start zone share of a batch; position-reliability bins = round(1 / early_frac)
BALANCE_REPAIR_PASSES = 3    # greedy repair passes of balanced_orders
RELIABILITY_PRIOR = 2.0      # Beta smoothing of position reliability toward the overall agreement
RELIABILITY_CLIP = (0.05, 0.995)   # reliability clip before the log-odds weight
WEIGHT_FLOOR = 0.05          # smallest vote weight
UNWEIGHTED_REL = 0.75        # weighted=False: one common reliability (inert: equal weights cancel in the vote share)
VOTE_TIE_DIGITS = 12         # weighted vote: weights are compared rounded to this many digits (ties -> counts)
SINGLE_TOKENS_MIN = 40       # single-label reply budget: ModelSpec.max_tokens or max(MIN, label tokens + PAD)
SINGLE_TOKENS_PAD = 32


def n_bins_for(early_frac: float) -> int:
    """Position-reliability bins: round(1/early_frac) (4 at the default 0.25): one bin per cold-start-zone width."""
    return max(1, int(round(1.0 / float(early_frac))))


def sub_seed(seed: int, *keys: int) -> int:
    """Independent child seed (SeedSequence) so different random streams never coincide."""
    return int(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, *[int(k) for k in keys]]).generate_state(1)[0])


@dataclass
class ContextLabels:
    labels: list[str | None]      # per item
    positions: list[int]          # per item: position inside its batch (after warm-up offset)
    k: int


def single_max_tokens(client, alias: str, spec: TaskSpec) -> int:
    m = getattr(client.models.get(alias), "max_tokens", None)
    return m or max(SINGLE_TOKENS_MIN, spec.max_label_tokens + SINGLE_TOKENS_PAD)


async def label_single(client, alias: str, spec: TaskSpec, texts: list[str], idx: list[int], phase: str = "single"):
    mt = single_max_tokens(client, alias, spec)
    res = await client.map([client.chat(alias, single_messages(spec, texts[i]), max_tokens=mt, phase=phase) for i in idx])
    return {i: parse_label(spec, r.text) for i, r in zip(idx, res)}


def random_order(n: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).permutation(n).tolist()


def balanced_orders(n: int, k: int, contexts: int, seed: int, early_frac: float = EARLY_FRAC) -> list[list[int]]:
    """Random batch orders such that each item lands in the early (cold-start) zone of at most one context.
    Feasible iff contexts * early <= k; otherwise the early zone is shrunk to k // contexts (with a warning)."""
    rng = np.random.default_rng(seed)
    e = max(1, int(round(k * early_frac)))
    if contexts * e > k:
        e_new = max(1, k // contexts)
        warnings.warn(f"balanced_orders: {contexts} contexts x early zone {e} > batch size {k}; using early zone {e_new}")
        e = e_new
    was_early = np.zeros(n, bool)
    orders = []
    for c in range(contexts):
        order = rng.permutation(n)
        pos = np.arange(n) % k
        if c > 0:
            for _ in range(BALANCE_REPAIR_PASSES):  # greedy repair: swap early-again items with late never-early ones
                bad = np.nonzero((pos < e) & was_early[order])[0]
                good = np.nonzero((pos >= e) & ~was_early[order])[0]
                rng.shuffle(good)
                for b, g in zip(bad, good):
                    order[b], order[g] = order[g], order[b]
                if len(bad) == 0:
                    break
        was_early[order[pos < e]] = True
        orders.append(order.tolist())
    return orders


async def label_batched(client, alias: str, spec: TaskSpec, texts: list[str], order: list[int], k: int,
                        warmup: int = 0, seed: int = 0, phase: str = "batched") -> ContextLabels:
    """Label items in `order`, k real items per call; optionally prefix each call with `warmup` throwaway items
    drawn from the rest of the workload (their labels are discarded)."""
    n = len(texts)
    batches = [order[i:i + k] for i in range(0, len(order), k)]
    rng = np.random.default_rng(seed + WARMUP_SEED_OFFSET)
    calls = []
    for b in batches:
        warm = []
        if warmup:
            bs = set(b)
            pool = rng.choice(n, size=min(n, warmup + len(b)), replace=False)
            warm = [int(j) for j in pool if j not in bs][:warmup]
        calls.append((b, warm))
    extra = int(getattr(client.models.get(alias), "batch_extra_tokens", 0) or 0)   # ModelSpec reasoning budget
    res = await client.map([client.chat(alias, batch_messages(spec, [texts[j] for j in w] + [texts[i] for i in b]),
                                        max_tokens=batch_max_tokens(spec, len(w) + len(b), extra), phase=phase)
                            for b, w in calls])
    labels: list[str | None] = [None] * n
    positions = [0] * n
    for (b, w), r in zip(calls, res):
        parsed = parse_batch(spec, r.text if r.ok else "", len(w) + len(b))
        for p, i in enumerate(b):
            labels[i] = parsed[len(w) + p]
            positions[i] = p
    return ContextLabels(labels, positions, k)


def _bin(pos: int, k: int, n_bins: int) -> int:
    return min(n_bins - 1, int(n_bins * pos / max(1, k)))


def position_reliability(ctxs: list[ContextLabels], n_bins: int = 4, prior: float = RELIABILITY_PRIOR) -> np.ndarray:
    """Unsupervised reliability per (position bin): leave-one-out agreement of a context's label with the plurality
    of the other contexts, pooled over contexts. Returns array [n_bins] in (0,1). Beta(prior) smoothing."""
    agree, tot = np.zeros(n_bins), np.zeros(n_bins)
    n = len(ctxs[0].labels)
    for c, ctx in enumerate(ctxs):
        for i in range(n):
            lab = ctx.labels[i]
            if lab is None:
                continue
            others = Counter(o.labels[i] for j, o in enumerate(ctxs) if j != c and o.labels[i] is not None)
            if not others:
                continue
            top_n = others.most_common(1)[0][1]
            tops = {l for l, m in others.items() if m == top_n}
            b = _bin(ctx.positions[i], ctx.k, n_bins)
            agree[b] += float(lab in tops) / len(tops)     # split credit on ties
            tot[b] += 1
    overall = agree.sum() / max(1, tot.sum())
    return (agree + prior * overall) / (tot + prior)


def weighted_vote(ctxs: list[ContextLabels], n_classes: int, reliability: np.ndarray | None = None,
                  label_order: list[str] | None = None):
    """Position-aware weighted vote. Weight of a vote = log-odds of its position's reliability (Dawid-Skene style
    one-coin weight, floored at a small positive value). Returns (answer, score in [0,1], raw_count_agreement).
    Ties break by unweighted count, then by the earliest context that voted for the label (deterministic)."""
    n = len(ctxs[0].labels)
    n_bins = 1 if reliability is None else len(reliability)
    rel = np.full(n_bins, UNWEIGHTED_REL) if reliability is None else np.clip(reliability, *RELIABILITY_CLIP)
    K = max(2, n_classes)
    w_bin = np.maximum(WEIGHT_FLOOR, np.log((K - 1) * rel / (1 - rel)))
    answers, scores, counts = [], np.zeros(n), np.zeros(n)
    for i in range(n):
        acc, cnt, first = defaultdict(float), Counter(), {}
        total = 0.0
        for c, ctx in enumerate(ctxs):
            lab = ctx.labels[i]
            b = _bin(ctx.positions[i], ctx.k, n_bins) if reliability is not None else 0
            total += w_bin[b]
            if lab is not None:
                acc[lab] += w_bin[b]
                cnt[lab] += 1
                first.setdefault(lab, c)
        if acc:
            lab = max(acc, key=lambda l: (round(acc[l], VOTE_TIE_DIGITS), cnt[l], -first[l]))
            answers.append(lab)
            scores[i] = acc[lab] / total
            counts[i] = cnt[lab]
        else:
            answers.append(None)
    return answers, scores, counts


async def vote_label(client, alias: str, spec: TaskSpec, texts: list[str], k: int = 20, contexts: int = 3,
                     warmup: int = 0, seed: int = 0, balanced: bool = True, weighted: bool = True, phase: str = "vote",
                     early_frac: float = EARLY_FRAC):
    """early_frac: the cold-start zone (balanced orders) and the reliability bins round(1/early_frac) (4 at 0.25)."""
    n = len(texts)
    orders = (balanced_orders(n, k, contexts, seed, early_frac) if balanced
              else [random_order(n, sub_seed(seed, CONTEXT_STREAM_KEY, c)) for c in range(contexts)])
    ctxs = []
    for c, order in enumerate(orders):
        ctxs.append(await label_batched(client, alias, spec, texts, order, k, warmup=warmup,
                                        seed=seed + CONTEXT_SEED_STRIDE * c, phase=phase))
    rel = position_reliability(ctxs, n_bins_for(early_frac)) if (weighted and contexts > 1) else None
    answers, scores, counts = weighted_vote(ctxs, len(spec.labels), rel)
    return {"answers": answers, "scores": scores, "counts": counts, "reliability": None if rel is None else rel.tolist(), "contexts": ctxs}
