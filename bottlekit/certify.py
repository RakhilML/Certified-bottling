"""Certificates for whole-workload agreement with an expert.

Target: with probability >= 1 - delta, the share of FINAL answers (helper answer if score >= tau, expert answer
otherwise) that disagree with the expert is <= eps.

Risk(tau) = (# items with score >= tau and helper != expert) / N  is non-decreasing as tau decreases, so thresholds
can be tested in a FIXED SEQUENCE from strict to lenient (Learn-then-Test): FWER <= delta without multiplicity
correction. Because the test statistic is monotone too, the fixed sequence over EVERY real threshold has a closed
form: with k_c = the largest calibration error count the test still rejects, tau is placed just above the
(k_c+1)-th highest-scoring calibration error (no grid, no quantile approximation).
Exact tests for "risk(tau) > eps":
  binomial        : calibration items are iid draws from a (super)population -- use for FUTURE items (bottle reuse)
  hypergeometric  : the workload is a fixed set of N items, calibration is a uniform sample WITHOUT replacement
                    (exact for the realized workload; tighter when n_cal/N is not small). Bounds M(tau) = ALL bad
                    items above tau, calibration items included.
  rectified       : (default in the harness) same sampling model, but bounds only what reaches the OUTPUT: every
                    calibration item is answered by the expert, so its error is corrected (rectifying inspection /
                    AOQ). The test bounds U(tau) = unlabelled bad items above tau = M(tau) - K(tau) exactly, for the
                    data-chosen closed-form threshold (proof in _rectified_levels). Never less lenient than
                    'hypergeometric'; the gain grows with n_cal/N and vanishes as n_cal/N -> 0. Finite-population
                    only: there is no rectified 'binomial' (future items are not corrected by this batch's labels).
  sequential      : anytime version with Bonferroni over L PRE-SPECIFIED looks (delta/L per look). Expert labels
                    are bought in stages; the look that maximises the number of BOTTLED items (the exact quantity
                    that minimises expert calls, since calibration items are answered by the expert anyway) is
                    selected, and labelling stops once that objective stops improving. Any look's certificate is
                    valid simultaneously, so the selection is free. With ONE look (single_look_design, the harness
                    default) it is the plain fixed-sequence test at level delta.
Risk profile (risk_profile): the closed-form k-rule leaves the number U of unlabelled bad items it bottles with a
known worst-case law, P[U > u] <= g_k(u) := max_{j<=k} HypCDF(j; N, u+1+j, n) for EVERY u (same proof as
_rectified_levels with m replaced by u), attained by any workload with >= u+1+k bad items. Hence the exact worst-case
expected output error E[U] <= sum_u g_k(u) = (k+1)(N-n)/(n+1), the whole tail P[U/N > eps'] and the delta actually
spent g_k(floor(eps N)) = v_{k_c}. A property of the procedure (n, N, k): no label enters it.

Explicit allowance (recurrence-aware pours): every finite-population entry point takes an optional integer
`m` = the number of output errors allowed in the CERTIFIED POPULATION of size N. None (the default) means
m = floor(eps * N), the standard allowance. The proofs below hold for any population size N and any integer
m >= 0 (eps only enters through m). The harness uses it to certify a sub-population R of a batch (the items not
answered from expert memory) with an allowance chosen from label-free quantities only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import as_strided, sliding_window_view
from scipy.special import gammaln
from scipy.stats import beta, binom, hypergeom

NONE = "__none__"

# ------------------------------------------------------------------------------------------------ named constants
# Design constants are label-free: any value fixed before the batch is valid. The VALUES below are the defaults;
# changing them changes every look size (and therefore every draw), so they are constants, not config knobs.
DEFAULT_DESIGN_C = 1.25      # single-look size constant n = c*sqrt(N/eps); fixed in advance, never tuned
TUNE_WINDOW = 0.1            # sawtooth tuning searches n0*(1 -+ window); label-free
MAX_CAL_SHARE = 0.5          # never calibrate more than this share of a batch BEFORE tuning (the N // 2 clip)
STAGED_FIRST, STAGED_FIRST_WEIGHT = 100, 1.0   # design='staged' (tuned on 4 development tasks): look 1
STAGED_SECOND, STAGED_WEIGHT = 300, 4.0        # then 300, 900, ... (x STAGED_GROWTH) at delta weight 4 each
STAGED_GROWTH = 3
STAGED_LAST_DIV = 10         # staged looks grow up to max(STAGED_SECOND, N // STAGED_LAST_DIV)
STAGED_MAX_LOOKS = 6
REPORT_DELTAS = (0.5, 0.05, 0.01)   # report-only delta grid of the risk profiles; the run's own delta is inserted
TUNE_TIE_TOL = 1e-12         # tune_size: a size must beat the best allowed error fraction by more than this
LAW_SUM_TOL = 1e-9           # chunk_law: a full-support law must sum to 1 within this
MIN_SIZE_SEARCH_LIMIT = 50_000_000   # min_certified_size: give up (ValueError) beyond this batch size
CACHE_K_CRITICAL, CACHE_TUNE_SIZE, CACHE_MIN_SIZE, CACHE_DESIGN_LAW = 65536, 4096, 256, 4096   # lru_cache sizes


def report_deltas(delta: float | None = None) -> tuple:
    """The report-only delta grid (0.5, delta, 0.05, 0.01) of risk profiles and chunk plans (delta omitted if None).
    One definition for certify.risk_profile, certify.batch_tail and the harness (consumers sort and dedupe it)."""
    return (REPORT_DELTAS[0],) + ((float(delta),) if delta is not None else ()) + REPORT_DELTAS[1:]


@dataclass
class Certificate:
    tau: float
    eps: float
    delta: float
    method: str
    n_cal: int
    N: int
    bad_at_tau: int | None
    looks: list = field(default_factory=list)
    chosen_look: int | None = None
    # Every PRE-SPECIFIED look of the procedure as {"n_cal", "alpha", "k"}: the calibration size, its delta share and
    # the k actually applied by its closed-form rule (after any max_expected_eps cap). Label-free; feeds risk_profile.
    plan: list = field(default_factory=list)

    @property
    def certified(self) -> bool:
        return bool(self.tau < np.inf)

    def as_dict(self):
        t = self.tau
        return {"tau": float(t) if np.isfinite(t) else None, "certified": self.certified, "accept_all": bool(t == -np.inf), "eps": self.eps, "delta": self.delta, "method": self.method,
                "n_cal": self.n_cal, "N": self.N, "bad_at_tau": self.bad_at_tau, "looks": self.looks,
                "chosen_look": self.chosen_look,
                "guarantee": f"P[share of final answers != expert > {self.eps}] <= {self.delta}"}


METHODS = ("binomial", "hypergeometric", "rectified")


def _allowance(N: int, eps: float, m: int | None) -> int:
    """Allowed number of output errors in a population of N: m if given (explicit), else floor(eps*N)."""
    if m is None:
        return int(math.floor(eps * N))
    m = int(m)
    if m < 0:
        raise ValueError("the allowance m must be >= 0")
    return m


def _rectified_levels(n: int, N: int, eps: float, m: int | None = None) -> np.ndarray:
    """RECTIFIED test. Entry k = sup over every workload of P[the closed-form k-rule bottles more than m (default
    floor(eps*N); any explicit integer m >= 0 works, the proof never uses eps) UNLABELLED bad items] (non-decreasing
    in k).

    Proof. Fix the workload; order its bad items by score (descending, any label-free tie-break). The k-rule bottles
    the unlabelled items scoring strictly above the (k+1)-th highest-scoring LABELLED bad item (everything if there
    are <= k labelled bad items). An error (> m unlabelled bad items bottled) therefore requires at least m+1
    unlabelled bad items before the (k+1)-th labelled one, i.e. among the first T = min(|B|, m+1+k) bad items at most
    T-m-1 are labelled. For a uniform n-sample the labelled count in a FIXED T-set is Hypergeom(N, T, n), so
        P[error] <= HypCDF(T-m-1; N, T, n) = v_{T-m-1},   v_j := HypCDF(j; N, m+1+j, n),
    which is maximised over workloads by max_{j <= k} v_j (|B| = m+1+j; |B| <= m cannot err; ties only shrink the
    accepted set). v_j = P[>= m+1 unlabelled among m+1+j fixed items] is non-decreasing in j (coupling: a longer
    prefix holds at least as many unlabelled items), so max_{j<=k} v_j = v_k; the running max is kept as a numerical
    safeguard. Sizes m+1+j > N are impossible workloads (v_j := 0); they only occur when n >= N-m, where fewer than
    m+1 items are unlabelled and no error is possible (all v_j = 0)."""
    m = _allowance(N, eps, m)
    js = np.arange(n + 1)
    T = m + 1 + js
    v = np.where(T <= N, hypergeom.cdf(js, N, np.minimum(T, N), n), 0.0)
    return np.maximum.accumulate(v)


def _pvalues(n: int, N: int, eps: float, method: str, m: int | None = None) -> np.ndarray:
    """p-value of 'risk > eps' for every possible calibration error count k = 0..n (increasing in k).
    rectified: the worst-case error probability of the closed-form k-rule (see _rectified_levels).
    m: explicit allowance (finite-population methods only; None = floor(eps*N))."""
    ks = np.arange(n + 1)
    if method == "binomial":
        if m is not None:
            raise ValueError("an explicit allowance m needs a finite-population method (not 'binomial')")
        return np.zeros(n + 1) if eps >= 1 else binom.cdf(ks, n, eps)
    if method == "rectified":
        return _rectified_levels(n, N, eps, m)
    if method != "hypergeometric":
        raise ValueError(f"method must be one of {METHODS}")
    M0 = _allowance(N, eps, m) + 1             # H0: at least M0 bad items in the workload
    if M0 > N:
        return np.zeros(n + 1)
    return hypergeom.cdf(ks, N, M0, n)


def _pvalue(k: int, n: int, N: int, eps: float, method: str, m: int | None = None) -> float:
    return float(_pvalues(n, N, eps, method, m)[k])


@lru_cache(maxsize=CACHE_K_CRITICAL)
def k_critical(n: int, N: int, eps: float, alpha: float, method: str = "hypergeometric", m: int | None = None) -> int:
    """Largest calibration error count that still certifies at level alpha (-1: nothing can be certified).
    m: explicit allowance (None = floor(eps*N)); part of the cache key."""
    if n == 0:
        return -1
    ok = np.nonzero(_pvalues(n, N, eps, method, m) <= alpha)[0]
    return int(ok.max()) if len(ok) else -1


def k_mean_cap(n: int, N: int, max_expected_eps: float) -> int:
    """Largest k whose closed-form rule has worst-case expected output error share (k+1)(N-n)/((n+1)N) <= cap
    (-1: none). A function of (n, N, cap) only -- label-free. sum_u g_k(u) equals this closed form exactly
    (risk_profile; tests), so the cap bounds E[U]/N for every workload."""
    n, N = int(n), int(N)
    if n >= N:
        return n                                  # no unlabelled item: no output error is possible
    cap = float(max_expected_eps)
    if not cap >= 0:
        raise ValueError("max_expected_eps must be >= 0")
    k = min(n, int(math.floor(cap * N * (n + 1) / (N - n))) - 1)
    while k >= 0 and (k + 1) * (N - n) > cap * N * (n + 1):
        k -= 1
    return max(k, -1)


def k_used(n: int, N: int, eps: float, alpha: float, method: str = "hypergeometric",
           *, max_expected_eps: float | None = None, m: int | None = None) -> int:
    """The k the closed-form rule applies: k_critical (with the explicit allowance m, if any), lowered (never raised)
    by the optional mean-risk cap."""
    kc = k_critical(n, N, eps, alpha, method) if m is None else k_critical(n, N, eps, alpha, method, int(m))
    if max_expected_eps is None or kc < 0:
        return kc
    return min(kc, k_mean_cap(n, N, max_expected_eps))


@lru_cache(maxsize=CACHE_TUNE_SIZE)
def tune_size(n0: int, N: int, eps: float, alpha: float, method: str = "hypergeometric", window: float = TUNE_WINDOW,
              m: int | None = None) -> int:
    """Exact tests are discrete: the largest certifiable error count k_c jumps at particular sample sizes (a 3%
    larger calibration set can allow ~30% more calibration errors). Pick the size in [n0(1-w), n0(1+w)] that maximises
    the allowed error fraction (k_c+1)/n (ties -> smallest n). Depends only on (N, eps, alpha): no labels, so any
    certificate built on the tuned size stays valid. m: explicit allowance (None = floor(eps*N)); label-free too."""
    lo, hi = max(1, int(n0 * (1 - window))), min(N, int(np.ceil(n0 * (1 + window))))
    if hi <= lo:
        return min(n0, N)
    best, best_val = min(n0, N), -1.0
    for n in range(lo, hi + 1):
        val = ((k_critical(n, N, eps, alpha, method) if m is None else k_critical(n, N, eps, alpha, method, int(m)))
               + 1) / n
        if val > best_val + TUNE_TIE_TOL:
            best, best_val = n, val
    return best


def fixed_sequence(scores, bad, N: int, eps: float, delta: float, method: str = "hypergeometric",
                   *, max_expected_eps: float | None = None, m: int | None = None):
    """scores, bad: arrays over the calibration items. Returns (tau, bad count at tau); tau=+inf: nothing certified,
    tau=-inf: everything certified. Exact over all real thresholds (see module docstring). m: explicit allowance
    for the population of N (None = floor(eps*N)).
    max_expected_eps (optional, stricter only): k is lowered to k_mean_cap(n, N, cap) if that is smaller, so the
    worst-case expected output error share (of the population N) is <= cap on top of P[share > eps] <= delta
    (v_k is non-decreasing in k)."""
    s = np.asarray(scores, float)
    b = np.asarray(bad, bool)
    if len(s) == 0 or N <= 0:
        return np.inf, None
    kc = k_used(len(s), N, eps, delta, method, max_expected_eps=max_expected_eps, m=m)
    if kc < 0:
        return np.inf, None
    bs = np.sort(s[b])[::-1]
    if len(bs) <= kc:
        return -np.inf, int(len(bs))
    tau = float(np.nextafter(bs[kc], np.inf))
    return tau, int((s[b] >= tau).sum())


def certified_eps(scores_cal, bad_cal, tau: float, N: int, delta: float = 0.1, method: str = "hypergeometric",
                  k_end: int | None = None, denom: int | None = None) -> float:
    """Invert the test for ONE threshold chosen without looking at calibration labels (e.g. by an expert budget):
    the smallest eps such that 'risk(tau) > eps' is rejected at level delta. Valid upper bound on the realized
    disagreement share of the final output, with probability >= 1 - delta.
    rectified: M(tau) <= M_upper (the hypergeometric inversion) with probability >= 1 - delta, and the bottled errors
    are exactly M(tau) - K_end, K_end = labelled bad items scoring >= tau in the FULL final labelled set (all are
    answered by the expert); default K_end = the count in (scores_cal, bad_cal). Returns max(0, M_upper - K_end)/N.
    denom (finite-population methods): the share is taken over `denom` final answers instead of N. Used when the
    certified population (N items) is a label-free subset of a batch of `denom` items whose other answers add no
    errors (recurrence-aware pours: the bound on M is over the population, the share over the whole batch)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    s, b = np.asarray(scores_cal, float), np.asarray(bad_cal, bool)
    n = len(s)
    if n == 0 or N <= 0:
        return 1.0
    k = int(((s >= tau) & b).sum())
    if method == "binomial":  # exact Clopper-Pearson upper bound
        return 1.0 if k >= n else float(beta.ppf(1 - delta, k + 1, n - k))
    lo, hi = k, N + 1           # smallest M0 with P[K <= k | M = M0] <= delta (cdf decreasing in M0)  =>  M <= M0-1
    while lo < hi:
        mid = (lo + hi) // 2
        if mid <= N and hypergeom.cdf(k, N, mid, n) <= delta:
            hi = mid
        else:
            lo = mid + 1
    D = N if denom is None else int(denom)
    if D < N:
        raise ValueError("denom (batch size) cannot be smaller than the certified population N")
    if method == "rectified":
        ke = k if k_end is None else int(k_end)
        if ke < k:
            raise ValueError("k_end counts the full labelled set, so it cannot be smaller than the count in this sample")
        m_upper = N if lo > N else lo - 1
        return max(0, m_upper - ke) / D
    if denom is None:
        return 1.0 if lo > N else (lo - 1) / N
    return (N if lo > N else lo - 1) / D


def certify(scores_cal, bad_cal, N: int, eps: float = 0.05, delta: float = 0.1, method: str = "hypergeometric",
            *, max_expected_eps: float | None = None, m: int | None = None) -> Certificate:
    tau, k = fixed_sequence(scores_cal, bad_cal, N, eps, delta, method, max_expected_eps=max_expected_eps, m=m)
    n = len(scores_cal)
    plan = [{"n_cal": int(n), "alpha": float(delta),
             "k": k_used(n, N, eps, delta, method, max_expected_eps=max_expected_eps, m=m)
                   if (n > 0 and N > 0) else -1}]
    return Certificate(tau, eps, delta, method, n, N, k, plan=plan)


def default_design(N: int, max_looks: int = STAGED_MAX_LOOKS) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Default staged design, fixed from N alone before any label is seen: a cheap first look of 100 (delta weight 1)
    then 300, 900, 2700, ... up to max(300, N/10) (weight 4 each). Chosen on 4 development tasks (best mean
    bottled share over eps in {.02,.05,.1}); only the two-look case (N <= 3000) was tuned on data."""
    pairs, n = [(STAGED_FIRST, STAGED_FIRST_WEIGHT)], STAGED_SECOND
    while n <= max(STAGED_SECOND, N // STAGED_LAST_DIV) and len(pairs) < max_looks:
        pairs.append((n, STAGED_WEIGHT))
        n *= STAGED_GROWTH
    merged: dict[int, float] = {}
    for l, w in pairs:
        l = min(l, max(1, _cal_clip(N)))      # never calibrate more than half of a small workload
        if l > 0:
            merged[l] = merged.get(l, 0.0) + w
    looks = tuple(sorted(merged))
    return looks, tuple(merged[l] for l in looks)


def default_looks(N: int) -> tuple[int, ...]:
    return default_design(N)[0]


def _cal_clip(N: int) -> int:
    """floor(MAX_CAL_SHARE * N): the largest pre-tuning look (N // 2 at the default 0.5; exact for any int N)."""
    return int(N) // 2 if MAX_CAL_SHARE == 0.5 else int(math.floor(MAX_CAL_SHARE * int(N)))


def single_look_design(N: int, eps: float, c: float = DEFAULT_DESIGN_C) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """One pre-specified look at the full delta (the harness default): n = round(c * sqrt(N / eps)),
    clipped to [1, N // 2] BEFORE tuning. A function of (N, eps, c) only -- no score, answer or label -- fixed before
    the calibration permutation is drawn; the harness then sawtooth-tunes it (+-10%, tune_size) from
    (N, eps, delta, method) only, which can take it slightly above N // 2 for small N (e.g. 53-54 at N=100).
    With one look certify_sequential is exactly the closed-form k-rule at level delta (no Bonferroni split, no stop).
    Why: under the rectified test the staged design (100, 300 at weights 1:4, tuned for the hypergeometric test) runs
    its first look at delta/5 (k_c ~ 0-1) and its second look labels ~54% of a 400-item batch; one full-delta look of
    ~sqrt(N/eps) labels bottles more on average. For N <= 4*c^2/eps (125 at
    eps=.05, c=1.25) the size is N // 2, the small-N look of default_design."""
    if not (eps > 0 and c > 0):
        raise ValueError("single_look_design needs eps > 0 and c > 0")
    n = max(1, min(_cal_clip(N), int(round(c * math.sqrt(max(int(N), 0) / eps)))))
    return (n,), (1.0,)


def disagreement(answer, expert) -> bool:
    """An item is BAD if the helper has no answer, or its answer differs from the expert's (an expert failure
    counts as disagreement: conservative)."""
    return answer is None or answer == NONE or expert is None or expert == NONE or answer != expert


def spend(looks, spending=None) -> list[float]:
    """Per-look share of delta (Bonferroni): uniform by default, or proportional to `spending` weights."""
    w = np.ones(len(looks)) if spending is None else np.asarray(spending, float)[:len(looks)]
    if len(w) != len(looks) or (w <= 0).any():
        raise ValueError("spending needs one positive weight per look")
    return (w / w.sum()).tolist()


def tuned_looks(looks, shares, N, eps, delta, method="hypergeometric", m: int | None = None,
                window: float = TUNE_WINDOW) -> list[int]:
    """Sawtooth-tuned look sizes (see tune_size), kept strictly increasing. m: explicit allowance (None: floor(eps*N)).
    window: the tuning window (TUNE_WINDOW by default; min_certified_size passes its own)."""
    out = []
    wkw = {} if window == TUNE_WINDOW else {"window": float(window)}     # default: the plain call (same cache entries)
    for n0, w in zip(looks, shares):
        n = (tune_size(n0, N, eps, delta * w, method, **wkw) if m is None
             else tune_size(n0, N, eps, delta * w, method, m=int(m), **wkw))
        out.append(max(n, out[-1] + 1) if out else n)
    return [min(n, N) for n in out]


async def certify_sequential(score_all: np.ndarray, answer_all: list, expert_fn, N: int, eps: float = 0.05,
                             delta: float = 0.1, looks=None, target_coverage: float | None = None, seed=0,
                             patience: int | None = None, method: str = "hypergeometric",
                             spending=None, tune: bool = False,
                             *, max_expected_eps: float | None = None, m: int | None = None) -> tuple[Certificate, dict]:
    """Anytime certificate: expert-label a uniform random permutation of the workload in stages (looks).
    At look j run the exact fixed-sequence test at level delta*w_j (Bonferroni; w uniform or `spending`). The
    objective of a look is the number of items it would BOTTLE = #(score >= tau, usable answer, not expert-labelled)
    -- expert calls of the whole run are exactly N minus this number. Stopping (never affects validity):
      * dominance: stop when the best objective so far is >= the most any later look could possibly bottle
        (#usable items left unlabelled after the next look) -- this never gives up a better look;
      * patience (optional heuristic): stop when the objective has not improved for `patience` looks (looks that
        certify nothing do not count: labelling costs nothing extra while nothing is bottled);
      * target_coverage reached, or the last look done.
    Return the best look (Bonferroni makes every look's certificate valid at once).
    max_expected_eps (optional, stricter only): look j's k is capped at k_mean_cap(n_j, N, cap * share_j), so the
    union bound on E[output errors]/N stays <= cap (E[max_j U_j] <= sum_j E[U_j]).
    The certificate's `plan` lists every pre-specified look with the k its rule applies (for risk_profile).

    expert_fn(idx_list) -> awaitable dict idx -> expert label (called incrementally, only for new items).
    m: explicit allowance for the population of N (None = floor(eps*N)); every look uses it.
    Returns (certificate, expert labels obtained {idx: label}).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    score_all = np.asarray(score_all, float)
    if not looks:
        looks, w0 = default_design(N)
        spending = spending or w0
    looks = sorted({min(int(l), N) for l in looks if int(l) > 0})
    L = max(1, len(looks))
    shares = spend(looks, spending if spending and len(spending) == len(looks) else None)
    if tune:
        looks = tuned_looks(looks, shares, N, eps, delta, method, m)
    caps = [None if max_expected_eps is None else float(max_expected_eps) * w for w in shares]
    plan = [{"n_cal": int(nl), "alpha": float(delta * w), "k": k_used(int(nl), N, eps, delta * w, method, max_expected_eps=c, m=m)}
            for nl, w, c in zip(looks, shares, caps)]
    order = np.random.default_rng(seed).permutation(N)
    usable = np.array([a is not None and a != NONE for a in answer_all], bool)
    labels: dict[int, object] = {}
    history, taus, ks = [], [], []
    stale = 0
    for j, n_look in enumerate(looks):
        new = [int(i) for i in order[len(labels):n_look]]
        if new:
            labels.update(await expert_fn(new))
        idx = order[:n_look]
        bad = np.array([disagreement(answer_all[i], labels[i]) for i in idx])
        tau, k = fixed_sequence(score_all[idx], bad, N, eps, delta * shares[j], method, max_expected_eps=caps[j], m=m)
        taus.append(tau)
        ks.append(k)
        free = usable.copy()
        free[idx] = False
        vals = [int(((score_all >= t) & free).sum()) for t in taus]   # every look so far, with the CURRENT labels
        history.append({"n_cal": int(n_look), "tau": float(tau) if np.isfinite(tau) else None, "certified": bool(tau < np.inf),
                        "accept_all": bool(tau == -np.inf),
                        "bottled": vals[-1], "coverage_est": float(((score_all >= tau) & usable).mean()) if N else 0.0})
        best_prev = max(vals[:-1], default=0)
        if vals[-1] > best_prev:
            stale = 0
        elif max(vals) > 0:
            stale += 1
        if j + 1 < len(looks):
            later_max = int(usable[order[looks[j + 1]:]].sum())        # everything usable left after the next look
            if max(vals) >= later_max:
                break
        if (target_coverage is not None and vals[-1] >= target_coverage * N) or (patience is not None and stale >= patience):
            break
    free = usable.copy()
    free[list(labels)] = False
    vals = [int(((score_all >= t) & free).sum()) for t in taus]
    jb = int(np.argmax(vals)) if vals else 0                           # ties -> earliest (strictest) look
    tau = taus[jb] if taus else np.inf
    cert = Certificate(tau, eps, delta, f"sequential-{method} (L={L} looks, Bonferroni delta shares {[round(x, 3) for x in shares]}, bottled-count look selection)",
                       len(labels), N, ks[jb] if ks else None, history, jb, plan=plan)
    return cert, labels


def sequential_certified_eps(score_all, answer_all, labels: dict, order_seed, looks, tau: float, N: int,
                             delta: float, method: str = "hypergeometric", spending=None,
                             denom: int | None = None) -> float:
    """certified_eps for a label-independent threshold after a SEQUENTIAL calibration: every look's prefix gives a
    valid bound at its Bonferroni share of delta, so the smallest over the looks actually labelled is valid.
    rectified: min over looks of (M_upper_j - K_end)/N, with M_upper_j from prefix j at delta*share_j and K_end the
    labelled bad items scoring >= tau among ALL labels (labels bought after look j only remove output errors).
    denom: share over `denom` final answers (a batch whose certified population is these N items), see certified_eps."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    score_all = np.asarray(score_all, float)
    if not looks:
        looks, w0 = default_design(N)
        spending = spending or w0
    looks = sorted({min(int(l), N) for l in looks if int(l) > 0})
    shares = spend(looks, spending if spending and len(spending) == len(looks) else None)
    order = np.random.default_rng(order_seed).permutation(N)
    k_end = None
    if method == "rectified":
        k_end = sum(1 for i, lab in labels.items() if score_all[int(i)] >= tau and disagreement(answer_all[int(i)], lab))
    best = 1.0
    for j, n_look in enumerate(looks):
        idx = order[:n_look]
        if not all(int(i) in labels for i in idx):
            break
        bad = [disagreement(answer_all[i], labels[int(i)]) for i in idx]
        best = min(best, certified_eps(score_all[idx], bad, tau, N, delta * shares[j], method, k_end=k_end,
                                       denom=denom))
    return best


def inversion_profile(deltas, score_all, answer_all, labels: dict, tau: float, N: int, method: str = "hypergeometric",
                      order_seed=None, looks=None, spending=None) -> dict:
    """Budget path (label-independent threshold tau): {f"{d:g}": certified eps at level d} for each d, by the same
    test inversion as eps_achieved (sequential_certified_eps when order_seed is given, else certified_eps on the
    fixed calibration set `labels`). Each entry is a (1-d) confidence bound on the output error share for that d
    separately -- a data-dependent bound, not a distribution tail: no expectation follows from it."""
    s = np.asarray(score_all, float)
    out = {}
    for d in sorted({float(x) for x in deltas}, reverse=True):
        if order_seed is not None:
            e = sequential_certified_eps(s, answer_all, labels, order_seed, looks, tau, N, d, method=method,
                                         spending=spending)
        else:
            idx = sorted(int(i) for i in labels)
            e = certified_eps(s[idx], [disagreement(answer_all[i], labels[i]) for i in idx], tau, N, d, method)
        out[f"{d:g}"] = float(e)
    return out


# ------------------------------------------------------------------------------------------------ risk profile
RISK_EPS_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
RISK_TOL = 1e-12
RISK_MAX_CELLS = 50_000_000      # cap on the dense (u, j) table of _tail_bound (about 400 MB of float64)


class RiskTooLarge(ValueError):
    """The exact tail table would exceed RISK_MAX_CELLS; callers fall back to the closed-form expectation."""


def _tail_bound(n: int, N: int, k: int, u_stop: int | None = None, tol: float = RISK_TOL) -> tuple[np.ndarray, float]:
    """Worst-case tail of U = #unlabelled bad items bottled by the closed-form k-rule with a uniform n-of-N sample:
        P[U > u] <= g(u) := max_{j<=k} HypCDF(j; N, u+1+j, n)       (0 when u+1+j > N),  every u, every workload.
    Proof (as _rectified_levels, with m replaced by u): U > u needs >= u+1 unlabelled bad items strictly above the
    (k+1)-th labelled bad item, so among the first T = min(|B|, u+1+k) bad items (score order, label-free tie-break)
    at most T-u-1 are labelled; the labelled count in a FIXED T-set is Hypergeom(N, T, n). Maximising over |B| gives
    g(u) (|B| <= u cannot exceed u; |B| = u+1+j > N is impossible). The running max over j <= k is kept as a numerical
    safeguard (v_j(u) is non-decreasing in j by coupling), and g is made non-increasing in u (also true exactly).
    Returns (g[0..u_stop], rest) with rest >= sum_{u > u_stop} g(u) (rigorous, by monotonicity in u). U <= N-n, so
    g(u) = 0 for u >= N-n. Default u_stop: the first u with g(u) < tol (binary search on j = k).
    Numerics: log-pmf by exact recurrences (no scipy calls; scipy's cdf costs ~0.3 ms/point at N = 1e5):
        log P[X=i | T] = st_i - Sa_i + Sa_T + Sb_{T-i},   st_i = sum_{t<i} log((n-t)/(N-t)),
        Sa_T = sum_{t<T} log((t+1)/(N-t)),   Sb_D = sum_{d<D} log((N-n-d)/(d+1))   (-inf once d >= N-n)."""
    n, N, k = int(n), int(N), int(k)
    U_max = N - n                                          # unlabelled items: U <= U_max, so g(u) = 0 for u >= U_max
    if k < 0 or U_max <= 0:
        return np.zeros(0), 0.0
    k = min(k, n)
    i = np.arange(k + 1)
    if u_stop is None:
        lnC = lambda a, b: gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)  # noqa: E731

        def cdf_k(u: int) -> float:                        # v_k(u) = HypCDF(k; N, u+1+k, n), only to place u_stop
            T = u + 1 + k
            if T > N:
                return 0.0
            j = i[(i <= n) & (n - i <= N - T)]
            return float(np.exp(lnC(T, j) + lnC(N - T, n - j) - lnC(N, n)).sum())

        lo, hi = 0, U_max - 1                              # smallest u with v_k(u) < tol (v_k non-increasing in u)
        if cdf_k(hi) >= tol:
            lo = hi
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf_k(mid) < tol:
                hi = mid
            else:
                lo = mid + 1
        u_stop = lo
    u_stop = max(0, min(int(u_stop), U_max - 1))
    R = u_stop + 2 + k                                     # rows T = 0 .. u_stop+1+k (rows with T > N are impossible: 0)
    Tn = min(N, R - 1)                                     # largest attainable T computed
    with np.errstate(divide="ignore"):
        t = np.arange(Tn, dtype=float)
        Sa = np.concatenate([[0.0], np.cumsum(np.log(t + 1.0) - np.log(N - t))])                    # T = 0..Tn
        Sb = np.concatenate([np.full(k + 1, -np.inf), [0.0],                                       # D < 0: -inf
                             np.cumsum(np.log(np.maximum(N - n - t, 0.0)) - np.log(t + 1.0))])
        st = np.concatenate([[0.0], np.cumsum(np.log((n - i[:-1]) / (N - i[:-1])))])
    base = st - Sa[i]                                      # log P[X=i | T] = base_i + Sa_T + Sb_{T-i}   (T >= i)
    W = sliding_window_view(Sb, k + 1)                     # W[T+1, ::-1][i] = Sb_{T-i}  (offset k+1 for D < 0)
    if R * (k + 1) > RISK_MAX_CELLS:
        raise RiskTooLarge(f"exact tail table {R} x {k + 1} exceeds {RISK_MAX_CELLS} cells (N={N}, n={n}, k={k})")
    C = np.zeros((R, k + 1))
    Lg = W[1:Tn + 2, ::-1] + base[None, :]
    Lg += Sa[:, None]
    np.exp(Lg, out=Lg)
    np.cumsum(Lg, axis=1, out=C[:Tn + 1])                  # C[T, j] = HypCDF(j; N, T, n)
    np.minimum(C, 1.0, out=C)
    G = as_strided(C[1:], shape=(u_stop + 1, k + 1), strides=(C.strides[0], C.strides[0] + C.strides[1]),
                   writeable=False)                        # G[u, j] = C[u+1+j, j] = v_j(u)
    g = G.max(axis=1)                                      # max over j <= k
    g = np.maximum.accumulate(g[::-1])[::-1]               # non-increasing in u
    rest = max(0, U_max - 1 - u_stop) * float(g[-1])
    return g, rest


def expected_eps_closed_form(n: int, N: int, k: int) -> float:
    """sum_u g_k(u) / N = (k+1)(N-n)/((n+1)N): the mean of the negative hypergeometric number of unlabelled items
    before the (k+1)-th labelled one (the all-bad workload attains every g_k(u) at once). 0 if k < 0 or n >= N."""
    n, N, k = int(n), int(N), int(k)
    if k < 0 or n >= N or N <= 0:
        return 0.0
    return (min(k, n) + 1) * (N - n) / ((n + 1) * N)


def risk_profile(n, N: int, k, eps: float | None = None, delta: float | None = None, eps_grid=RISK_EPS_GRID,
                 deltas=None, n_unusable: int = 0, tol: float = RISK_TOL) -> dict:
    """Exact worst-case risk profile of the closed-form k-rule(s) of a certificate (report-only; label-free).
    n, k: ints (one look: basis 'exact'), or equal-length sequences, one entry per PRE-SPECIFIED look of a staged or
    custom design (basis 'union bound': the selected look j* bottles at most U_{j*} <= max_j U_j unlabelled bad items,
    so P[U > u] <= min(1, sum_j g_j(u)) and E[U] <= sum_u min(1, sum_j g_j(u))). Looks with k = -1 contribute 0.
    U = #bottled answers != expert among items NOT labelled by the expert; the final disagreement share is
    <= U/N + F/N, F = n_unusable (expert-routed items without a usable expert answer, observed after the run).
    Returns
      expected_eps_bound : sup over workloads of E[U]/N (single look: = (k+1)(N-n)/((n+1)N), attained by a workload
                           whose bad items outnumber u+1+k for every relevant u). F/N is NOT part of it.
      tail               : {eps': bound on P[U/N > eps']} = g(floor(eps' N)); each holds for the PROCEDURE (a fixed
                           eps'), not for an eps' chosen after seeing this batch's calibration labels.
      eps_at_delta       : {d: smallest u/N with g(u) <= d}, i.e. P[U/N > that] <= d.
      delta_spent        : g(floor(eps N)) (single look: v_{k_c} <= delta).
    All u describe the same random variable U, so the whole curve holds at once (no multiplicity)."""
    ns, ks = [int(x) for x in np.atleast_1d(n)], [int(x) for x in np.atleast_1d(k)]
    if len(ns) != len(ks) or not ns:
        raise ValueError("risk_profile needs one k per look")
    N = int(N)
    basis = "exact" if len(ns) == 1 else "union bound"
    parts = [_tail_bound(a, N, b, tol=tol) for a, b in zip(ns, ks)]
    L = max(len(g) for g, _ in parts)
    if len(parts) > 1:      # common u range: recompute every look up to the longest truncation point
        parts = [_tail_bound(a, N, b, u_stop=L - 1, tol=tol) for a, b in zip(ns, ks)]
    G = np.zeros(L)
    for g, _ in parts:
        G[:len(g)] += g
    G = np.minimum(G, 1.0)
    rest = float(sum(r for _, r in parts))
    u_max = max(0, N - min(ns))                            # U <= N - n_1 (the fewest labels), so g(u) = 0 beyond

    def at(u: int) -> float:                               # bound on P[U > u]
        if u < L:
            return float(G[u])
        return float(min(1.0, sum(float(g[-1]) for (g, _), a in zip(parts, ns) if len(g) and u < N - a)))

    exp_count = float(G.sum()) + rest
    if len(ns) == 1:        # one look: the closed form is the exact sup (keeps float noise from undercutting it)
        exp_count = max(exp_count, expected_eps_closed_form(ns[0], N, ks[0]) * N)
    if deltas is None:
        deltas = report_deltas(delta)
    eps_at = {}
    for d in sorted({float(x) for x in deltas}, reverse=True):
        hit = np.nonzero(G <= d)[0]
        eps_at[f"{d:g}"] = ((int(hit[0]) if len(hit) else (L if at(L) <= d else u_max)) / N) if N else 0.0
    return {"basis": basis, "looks": [{"n_cal": a, "k": b} for a, b in zip(ns, ks)],
            "expected_eps_bound": exp_count / N if N else 0.0,
            "expected_eps_closed_form": float(sum(expected_eps_closed_form(a, N, b) for a, b in zip(ns, ks))),
            "tail": {f"{e:g}": at(int(math.floor(e * N))) for e in eps_grid},
            "eps_at_delta": eps_at,
            "delta_spent": at(int(math.floor(eps * N))) if eps is not None else None,
            "eps": eps, "delta": delta,
            "n_expert_unusable": int(n_unusable), "unusable_share": int(n_unusable) / N if N else 0.0,
            "scope": "U = #bottled answers != expert (bottled = accepted and not expert-labelled); final share <= "
                     "U/N + unusable_share. Worst case over workloads, over the calibration draw; depends on (n, N, k) "
                     "only"}


# ------------------------------------------------------------------------------------------------ chunked batches
BATCH_FULL_SUPPORT_MAX = 6_000   # per-chunk unlabelled count N_j - n_j up to which the full law (no truncation) is
                                 # used; beyond it the law is cut where g < tol and the cut mass is added to every tail


def chunk_law(n: int, N: int, k: int, tol: float = RISK_TOL, full: bool | None = None) -> tuple[np.ndarray, float]:
    """Worst-case law of U = #unlabelled bad items bottled by ONE closed-form k-rule (uniform n-of-N sample) as a
    probability mass function: p(u) = g(u-1) - g(u) with g(-1) = 1 and g from _tail_bound (P[U > u] <= g(u) for every
    u, every workload). Any U with P[U > u] <= g(u) for all u is stochastically dominated by V ~ p.
    full (default: N - n <= BATCH_FULL_SUPPORT_MAX): the whole support 0..N-n (no truncation; sums to 1). Otherwise g is
    computed up to u_stop with g(u_stop) < tol, and the remaining mass P[V > u_stop] = g(u_stop) is returned
    separately as `overflow` (callers add it to every tail: a rigorous union bound).
    Returns (pmf, overflow). k < 0 or n >= N: U = 0 surely (pmf = [1])."""
    n, N, k = int(n), int(N), int(k)
    U_max = N - n
    if k < 0 or U_max <= 0:
        return np.ones(1), 0.0
    if full is None:
        full = U_max <= BATCH_FULL_SUPPORT_MAX
    g, _ = _tail_bound(n, N, k, u_stop=(U_max - 1) if full else None, tol=tol)
    g = np.clip(g, 0.0, 1.0)
    gm = np.concatenate([[1.0], g])                       # g(-1) = 1, g(0), ..., g(u_stop)
    pmf = np.clip(gm[:-1] - gm[1:], 0.0, None)            # masses at 0..u_stop (g non-increasing: already >= 0)
    if full:                                              # u_stop = U_max - 1: P[U = U_max] = g(U_max - 1), g(U_max) = 0
        pmf = np.append(pmf, gm[-1])
        s = float(pmf.sum())
        if not abs(s - 1.0) < LAW_SUM_TOL:
            raise AssertionError(f"chunk law does not sum to 1 (sum={s!r}, n={n}, N={N}, k={k})")
        return pmf, 0.0
    return pmf, float(gm[-1])


BATCH_TAIL_DELTA = 0.1      # batch_tail: delta inserted into the default report grid when no delta is given


def batch_tail(designs, N_total: int | None = None, eps: float | None = None, deltas=None,
               eps_grid=RISK_EPS_GRID, tol: float = RISK_TOL, survival: bool = False, *,
               delta: float | None = None) -> dict:
    """Exact worst-case law of the WHOLE-BATCH output error count U = U_1 + ... + U_J of a batch certified as J chunks
    in sequence (chunked pours), each chunk j by its own closed-form k-rule with a uniform calibration sample.
    designs: [(n_j, N_j, k_j), ...], every entry fixed before chunk 1 starts (label-free).
    Claim: P[U > u] <= P[V_1 + ... + V_J > u] for every u, V_j independent with the worst-case laws of chunk_law.
    Proof. Let F_{j-1} = everything up to the end of chunk j-1 (its expert labels included). Chunk j's scores and
    answers are F_{j-1}-measurable and its calibration sample is a fresh uniform draw independent of F_{j-1}, so the
    single-batch theorem (_tail_bound) holds conditionally: P[U_j > u | F_{j-1}] <= g_j(u) for every u. With
    S_j = U_1 + ... + U_j and h(x) = P[V_j > t - x] (non-decreasing in x):
        P[S_j > t] = E[P[U_j > t - S_{j-1} | F_{j-1}]] <= E[h(S_{j-1})] <= E[h(W_{j-1})] = P[W_{j-1} + V_j > t],
    where W_{j-1} = V_1 + ... + V_{j-1} (independent of V_j) and the second inequality is the induction hypothesis
    S_{j-1} <=_st W_{j-1} applied to the non-decreasing h. The law of W_J is the convolution of the chunk pmfs.
    Returns
      delta_batch        : P[U > floor(eps * N_total)] bound (needs eps)
      tail               : {eps': bound on P[U/N_total > eps']}
      eps_at_delta       : {d: smallest u/N_total with bound <= d}
      expected_eps_bound : sum_j (k_j+1)(N_j-n_j)/(n_j+1) / N_total (exact worst-case mean; no convolution needed)
      overflow           : total truncated mass (0 with full supports), added to every tail value.
      survival           : (only with survival=True) list s with s[u] = bound on P[U > u], u = 0, 1, ...
    N_total defaults to sum N_j. A property of the plan only: no label enters it.
    deltas default: report_deltas(delta) = (0.5, delta, 0.05, 0.01); delta=None uses the grid with delta 0.1."""
    ds = [(int(n), int(N), int(k)) for n, N, k in designs]
    if not ds:
        raise ValueError("batch_tail needs at least one chunk design")
    Ntot = int(N_total) if N_total is not None else sum(N for _, N, _ in ds)
    if Ntot <= 0:
        raise ValueError("N_total must be > 0")
    conv, over = np.ones(1), 0.0
    for n, N, k in ds:
        p, o = chunk_law(n, N, k, tol=tol)
        conv = np.convolve(conv, p)
        over += o
    conv = np.clip(conv, 0.0, None)
    surv = np.concatenate([np.cumsum(conv[::-1])[::-1][1:], [0.0]])   # surv[u] = P[W > u] (sub-probability part)

    def at(u: int) -> float:
        if u < 0:
            return 1.0
        return float(min(1.0, (surv[u] if u < len(surv) else 0.0) + over))

    exp_count = float(sum(expected_eps_closed_form(n, N, k) * N for n, N, k in ds))
    if deltas is None:
        deltas = report_deltas(BATCH_TAIL_DELTA if delta is None else delta)
    eps_at = {}
    for d in sorted({float(x) for x in deltas}, reverse=True):
        hit = np.nonzero(np.minimum(1.0, surv + over) <= d)[0]
        eps_at[f"{d:g}"] = (int(hit[0]) if len(hit) else len(surv)) / Ntot
    extra = {"survival": np.minimum(1.0, surv + over).tolist()} if survival else {}
    return {"basis": "convolution", "chunks": [{"n_cal": n, "N": N, "k": k} for n, N, k in ds], "N": Ntot,
            **extra, "expected_eps_bound": exp_count / Ntot,
            "tail": {f"{e:g}": at(int(math.floor(e * Ntot))) for e in eps_grid},
            "eps_at_delta": eps_at,
            "delta_batch": at(int(math.floor(eps * Ntot))) if eps is not None else None,
            "eps": eps, "overflow": float(over),
            "scope": "U = #bottled answers != expert over the whole batch (sum over chunks of unlabelled bad items "
                     "bottled); worst case over workloads and calibration draws, by conditional stochastic dominance "
                     "and exact convolution of the per-chunk laws; depends on the plan (n_j, N_j, k_j) only"}


BATCH_SPEND_MAX_ALPHA = 0.5     # batch-spent chunk plans: default ceiling on the common per-chunk level
                                # (BottleConfig.chunk_alpha_max): above 1/2 a chunk's own claim is a coin flip


def batch_spent_alpha(designs_n, N_total: int | None, eps: float, delta: float, method: str = "rectified",
                      *, max_expected_eps: float | None = None, alpha_max: float = BATCH_SPEND_MAX_ALPHA,
                      tol: float = RISK_TOL) -> tuple[float, list, dict]:
    """Batch-spent chunk plan: spend the whole batch's delta across the chunks of a chunked pour by
    raising every chunk's k to a COMMON per-chunk level alpha* >= delta, at the planned calibration sizes.
    designs_n: [(n_j, N_j), ...], fixed before chunk 1 (label-free); the n_j are kept (no re-tuning at alpha*).
    k_j(alpha) = k_used(n_j, N_j, eps, alpha, method, max_expected_eps=cap): exactly the k chunk j's closed-form rule
    applies when it is certified with one look of n_j at level alpha (full share, so the mean-risk cap is applied as
    the chunk applies it).
    Returns the LARGEST k-vector of the form (k_j(alpha))_j with alpha in [delta, alpha_max] whose whole-batch tail
    batch_tail(...)['delta_batch'] <= delta, and alpha_star = the SMALLEST alpha giving that k-vector
    (= max(delta, max_j pv_j[k_j]); every chunk then holds at (eps, alpha_star) on its own).
    Exact search: k_j(alpha) is a step function that changes only at chunk j's p-values pv_j = _pvalues(n_j, N_j, eps)
    (non-decreasing in k), so the candidates are delta and every such p-value in (delta, alpha_max]. delta_batch is
    non-decreasing along them (k_j is non-decreasing in alpha; g_k(u) = max_{i<=k} HypCDF(i; N, u+1+i, n) is
    non-decreasing in k, so each chunk law, and hence their convolution, grows stochastically): bisection. The result
    is then RE-CHECKED exactly with batch_tail on the final integer k-vector (stepping down if a numerical edge ever
    failed it); the per-chunk law g_k never uses alpha, so the convolution proof of batch_tail holds unchanged.
    If the delta-level plan itself has delta_batch > delta, alpha_star = delta and the delta-level designs are
    returned with info['ok'] = False. Label-free: a function of (designs_n, N_total, eps, delta, method, cap) only.
    Returns (alpha_star, designs [(n_j, N_j, k_j)], info)."""
    ds = [(int(n), int(N)) for n, N in designs_n]
    if not ds:
        raise ValueError("batch_spent_alpha needs at least one chunk design")
    if not (0 < delta < 1 and delta <= alpha_max < 1):
        raise ValueError("need 0 < delta <= alpha_max < 1")
    Ntot = int(N_total) if N_total is not None else sum(N for _, N in ds)
    cap = max_expected_eps

    def ks(a: float) -> list:
        return [int(k_used(n, N, eps, a, method, max_expected_eps=cap)) if (n > 0 and N > 0) else -1 for n, N in ds]

    def tail(kv: list) -> float:
        return float(batch_tail([(n, N, k) for (n, N), k in zip(ds, kv)], Ntot, eps, tol=tol)["delta_batch"])

    cands = {float(delta)}
    pvs = []
    for n, N in ds:
        pv = _pvalues(n, N, eps, method) if (n > 0 and N > 0) else np.zeros(0)
        pvs.append(pv)
        cands.update(float(p) for p in np.unique(pv) if delta < p <= alpha_max)
    cands = sorted(cands)
    k0 = ks(cands[0])
    d0 = tail(k0)
    info = {"k_chunk_level": k0, "delta_batch_chunk_level": d0, "candidates": len(cands), "alpha_max": alpha_max}
    if not d0 <= delta:
        return float(delta), [(n, N, k) for (n, N), k in zip(ds, k0)], {**info, "ok": False, "delta_batch": d0}
    lo, hi = 0, len(cands) - 1                 # cands[lo] passes; find the largest passing candidate
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tail(ks(cands[mid])) <= delta:
            lo = mid
        else:
            hi = mid - 1
    i = lo
    kv, db = ks(cands[i]), None
    while True:                                # exact re-check on the final integer k-vector
        db = tail(kv)
        if db <= delta or i == 0:
            break
        i -= 1
        kv = ks(cands[i])
    if not db <= delta:                        # cannot happen: cands[0] = delta passed above
        raise AssertionError(f"batch-spent plan fails its own re-check (delta_batch={db!r} > {delta})")
    alpha = float(delta)
    for pv, k in zip(pvs, kv):
        if k >= 0 and len(pv):
            alpha = max(alpha, float(pv[k]))
    if ks(alpha) != kv:                        # the smallest common level must give exactly this k-vector
        raise AssertionError(f"batch-spent level {alpha!r} does not reproduce the k-vector {kv}")
    designs = [(n, N, k) for (n, N), k in zip(ds, kv)]
    return alpha, designs, {**info, "ok": True, "delta_batch": db}


def single_look_size(N: int, eps: float, delta: float, method: str = "rectified", design_c: float = DEFAULT_DESIGN_C,
                     tune: bool = True) -> int:
    """Calibration size of the default single-look design for a batch of N (exactly what the harness uses with
    design='single', calibration='sequential', no memo): round(design_c*sqrt(N/eps)) clipped to N//2, then
    sawtooth-tuned at the full delta. Label-free."""
    looks, shares = single_look_design(int(N), eps, design_c)
    looks = list(looks)
    if tune:
        looks = tuned_looks(looks, spend(looks, shares), int(N), eps, delta, method)
    return int(looks[-1])


DEFAULT_MIN_TAX = 0.30      # minimum certified size: smallest N with planned single-look share n(N)/N <= this
MAX_MIN_TAX = 0.45          # below 4*c^2/eps items the single look is clipped to N // 2 (share ~0.5 by construction)


@lru_cache(maxsize=CACHE_MIN_SIZE)
def min_certified_size(eps: float, delta: float, design_c: float = DEFAULT_DESIGN_C, tune: bool = True,
                       method: str = "rectified", tax: float = DEFAULT_MIN_TAX, window: float = TUNE_WINDOW) -> int:
    """THE minimum certified batch size (one exact helper for BottleSession coalescing and chunked pours):
    the smallest batch size N whose planned single-look calibration share n(N)/N is <= tax, where n(N) is the
    harness's own label-free look (single_look_design + sawtooth tuning, exactly as Bottler sizes a 'single' look;
    = single_look_size). A function of (eps, delta, design_c, tune, method, tax) only: no label, score or
    certificate. Cached.
    The scan starts at N = floor(4*c^2/eps) (125 at eps=.05): below it the look is clipped to N // 2, i.e. about half
    of the batch by construction, and only integer flooring of tiny looks (N < 10) could fake a low share. That is
    also why tax must be <= MAX_MIN_TAX = 0.45 (tuning moves the clipped share within about 0.45-0.55).
    Exact and fast: tune_size never goes below lo = max(1, int(n0*(1-window))), so min(lo, N)/N is a lower bound on
    the share; every N before the first N where that bound is <= tax has share > tax and is skipped without tuning.
    Values at delta=0.1, rectified, c=1.25, tax=0.30: 355 (eps=.05), 190 (.1), 875 (.02).
    It always sizes from the single-look design, whatever cfg.design / looks / calibration select.
    session.min_certified_size is the same function. An exact scan is needed: bisecting the sawtooth-shaped share
    can land on a later crossing (400 instead of 355 at eps=.05)."""
    if not 0 < tax <= MAX_MIN_TAX:
        raise ValueError(f"tax must be in (0, {MAX_MIN_TAX}]: below 4*c^2/eps items a single look labels about half "
                         "the batch")
    if not (eps > 0 and 0 < delta < 1 and design_c > 0):
        raise ValueError("need eps > 0, 0 < delta < 1, design_c > 0")
    limit = MIN_SIZE_SEARCH_LIMIT

    def n0(N):
        return single_look_design(N, eps, design_c)[0][0]

    def lower(N):
        a = n0(N)
        lo = max(1, int(a * (1 - window))) if tune else a
        return min(lo, N) / N

    def share(N):
        looks, sp = single_look_design(N, eps, design_c)
        looks = list(looks)
        if tune:
            looks = tuned_looks(looks, spend(looks, sp), N, eps, delta, method, window=window)
        return looks[-1] / N

    N = max(1, int(math.floor(4 * design_c ** 2 / eps)))     # 4 = 1 / MAX_CAL_SHARE**2: where the clip stops binding
    while lower(N) > tax:
        N += 1
        if N > limit:
            raise ValueError("no batch size reaches this calibration tax")
    while share(N) > tax:
        N += 1
        if N > limit:
            raise ValueError("no batch size reaches this calibration tax")
    return N


def min_batch_for_tax(eps: float, delta: float, method: str = "rectified", design_c: float = DEFAULT_DESIGN_C,
                      tax: float = DEFAULT_MIN_TAX, tune: bool = True) -> int:
    """N_min of chunked pours (harness.Bottler.chunk_plan). A thin wrapper over the
    exact scan min_certified_size: the smallest N whose single-look share single_look_size(N)/N is <= tax, with tax in
    (0, MAX_MIN_TAX] (one domain for both helpers; a larger tax raises ValueError). 355 at eps=.05, delta=.1 (a
    bisection can return 400, a later crossing of the sawtooth); 190 at .1, 875 at .02.
    Tax (eps=.05, delta=.1, rectified, tuned): share 0.54 at N=50, 0.45 at 150, 0.29 at 400, 0.21 at 800, 0.17 at
    1,200."""
    if not 0 < tax <= MAX_MIN_TAX:
        raise ValueError(f"tax must be in (0, {MAX_MIN_TAX}]")
    return min_certified_size(float(eps), float(delta), float(design_c), bool(tune), str(method), float(tax))


# ------------------------------------------------------------------------------------------------ stream ledger
STREAM_TOL = 1e-13       # truncation of a design's law when its support is larger than BATCH_FULL_SUPPORT_MAX
STREAM_TRIM = 1e-16      # running stream law: top mass below this goes to overflow, bottom mass below it is moved up
STREAM_DELTAS = (0.5, 0.1, 0.01, 0.001)


def _design_key(looks) -> tuple:
    out = tuple((int(n), int(k)) for n, k in looks)
    if not out:
        raise ValueError("a design needs at least one look")
    return out


@lru_cache(maxsize=CACHE_DESIGN_LAW)
def design_law(looks: tuple, N: int, tol: float = STREAM_TOL) -> tuple[np.ndarray, float]:
    """Dominating law of the output error count U of ONE certified population of N items, from its PRE-SPECIFIED
    design looks = ((n_1, k_1), ...) (the n labelled at each look and the k its closed-form rule applies):
      one look  : chunk_law(n, N, k) (the exact worst-case law, attained; k < 0 or n >= N: U = 0 surely);
      L looks   : survival G(u) = min(1, sum_j g_j(u)) (the union bound of risk_profile: the selected look bottles at
                  most max_j U_j unlabelled bad items, whatever the stop rule did). G is non-increasing, so it is the
                  survival function of a law that dominates U.
    Full support when every look has N - n_j <= BATCH_FULL_SUPPORT_MAX (no truncation), else cut where G < tol with the
    cut mass returned as `overflow` (callers add it to every tail). Returns (read-only pmf over 0, 1, ..., overflow).
    A function of (looks, N) only: label-free. Cached (streams repeat the same few designs)."""
    looks, N = _design_key(looks), int(N)
    if len(looks) == 1:
        p, o = chunk_law(looks[0][0], N, looks[0][1], tol=tol)
        p = np.array(p, float)
    else:
        u_max = max(N - n for n, _ in looks)
        live = [(n, k) for n, k in looks if k >= 0 and N - n > 0]
        if not live or u_max <= 0:
            p, o = np.ones(1), 0.0
        else:
            full = u_max <= BATCH_FULL_SUPPORT_MAX
            if full:
                L = u_max
            else:
                L = max(len(_tail_bound(n, N, k, tol=tol)[0]) for n, k in live)
            G = np.zeros(L)
            for n, k in live:
                g, _ = _tail_bound(n, N, k, u_stop=L - 1, tol=tol)
                G[:len(g)] += np.clip(g, 0.0, 1.0)
            G = np.minimum(G, 1.0)
            G = np.minimum.accumulate(G)                         # non-increasing (true already; numerical guard)
            gm = np.concatenate([[1.0], G])
            p = np.clip(gm[:-1] - gm[1:], 0.0, None)
            if full:                                             # G(u_max) = 0: the rest sits at u_max
                p, o = np.append(p, gm[-1]), 0.0
            else:
                o = float(gm[-1])
    p = np.asarray(p, float)
    p.setflags(write=False)
    return p, float(o)


class StreamLaw:
    """Running worst-case law of the output error count W = U_1 + U_2 + ... of a STREAM of certified batches (or
    chunks), each certified by its own pre-specified closed-form k-rule(s) on a fresh uniform calibration sample.
    Report-only: nothing here feeds back into any k, n, threshold or answer.

    Claim (the batch_tail proof, applied across batches). Suppose the designs (n_b, N_b, k_b) of the included batches
    and the horizon B are fixed independently of every output (flush sizes from the submission process and label-free
    rules only). F_{b-1} = everything up to the end of batch b-1 (its expert labels included); batch b's scores and
    answers are F_{b-1}-measurable and its calibration sample is a fresh uniform draw independent of F_{b-1}, so
    P[U_b > u | F_{b-1}] <= G_b(u) for every u (design_law). With h(x) = P[V_b > t - x] non-decreasing,
        P[S_b > t] = E[P[U_b > t - S_{b-1} | F_{b-1}]] <= E[h(S_{b-1})] <= E[h(W_{b-1})] = P[W_{b-1} + V_b > t],
    so P[U_1 + ... + U_B > u] <= P[V_1 + ... + V_B > u] for every u at once, V_b independent with the laws design_law.
    The law of the sum is the convolution kept here. Batch contents may depend on earlier outputs; designs may not.

    Numerics: each design's law is exact (full support) up to BATCH_FULL_SUPPORT_MAX unlabelled items, else cut at
    tol with the cut mass added to `overflow`. After each convolution the top entries holding < trim mass in total
    are cut to `overflow` and the bottom entries holding < trim mass are MOVED UP to the first kept entry
    (conservative: it can only raise P[W > u]). tail(u) = P[W_kept > u] + overflow (an upper bound for every u).
    Persisted as the list of designs only (to_plain); from_plain rebuilds the same floats by re-adding them in order."""

    def __init__(self, tol: float = STREAM_TOL, trim: float = STREAM_TRIM):
        self.tol, self.trim = float(tol), float(trim)
        self.designs: list = []           # [[[n, k], ...], N] per added design, in order
        self.N = 0                        # items covered (sum of the designs' populations)
        self.overflow = 0.0
        self.expected_closed_form = 0.0   # sum over looks of (k+1)(N-n)/(n+1): >= E[W] (= for single looks)
        self._pmf = np.ones(1)
        self._off = 0
        self._surv = None

    # ------------------------------------------------------------------ building
    def add_design(self, looks, N: int) -> "StreamLaw":
        """Add one certified population of N items with pre-specified looks ((n, k), ...)."""
        key, N = _design_key(looks), int(N)
        if N <= 0:
            raise ValueError("a design needs N > 0")
        p, o = design_law(key, N, self.tol)
        w = np.convolve(self._pmf, p)
        over = self.overflow + o
        if len(w) > 1 and self.trim > 0:
            top = np.cumsum(w[::-1])
            cut = min(int(np.searchsorted(top, self.trim, side="left")), len(w) - 1)
            if cut:
                over += float(top[cut - 1])
                w = w[:len(w) - cut]
            low = np.cumsum(w)
            sh = min(int(np.searchsorted(low, self.trim, side="left")), len(w) - 1)
            if sh:
                w = w[sh:].copy()
                w[0] += float(low[sh - 1])
                self._off += sh
        self._pmf, self.overflow = w, over
        self.designs.append([[list(x) for x in key], N])
        self.N += N
        self.expected_closed_form += float(sum(expected_eps_closed_form(n, N, k) * N for n, k in key))
        self._surv = None
        return self

    def add(self, n: int, N: int, k: int) -> "StreamLaw":
        """One single-look design (n, N, k)."""
        return self.add_design(((n, k),), N)

    def add_designs(self, designs) -> "StreamLaw":
        """Several single-look designs [(n, N, k), ...] (e.g. the chunks of a chunked pour: one entry per chunk)."""
        for n, N, k in designs:
            self.add(n, N, k)
        return self

    # ------------------------------------------------------------------ queries
    def _survival(self) -> np.ndarray:
        if self._surv is None:
            w = self._pmf
            self._surv = np.concatenate([np.cumsum(w[::-1])[::-1][1:], [0.0]])   # s[i] = P[W_kept > off + i]
        return self._surv

    def tail(self, u) -> float:
        """Bound on P[W > u] (W = total output error count of the stream)."""
        u = int(math.floor(u))
        if u < 0:
            return 1.0
        s = self._survival()
        i = u - self._off
        if i < 0:
            return float(min(1.0, float(self._pmf.sum()) + self.overflow))
        return float(min(1.0, (s[i] if i < len(s) else 0.0) + self.overflow))

    def survival(self) -> tuple[int, np.ndarray]:
        """(offset, s) with s[i] = bound on P[W > offset + i]; below offset the bound is the total mass (about 1)."""
        return self._off, np.minimum(1.0, self._survival() + self.overflow)

    def u_at_delta(self, d: float) -> int | None:
        """Smallest u with tail(u) <= d (None if no u qualifies, i.e. overflow > d)."""
        s = self._survival() + self.overflow
        hit = np.nonzero(s <= float(d))[0]
        if not len(hit):
            return None
        return int(self._off + hit[0])

    def eps_at_delta(self, deltas=STREAM_DELTAS, N: int | None = None) -> dict:
        """{d: smallest u/N with P[W/N > u/N] <= d}; N defaults to the covered items."""
        N = self.N if N is None else int(N)
        out = {}
        for d in sorted({float(x) for x in deltas}, reverse=True):
            u = self.u_at_delta(d)
            out[f"{d:g}"] = (1.0 if u is None else u / N) if N else 0.0
        return out

    def mean(self) -> float:
        """Mean of the kept part of the dominating law (<= the closed-form sum; equal up to trims for single looks)."""
        w = self._pmf
        return float(np.dot(np.arange(self._off, self._off + len(w)), w))

    def summary(self, eps: float | None = None, deltas=STREAM_DELTAS, eps_grid=RISK_EPS_GRID,
                n_unusable: int = 0) -> dict:
        """Report dict. Shares are over the N covered items; F = n_unusable (expert-routed items without a usable expert
        answer, observed) is ADDED to every share bound (never folded into the tail): the stream share of final
        answers != expert is <= W/N + F/N."""
        N, F = self.N, int(n_unusable)
        if N <= 0:
            return {"N": 0, "designs": 0}
        fs = F / N
        eu = self.eps_at_delta(deltas, N)
        return {"basis": "convolution of per-batch worst-case laws", "N": N, "designs": len(self.designs),
                "n_expert_unusable": F, "unusable_share": fs,
                "delta_stream": self.tail(math.floor(eps * N)) if eps is not None else None, "eps": eps,
                "tail": {f"{e:g}": self.tail(math.floor(e * N)) for e in eps_grid},
                "eps_at_delta_U": eu,
                "eps_at_delta": {d: min(1.0, v + fs) for d, v in eu.items()},
                "expected_errors_bound": self.expected_closed_form,
                "expected_share_bound": self.expected_closed_form / N,
                "overflow": float(self.overflow), "support": [int(self._off), int(self._off + len(self._pmf) - 1)],
                "tol": self.tol, "trim": self.trim}

    # ------------------------------------------------------------------ persistence (designs only)
    def to_plain(self) -> dict:
        return {"v": 1, "tol": self.tol, "trim": self.trim,
                "designs": [[[list(x) for x in d[0]], int(d[1])] for d in self.designs]}

    @classmethod
    def from_plain(cls, d: dict) -> "StreamLaw":
        if int(d.get("v", 0)) != 1:
            raise ValueError("unknown StreamLaw format")
        law = cls(tol=float(d["tol"]), trim=float(d["trim"]))
        for looks, N in d["designs"]:
            law.add_design(looks, N)
        return law
