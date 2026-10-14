"""Tests for the RECTIFIED (outgoing-quality) certificate: calibration items are answered by the expert, so the test
bounds only the unlabelled bad items that are bottled. No servers needed.

  * exact validity by exhaustive enumeration of every calibration subset (N <= 16, all M, distinct and tied scores)
  * adversarial Monte Carlo at bench sizes (N = 400; n in {94, 203, 217}), bad items at the top
  * never stricter than v0.2 ('hypergeometric') on a grid; 'hypergeometric' and 'binomial' unchanged
  * the rectified budget inversion (single and staged) covers the realized output error
  * staged (sequential) validity under adversarial workloads, and pathwise dominance over v0.2
  * strict thresholds under ties, edge cases (cummax, impossible sizes, accept-all), harness wiring and reporting
"""

import asyncio
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import binom, hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottlekit import NONE, BottleConfig, Bottler, certified_eps, certify, evaluate, fixed_sequence  # noqa: E402
from bottlekit.certify import (_pvalues, certify_sequential, default_design, k_critical,  # noqa: E402
                               sequential_certified_eps, spend, tune_size, tuned_looks)
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402


# ----------------------------------------------------------------------------------------------- helpers
def _v(n, N, eps):
    """Reference (independent) computation of v_j = HypCDF(j; N, m+1+j, n), 0 for impossible sizes."""
    m = int(math.floor(eps * N))
    return np.array([hypergeom.cdf(j, N, m + 1 + j, n) if m + 1 + j <= N else 0.0 for j in range(n + 1)])


def _errors_vec(cal, bad, score, k):
    """Unlabelled bad items bottled by the closed-form k-rule, for every row of the boolean calibration matrix `cal`
    (vectorised re-implementation of fixed_sequence; cross-checked against it in test_vectorised_rule_matches...)."""
    order = np.argsort(-score, kind="stable")
    bidx = order[bad[order]]
    bs = score[bidx]
    L = cal[:, bidx]
    if L.shape[1] == 0:
        return np.zeros(len(cal), int)
    hit = np.cumsum(L, axis=1) >= k + 1
    has = hit.any(axis=1)
    sstar = np.where(has, bs[hit.argmax(axis=1)], -np.inf)
    acc = (bs[None, :] > sstar[:, None]) | (~has)[:, None]           # strictly above the (k+1)-th labelled bad item
    return ((~L) & acc).sum(axis=1)


_SUBSETS = {}


def _subsets(N, n):
    key = (N, n)
    if key not in _SUBSETS:
        rows = np.zeros((math.comb(N, n), N), bool)
        for r, c in enumerate(itertools.combinations(range(N), n)):
            rows[r, list(c)] = True
        _SUBSETS[key] = rows
    return _SUBSETS[key]


def _tie_patterns(N, rng):
    """Score vectors over N items: all distinct, all tied, pairs, triples, random coarse grid."""
    return {"distinct": np.linspace(1, 0, N),
            "all_tied": np.full(N, 0.5),
            "pairs": np.repeat(np.linspace(1, 0, (N + 1) // 2), 2)[:N],
            "triples": np.repeat(np.linspace(1, 0, (N + 2) // 3), 3)[:N],
            "grid": rng.integers(0, 4, N) / 3.0}


# ----------------------------------------------------------------------------------------------- exact validity
def test_rectified_levels_are_cummax_of_v_and_monotone():
    for (n, N, eps) in [(5, 12, 0.2), (94, 400, 0.05), (217, 400, 0.05), (217, 400, 0.1), (327, 1200, 0.05),
                        (16, 16, 0.3), (40, 50, 0.5), (10, 10, 1.0)]:
        p = _pvalues(n, N, eps, "rectified")
        v = _v(n, N, eps)
        assert np.all(np.diff(p) >= 0)
        np.testing.assert_allclose(p, np.maximum.accumulate(v), rtol=0, atol=1e-15)
    # Structure: over attainable sizes v_j is non-decreasing (coupling), and impossible sizes (m+1+j > N) only occur
    # when n >= N-m, where no error is possible at all; so the running max is a safeguard, never binding.
    for N in (10, 16, 40, 400):
        for n in range(1, N + 1, max(1, N // 20)):
            for eps in (0.05, 0.2, 0.5):
                m = int(math.floor(eps * N))
                v = _v(n, N, eps)
                assert np.all(np.diff(v) >= -1e-12), (N, n, eps)
                if m + 1 + n > N:
                    assert n >= N - m and np.all(v == 0)


def test_vectorised_rule_matches_fixed_sequence_on_every_subset():
    """Ties the exhaustive checks below to the real implementation: for N=10, every calibration subset, every M and
    several tie patterns (good items interleaved/tied with bad ones), the number of bottled unlabelled bad items from
    fixed_sequence(method='rectified') equals the vectorised rule."""
    rng = np.random.default_rng(0)
    N = 10
    for name, score in _tie_patterns(N, rng).items():
        for n in (3, 5, 7):
            cal = _subsets(N, n)
            for M in (0, 2, 4, 7, 10):
                bad = np.zeros(N, bool)
                bad[rng.choice(N, M, replace=False)] = True
                for eps, alpha in ((0.1, 0.3), (0.2, 0.2), (0.3, 0.1)):
                    k = k_critical(n, N, eps, alpha, "rectified")
                    vec = _errors_vec(cal, bad, score, k) if k >= 0 else np.zeros(len(cal), int)
                    for r in range(len(cal)):
                        idx = np.nonzero(cal[r])[0]
                        tau, _ = fixed_sequence(score[idx], bad[idx], N, eps, alpha, "rectified")
                        acc = score >= tau
                        acc[idx] = False
                        assert int((acc & bad).sum()) == int(vec[r]), (name, n, M, eps, alpha, r)


@pytest.mark.parametrize("N", [10, 13, 16])
def test_exhaustive_enumeration_exact_validity(N):
    """P[more than floor(eps N) unlabelled bad items bottled] <= alpha EXACTLY, over every n-subset, every number of
    bad items M and every tie pattern; and without ties the worst case over M equals the closed form (the test is
    tight: max_j<=k v_j is attained)."""
    rng = np.random.default_rng(N)
    pats = _tie_patterns(N, rng)
    eps_grid = (0.1, 0.2, 0.3)
    alpha_grid = (0.05, 0.1, 0.2, 0.3) if N < 16 else (0.05, 0.2)
    if N == 16:
        pats = {k: pats[k] for k in ("distinct", "pairs", "grid")}
    worst_ratio = 0.0
    for n in range(1, N):
        cal = _subsets(N, n)
        for eps in eps_grid:
            m = int(math.floor(eps * N))
            for alpha in alpha_grid:
                k = k_critical(n, N, eps, alpha, "rectified")
                if k < 0:
                    continue
                for name, score in pats.items():
                    order = np.argsort(-score, kind="stable")
                    worst = 0.0
                    for M in range(N + 1):
                        bad = np.zeros(N, bool)
                        bad[order[:M]] = True                      # bad items at the top (any placement: same law)
                        p_err = float(np.mean(_errors_vec(cal, bad, score, k) > m))
                        assert p_err <= alpha + 1e-12, (N, n, eps, alpha, name, M, p_err)
                        worst = max(worst, p_err)
                    if name == "distinct":
                        assert worst == pytest.approx(_pvalues(n, N, eps, "rectified")[k], abs=1e-9)
                    worst_ratio = max(worst_ratio, worst / alpha)
    assert worst_ratio > 0.5, "enumeration never came close to alpha: the check would be vacuous"


def test_exhaustive_random_bad_placement_with_ties():
    """Bad items placed anywhere (not only at the top) and good items tied with them: still <= alpha exactly."""
    rng = np.random.default_rng(11)
    N = 13
    for _ in range(60):
        score = rng.integers(0, 5, N) / 4.0
        M = int(rng.integers(1, N + 1))
        bad = np.zeros(N, bool)
        bad[rng.choice(N, M, replace=False)] = True
        n = int(rng.integers(2, N))
        eps, alpha = float(rng.choice([0.1, 0.2, 0.3])), float(rng.choice([0.05, 0.1, 0.2]))
        k = k_critical(n, N, eps, alpha, "rectified")
        if k < 0:
            continue
        p_err = float(np.mean(_errors_vec(_subsets(N, n), bad, score, k) > math.floor(eps * N)))
        assert p_err <= alpha + 1e-12


# ----------------------------------------------------------------------------------------------- bench-size Monte Carlo
@pytest.mark.parametrize("n,alpha", [(94, 0.02), (203, 0.08), (217, 0.08)])
def test_adversarial_monte_carlo_bench_sizes(n, alpha):
    """N=400, eps=.05: bad items occupy the top M score ranks (the worst case). The violation rate of the REAL
    fixed_sequence must match the exact closed form and stay <= alpha."""
    rng = np.random.default_rng(n)
    N, eps, R = 400, 0.05, 3000
    m = int(math.floor(eps * N))
    k = k_critical(n, N, eps, alpha, "rectified")
    assert k >= k_critical(n, N, eps, alpha, "hypergeometric")
    score = np.linspace(1, 0, N)
    worst_seen = 0.0
    for M in sorted({m + 1, m + 1 + k // 2, m + 1 + k, m + 6 + k, min(N, m + 1 + 2 * k + 10)}):
        bad = np.zeros(N, bool)
        bad[:M] = True
        viol = 0
        for _ in range(R):
            idx = rng.choice(N, n, replace=False)
            tau, _ = fixed_sequence(score[idx], bad[idx], N, eps, alpha, "rectified")
            acc = score >= tau
            acc[idx] = False
            viol += int((acc & bad).sum()) > m
        T = min(M, m + 1 + k)
        exact = float(hypergeom.cdf(T - m - 1, N, T, n))
        sd = math.sqrt(max(exact * (1 - exact), 1e-4) / R)
        assert abs(viol / R - exact) < 4.5 * sd + 1e-3, (M, viol / R, exact)
        assert exact <= alpha + 1e-12
        worst_seen = max(worst_seen, viol / R)
    assert worst_seen <= alpha + 3.5 * math.sqrt(alpha * (1 - alpha) / R)


def test_random_workload_validity_fixed_rule():
    """Smooth random workloads at N=400 (high calibration fraction): violation rate of the output <= delta."""
    rng = np.random.default_rng(3)
    N, n, eps, delta, R = 400, 217, 0.05, 0.1, 1500
    viol = 0
    for _ in range(R):
        score = rng.random(N)
        bad = rng.random(N) < 0.02 + 0.5 * (1 - score) ** 3
        idx = rng.choice(N, n, replace=False)
        tau, _ = fixed_sequence(score[idx], bad[idx], N, eps, delta, "rectified")
        acc = score >= tau
        acc[idx] = False
        viol += (acc & bad).sum() > math.floor(eps * N)
    assert viol / R <= delta + 3 * math.sqrt(delta * (1 - delta) / R)


# ----------------------------------------------------------------------------------------------- dominance + regressions
def test_rectified_never_stricter_than_hypergeometric():
    for N in (20, 50, 150, 400, 1200, 3000):
        for n in sorted({5, 30, 94, 100, 217, 300, 327, min(N - 1, 900), N // 2, N - 1}):
            if not 1 <= n <= N:
                continue
            for eps in (0.01, 0.02, 0.05, 0.1, 0.2):
                for alpha in (0.02, 0.05, 0.08, 0.1):
                    kr, kh = k_critical(n, N, eps, alpha, "rectified"), k_critical(n, N, eps, alpha, "hypergeometric")
                    assert kr >= kh, (N, n, eps, alpha, kr, kh)
    # the gain shrinks as n/N -> 0 (asymptotically the same test) and is large when n/N is large
    assert k_critical(217, 400, 0.05, 0.08, "rectified") >= 2 * k_critical(217, 400, 0.05, 0.08, "hypergeometric")
    assert k_critical(300, 200_000, 0.05, 0.08, "rectified") - k_critical(300, 200_000, 0.05, 0.08, "hypergeometric") <= 1


def test_hypergeometric_and_binomial_unchanged():
    """The v0.2 methods are bit-identical to an independent reference implementation."""
    rng = np.random.default_rng(4)
    for _ in range(200):
        N = int(rng.integers(20, 5000))
        n = int(rng.integers(1, min(N, 600) + 1))
        eps, alpha = float(rng.choice([0.01, 0.02, 0.05, 0.1, 0.3])), float(rng.choice([0.02, 0.08, 0.1]))
        ks = np.arange(n + 1)
        M0 = int(math.floor(eps * N)) + 1
        ph = np.zeros(n + 1) if M0 > N else hypergeom.cdf(ks, N, M0, n)
        pb = binom.cdf(ks, n, eps)
        for meth, p in (("hypergeometric", ph), ("binomial", pb)):
            ok = np.nonzero(p <= alpha)[0]
            assert k_critical(n, N, eps, alpha, meth) == (int(ok.max()) if len(ok) else -1)
            np.testing.assert_array_equal(_pvalues(n, N, eps, meth), p)
        s, b = rng.random(n), rng.random(n) < 0.2
        tau = float(rng.random())
        k = int(((s >= tau) & b).sum())
        lin = next(((M - 1) / N for M in range(k, N + 1) if hypergeom.cdf(k, N, M, n) <= alpha), 1.0)
        assert certified_eps(s, b, tau, N, alpha) == pytest.approx(lin, abs=0)
    with pytest.raises(ValueError):
        k_critical(10, 100, 0.05, 0.1, "rectified-binomial")
    with pytest.raises(ValueError):
        certified_eps([0.5], [True], 0.1, 100, 0.1, "nope")


# ----------------------------------------------------------------------------------------------- ties, edges
def test_tie_at_the_critical_score_is_not_bottled():
    N, eps, alpha = 400, 0.05, 0.08
    n = 217
    k = k_critical(n, N, eps, alpha, "rectified")
    rng = np.random.default_rng(5)
    s = np.round(rng.random(n), 1)                                   # heavy ties
    b = np.zeros(n, bool)
    hi = np.argsort(-s, kind="stable")
    b[hi[: k + 3]] = True                                            # k+3 labelled bad items at the top scores
    tau, _ = fixed_sequence(s, b, N, eps, alpha, "rectified")
    crit = np.sort(s[b])[::-1][k]
    assert tau > crit and tau == np.nextafter(crit, np.inf)
    assert not np.any((s == crit) & (s >= tau)), "an item tied with the critical labelled bad item was bottled"
    assert int((b & (s >= tau)).sum()) <= k


def test_edge_cases():
    # accept-all when at most k labelled bad items
    n, N = 217, 400
    k = k_critical(n, N, 0.05, 0.08, "rectified")
    s = np.linspace(1, 0, n)
    b = np.zeros(n, bool)
    b[:k] = True
    assert fixed_sequence(s, b, N, 0.05, 0.08, "rectified")[0] == -np.inf
    b[k] = True
    assert np.isfinite(fixed_sequence(s, b, N, 0.05, 0.08, "rectified")[0])
    # eps >= 1: no workload can err; everything passes. Whole workload labelled: nothing left to err.
    assert k_critical(30, 40, 1.0, 0.05, "rectified") == 30
    assert k_critical(40, 40, 0.05, 0.05, "rectified") == 40
    # impossible sizes m+1+j > N contribute 0, and then fewer than m+1 items are unlabelled: everything passes
    assert np.all(_pvalues(9, 10, 0.5, "rectified") == 0) and k_critical(9, 10, 0.5, 0.01, "rectified") == 9
    # nothing labelled / tiny eps
    assert k_critical(0, 100, 0.05, 0.1, "rectified") == -1
    assert fixed_sequence([], [], 100, 0.05, 0.1, "rectified")[0] == np.inf
    assert k_critical(50, 100_000, 0.00001, 0.1, "rectified") == -1


# ----------------------------------------------------------------------------------------------- budget inversion
def test_rectified_budget_inversion_exact_coverage():
    """For a label-free tau, P[M > M_upper] <= delta for EVERY M (exact sum over the hypergeometric law); the
    rectified bound subtracts the labelled bad items, which are exactly the ones the expert corrects."""
    for (N, n, delta) in [(400, 217, 0.1), (400, 94, 0.02), (1200, 327, 0.08), (60, 30, 0.1)]:
        mu = {k: certified_eps(np.ones(k), np.ones(k, bool), 0.5, N, delta, "rectified", k_end=k) * N + k
              for k in range(n + 1)}                                  # M_upper(k) = rect bound * N + k
        for M in range(0, N + 1, max(1, N // 80)):
            ks = np.arange(0, min(n, M) + 1)
            pmf = hypergeom.pmf(ks, N, M, n)
            fail = sum(p for kk, p in zip(ks, pmf) if mu[int(kk)] < M - 1e-9)
            assert fail <= delta + 1e-9, (N, n, M, fail)


def test_rectified_budget_inversion_monte_carlo_with_k_end():
    """Staged labelling (two looks) + a label-free budget threshold: (unlabelled bad bottled)/N <= eps_hat in
    >= 1 - delta of runs, using sequential_certified_eps (K_end from ALL labels)."""
    rng = np.random.default_rng(6)
    N, delta, R = 400, 0.1, 600
    looks = (94, 217)
    cover, gaps = 0, []
    for r in range(R):
        score = rng.random(N)
        bad = rng.random(N) < 0.45 * (1 - score)
        answer = np.where(bad, "x", "y").tolist()
        order = np.random.default_rng(r).permutation(N)
        labels = {int(i): "y" for i in order[:looks[-1]]}
        tau_b = float(np.sort(score)[150])
        eps_hat = sequential_certified_eps(score, answer, labels, r, looks, tau_b, N, delta, method="rectified",
                                           spending=(1, 4))
        eps_old = sequential_certified_eps(score, answer, labels, r, looks, tau_b, N, delta, spending=(1, 4))
        acc = score >= tau_b
        acc[list(labels)] = False
        realized = (acc & bad).sum() / N
        cover += realized <= eps_hat
        gaps.append(eps_old - eps_hat)
    assert cover / R >= 1 - delta - 0.03
    assert min(gaps) >= -1e-12 and np.mean(gaps) > 0.02, "rectified bound should be tighter at n/N = .54"
    with pytest.raises(ValueError):
        certified_eps([0.9, 0.8], [True, True], 0.5, 100, 0.1, "rectified", k_end=1)


# ----------------------------------------------------------------------------------------------- staged validity
def _run_seq(score, answer, N, eps, delta, looks, spending, seed, method):
    async def expert(idx):
        return {i: "y" for i in idx}
    return asyncio.run(certify_sequential(score, answer, expert, N, eps, delta, looks, seed=seed, method=method,
                                          spending=spending))


@pytest.mark.parametrize("N,case", [(400, "smooth"), (400, "top"), (1200, "top")])
def test_staged_rectified_validity(N, case):
    """Default staged design (rectified-tuned looks, Bonferroni 1:4): output violation rate <= delta, including the
    adversarial 'bad items at the top' workload sized at the worst case of the second look."""
    eps, delta = 0.05, 0.1
    looks0, w = default_design(N)
    shares = spend(looks0, w)
    looks = tuned_looks(looks0, shares, N, eps, delta, "rectified")
    m = int(math.floor(eps * N))
    rng = np.random.default_rng(N + len(case))
    R = 1200
    viol = 0
    k2 = k_critical(looks[-1], N, eps, delta * shares[-1], "rectified")
    for r in range(R):
        if case == "smooth":
            score = rng.random(N)
            bad = rng.random(N) < 0.02 + 0.5 * (1 - score) ** 3
        else:
            score = np.linspace(1, 0, N)
            bad = np.zeros(N, bool)
            bad[: m + 1 + k2] = True
        answer = np.where(bad, "x", "y").tolist()
        cert, lab = _run_seq(score, answer, N, eps, delta, looks, w, r, "rectified")
        assert "rectified" in cert.method
        acc = score >= cert.tau
        acc[list(lab)] = False
        viol += (acc & bad).sum() > m
    rate = viol / R
    assert rate <= delta + 3 * math.sqrt(delta * (1 - delta) / R), rate
    if case == "top":
        assert rate > 0.02, "the adversarial case should actually stress the test (otherwise this check is vacuous)"


def test_pathwise_rectified_bottles_at_least_as_much():
    """Same seed, same look sizes: rectified bottles >= hypergeometric on EVERY path (each look's k only grows and
    the dominance stop only stops at a value no later look can beat)."""
    rng = np.random.default_rng(8)
    for r in range(150):
        N = int(rng.choice([400, 1200]))
        eps = float(rng.choice([0.02, 0.05, 0.1]))
        score = rng.random(N)
        bad = rng.random(N) < float(rng.uniform(0.05, 0.6)) * (1 - score) ** float(rng.uniform(1, 3))
        answer = np.where(bad, "x", "y").tolist()
        looks = (100, 300) if r % 2 else (94, 217)
        out = {}
        for meth in ("hypergeometric", "rectified"):
            cert, lab = _run_seq(score, answer, N, eps, 0.1, looks, (1, 4), r, meth)
            acc = score >= cert.tau
            acc[list(lab)] = False
            out[meth] = int(acc.sum())
        assert out["rectified"] >= out["hypergeometric"], (r, out)


# ----------------------------------------------------------------------------------------------- harness wiring
class NoneExpertClient(MockClient):
    """Mock whose expert gives an unusable reply on a fixed subset of items (outage / parser miss)."""

    def __init__(self, gold, none_texts, **kw):
        super().__init__(gold, **kw)
        self.none_texts = set(none_texts)

    def _answer(self, alias, text, salt, pos=None, k=None):
        if alias == "expert" and pos is None and text in self.none_texts:
            return "I am not sure."
        return super()._answer(alias, text, salt, pos, k)


def test_config_validation_and_method_strings():
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", certificate="binomial")    # never rectify / use future-items test here
    assert BottleConfig(cheap="c", expert="e").certificate == "rectified"
    c = certify(np.linspace(1, 0, 50), np.zeros(50, bool), 100, 0.05, 0.1, "rectified")
    assert c.method == "rectified" and c.certified


def test_harness_rectified_bottles_at_least_hypergeometric_end_to_end():
    """Same workload/seed, same (untuned) looks: the default rectified harness bottles >= the v0.2 certificate."""
    wl = toy_workload(400, seed=2)
    gold = {t[:400]: g for t, g in zip(wl.texts, wl.gold)}
    cov = {}
    for meth in ("hypergeometric", "rectified"):
        cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, eps=0.05, seed=3, certificate=meth,
                           tune=False, looks=(94, 217), spending=(1, 4))
        res = asyncio.run(Bottler(MockClient(gold, acc={"cheap": 0.75, "expert": 0.95}), SPEC, cfg).run(wl.texts))
        assert ("rectified" in res.certificate["method"]) == (meth == "rectified")
        assert res.certificate["eps_certified"] == pytest.approx(0.05) and res.certificate["n_expert_unusable"] == 0
        cov[meth] = res.coverage()
    assert cov["rectified"] >= cov["hypergeometric"]


def test_harness_counts_unusable_expert_answers_in_calibration_and_escalation():
    wl = toy_workload(400, seed=4)
    gold = {t[:400]: g for t, g in zip(wl.texts, wl.gold)}
    none_texts = wl.texts[::25]                                      # 16 items: some calibrated, some escalated
    client = NoneExpertClient(gold, none_texts, acc={"cheap": 0.75, "expert": 0.95})
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, eps=0.05, seed=1)
    res = asyncio.run(Bottler(client, SPEC, cfg).run(wl.texts))
    nset = set(none_texts)
    routed = [i for i, p in enumerate(res.provenance) if p.startswith("expert") and wl.texts[i] in nset]
    assert res.certificate["n_expert_unusable"] == len(routed) > 0
    kinds = {res.provenance[i].split("+")[0] for i in routed}
    assert res.certificate["eps_certified"] == pytest.approx(0.05 + len(routed) / 400)
    assert "expert-routed" in res.certificate["guarantee"]
    # the shortfall is exactly these items: realized disagreement - F/N = bottled errors / N
    expert_all = [NONE if t in nset else client._answer("expert", t, "single") for t in wl.texts]
    ev = evaluate(res, wl.gold, expert_all)
    bot_err = sum(1 for i, p in enumerate(res.provenance) if p == "bottle" and res.answers[i] != expert_all[i])
    assert ev["realized_disagreement_with_expert"] * 400 == pytest.approx(bot_err + len(routed))
    assert kinds <= {"expert_calibration", "expert_escalation"}


def test_harness_budget_path_rectified():
    wl = toy_workload(600)
    gold = {t[:400]: g for t, g in zip(wl.texts, wl.gold)}
    out = {}
    for meth in ("hypergeometric", "rectified"):
        cfg = BottleConfig(cheap="cheap", expert="expert", mode="student", k=20, n_build=300, calibration="fixed",
                           n_cal=150, eps=0.01, expert_budget=50, seed=1, certificate=meth)
        res = asyncio.run(Bottler(MockClient(gold, acc={"cheap": 0.7, "expert": 0.95}), SPEC, cfg).run(wl.texts))
        assert sum(p == "expert_escalation" for p in res.provenance) <= 50
        out[meth] = res.certificate
    if out["rectified"].get("budget_capped") and out["hypergeometric"].get("budget_capped"):
        assert out["rectified"]["eps_achieved"] <= out["hypergeometric"]["eps_achieved"] + 1e-12
        assert out["rectified"]["eps_certified"] >= out["rectified"]["eps_achieved"]


def test_tuned_sizes_are_label_free_and_method_specific():
    for N in (400, 1200):
        looks, w = default_design(N)
        sh = spend(looks, w)
        a = tuned_looks(looks, sh, N, 0.05, 0.1, "rectified")
        assert a == tuned_looks(looks, sh, N, 0.05, 0.1, "rectified")     # deterministic from (N, eps, delta, method)
        assert all(x < y for x, y in zip(a, a[1:]))
        assert tune_size(300, N, 0.05, 0.08, "rectified") in range(270, 331)
