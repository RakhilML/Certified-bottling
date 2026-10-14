"""Tests for the exact RISK PROFILE of a certificate (certify.risk_profile; certificate['risk']) and the optional
stricter-only mean-risk cap (BottleConfig.max_expected_eps). No servers needed.

The closed-form k-rule bottles U unlabelled bad items with P[U > u] <= g_k(u) = max_{j<=k} HypCDF(j; N, u+1+j, n) for
EVERY u, so E[U] <= sum_u g_k(u) = (k+1)(N-n)/(n+1).
  (a) the closed form equals the numeric sum (1e-9 relative); the fast recurrence equals scipy pointwise
  (b) exhaustive enumeration (N <= 16): every calibration subset, every bad set (N = 10) / bad prefix (N = 13, 16),
      tied scores, |B| < m+1+k: P[U > u] <= g(u) for every u and E[U] <= sum g; tight for the all-bad workload
  (c) Monte Carlo at bench sizes (2,000 draws): empirical tail within 3 MC-se of g, E[U]/N within 3 se of the bound
  (d) default harness certificates: 0.0300 (N=400) / 0.0334 (N=1200) at eps=.05, delta=.1; delta_spent = v_{k_c}
  (e) speed: <= 50 ms at N = 100,000
  non-default paths: staged/custom ('union bound', >= every single look), fixed, hypergeometric, not certified (U=0),
  budget ('inversion' only, no expectation), F/N reported separately; max_expected_eps (k <= k_c, label-free, both
  constraints hold by enumeration, default unchanged)
"""

import argparse
import asyncio
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottlekit import BottleConfig, Bottler, fixed_sequence, risk_profile  # noqa: E402
from bottlekit.certify import (_rectified_levels, _tail_bound, expected_eps_closed_form, k_critical,  # noqa: E402
                               k_mean_cap, k_used, single_look_design, spend, tune_size, tuned_looks)
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402
from test_bottlekit_v3 import NoneExpertClient  # noqa: E402

DELTA = 0.1


# ----------------------------------------------------------------------------------------------- helpers
def _g_ref(n, N, k, u_max):
    """Independent scipy computation of g(u) = max_{j<=k} HypCDF(j; N, u+1+j, n) (0 for impossible sizes)."""
    if k < 0:
        return np.zeros(u_max)
    return np.array([max((hypergeom.cdf(j, N, u + 1 + j, n) if u + 1 + j <= N else 0.0) for j in range(k + 1))
                     for u in range(u_max)])


def _g_full(n, N, k):
    """g(u) for u = 0..N (exact zeros where the implementation stopped or U cannot exceed u)."""
    g, _ = _tail_bound(n, N, k, u_stop=N)
    out = np.zeros(N + 1)
    out[:len(g)] = g
    return out


def _errors_vec(cal, bad, score, k):
    """Unlabelled bad items bottled by the closed-form k-rule, for every row of the boolean calibration matrix `cal`
    (vectorised re-implementation of fixed_sequence; cross-checked against it below)."""
    order = np.argsort(-score, kind="stable")
    bidx = order[bad[order]]
    if k < 0:
        return np.zeros(len(cal), int)
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
    if (N, n) not in _SUBSETS:
        rows = np.zeros((math.comb(N, n), N), bool)
        for r, c in enumerate(itertools.combinations(range(N), n)):
            rows[r, list(c)] = True
        _SUBSETS[N, n] = rows
    return _SUBSETS[N, n]


def _patterns(N, rng):
    return {"distinct": np.linspace(1, 0, N),
            "all_tied": np.full(N, 0.5),
            "pairs": np.repeat(np.linspace(1, 0, (N + 1) // 2), 2)[:N],
            "grid": rng.integers(0, 4, N) / 3.0}


def _check_law(U, g, label):
    """U: error counts, one per (equally likely) calibration subset. P[U > u] <= g(u) for every u; E[U] <= sum g."""
    U = np.asarray(U, int)
    surv = 1.0 - np.cumsum(np.bincount(U, minlength=len(g)))[:len(g)] / len(U)      # P[U > u], u = 0..len(g)-1
    bad = np.nonzero(surv > g + 1e-12)[0]
    assert len(bad) == 0, (label, int(bad[0]), surv[bad[0]], g[bad[0]]) if len(bad) else None
    assert float(U.mean()) <= g.sum() + 1e-9, (label, U.mean(), g.sum())


# ----------------------------------------------------------------------------------------------- (a) numerics
@pytest.mark.parametrize("n,N,k", [(117, 400, 4), (199, 1200, 7), (94, 400, 0), (217, 400, 9), (327, 1200, 12),
                                   (54, 100, 30), (5, 12, 2), (8, 16, 7), (1, 2, 0), (1, 2, 1), (15, 16, 15),
                                   (40, 50, 20), (300, 3000, 9), (600, 10_000, 25)])
def test_closed_form_equals_numeric_sum_and_scipy_pointwise(n, N, k):
    g, rest = _tail_bound(n, N, k)                               # default truncation (g < 1e-12)
    cf = expected_eps_closed_form(n, N, k) * N
    assert (g.sum() + rest) == pytest.approx(cf, rel=1e-9, abs=1e-12)
    assert rest <= (N - n) * 1e-12 + 1e-15
    gf, restf = _tail_bound(n, N, k, u_stop=N)                   # no truncation
    assert len(gf) == N - n and restf == 0.0
    assert gf.sum() == pytest.approx(cf, rel=1e-9, abs=1e-12)
    if N <= 1200:                                                # independent scipy reference, every u
        np.testing.assert_allclose(gf, _g_ref(n, N, k, N - n), rtol=0, atol=1e-12)
    assert np.all(np.diff(gf) <= 0) and gf[0] <= 1.0
    # the reported number is the exact closed form for one look
    r = risk_profile(n, N, k, 0.05, DELTA)
    assert r["basis"] == "exact" and r["expected_eps_bound"] == pytest.approx(cf / N, rel=1e-9)
    assert r["expected_eps_bound"] >= cf / N


def test_edge_cases_and_cummax_safeguard():
    g, rest = _tail_bound(10, 10, 3)
    assert len(g) == 0 and rest == 0.0
    r = risk_profile(10, 10, 3, 0.05, DELTA)                     # everything labelled: no output error possible
    assert r["expected_eps_bound"] == 0 and all(v == 0 for v in r["tail"].values())
    r = risk_profile(100, 400, -1, 0.05, DELTA)                  # k = -1: nothing bottled
    assert r["expected_eps_bound"] == 0 and r["delta_spent"] == 0 and all(v == 0 for v in r["eps_at_delta"].values())
    # k > n is clipped to n (accept-all rule: every unlabelled bad item is bottled)
    assert expected_eps_closed_form(5, 12, 9) == expected_eps_closed_form(5, 12, 5) == pytest.approx(7 / 12)
    # the j-max: g(u) >= v_j(u) for every j <= k (not only j = k), on a grid incl. impossible sizes (u+1+j > N)
    for (n, N) in [(5, 12), (8, 16), (40, 50), (117, 400)]:
        for k in (0, 1, 3, n):
            g = _g_full(n, N, k)
            for j in range(min(k, n) + 1):
                v = np.array([hypergeom.cdf(j, N, u + 1 + j, n) if u + 1 + j <= N else 0.0 for u in range(N - n)])
                assert np.all(g[:N - n] >= v - 1e-12)


def test_reported_fields_bench_sizes_and_delta_spent_is_v_kc():
    for N, n_exp, k_exp, e_exp in [(400, 117, 4, 0.0300), (1200, 199, 7, 0.0334)]:
        (n,) = tuned_looks([single_look_design(N, 0.05)[0][0]], [1.0], N, 0.05, DELTA, "rectified")
        k = k_critical(n, N, 0.05, DELTA, "rectified")
        assert (n, k) == (n_exp, k_exp)
        r = risk_profile(n, N, k, 0.05, DELTA)
        assert round(r["expected_eps_bound"], 4) == e_exp
        v = _rectified_levels(n, N, 0.05)[k]
        assert r["delta_spent"] == pytest.approx(v, abs=1e-12) and r["delta_spent"] <= DELTA
        assert r["tail"]["0.05"] == r["delta_spent"]
        assert r["eps_at_delta"]["0.1"] <= 0.05                     # P[U/N > eps] <= delta
        for d, e in r["eps_at_delta"].items():                      # P[U/N > eps_at_delta[d]] <= d, smallest such
            u = round(e * N)
            assert _g_full(n, N, k)[u] <= float(d) + 1e-15
            if u > 0:
                assert _g_full(n, N, k)[u - 1] > float(d)
        tails = [r["tail"][f"{x:g}"] for x in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1)]
        assert all(a >= b for a, b in zip(tails, tails[1:]))
    r4, r12 = risk_profile(117, 400, 4, 0.05, DELTA), risk_profile(199, 1200, 7, 0.05, DELTA)
    assert round(r4["tail"]["0.06"], 3) == 0.040 and round(r4["tail"]["0.07"], 3) == 0.015
    assert round(r12["tail"]["0.03"], 2) == 0.55 and round(r12["tail"]["0.07"], 4) == 0.0078


# ----------------------------------------------------------------------------------------------- (b) exhaustive
def test_vectorised_rule_matches_fixed_sequence():
    """Ties the enumeration below to the real code: fixed_sequence(method='rectified', optional cap) bottles exactly
    the vectorised rule's U on random subsets, tie patterns and bad sets."""
    rng = np.random.default_rng(0)
    N = 12
    for name, score in _patterns(N, rng).items():
        for n in (3, 6, 9):
            cal = _subsets(N, n)
            for M in (0, 3, 6, 12):
                bad = np.zeros(N, bool)
                bad[rng.choice(N, M, replace=False)] = True
                for eps, alpha, cap in ((0.1, 0.3, None), (0.2, 0.2, None), (0.3, 0.1, 0.15), (0.2, 0.3, 0.1)):
                    k = k_used(n, N, eps, alpha, "rectified", max_expected_eps=cap)
                    vec = _errors_vec(cal, bad, score, k)
                    for r in rng.choice(len(cal), min(40, len(cal)), replace=False):
                        idx = np.nonzero(cal[r])[0]
                        tau, _ = fixed_sequence(score[idx], bad[idx], N, eps, alpha, "rectified", max_expected_eps=cap)
                        acc = score >= tau
                        acc[idx] = False
                        assert int((acc & bad).sum()) == int(vec[r]), (name, n, M, eps, alpha, cap)


def test_exhaustive_every_bad_set_n10():
    """N = 10: EVERY bad set (2^10), every calibration subset, several n, k and tie patterns: the whole tail and the
    mean are bounded; with all items bad and distinct scores the bound is attained exactly (non-vacuous)."""
    N = 10
    rng = np.random.default_rng(1)
    bad_sets = np.array(list(itertools.product([False, True], repeat=N)), bool)
    for n in (2, 4, 6, 8):
        cal = _subsets(N, n)
        for k in sorted({0, 1, 2, min(3, n), n}):
            g = _g_full(n, N, k)
            for name, score in _patterns(N, rng).items():
                if name == "all_tied" and k > 1:
                    continue
                for bad in bad_sets:
                    _check_law(_errors_vec(cal, bad, score, k), g, (n, k, name, bad.sum()))
            U = _errors_vec(cal, np.ones(N, bool), np.linspace(1, 0, N), k)     # all bad, distinct: attains g
            np.testing.assert_allclose([np.mean(U > u) for u in range(N + 1)], g, atol=1e-12)
            assert U.mean() == pytest.approx(g.sum(), abs=1e-12)


@pytest.mark.parametrize("N", [13, 16])
def test_exhaustive_bad_prefix_with_ties(N):
    """N = 13, 16: every calibration subset, every |B| = M (bad items at the top, the worst placement; ties shrink
    the accepted set), k from the certificate at several (eps, alpha) and k = 0, so |B| < m+1+k is covered."""
    rng = np.random.default_rng(N)
    pats = _patterns(N, rng)
    if N == 16:
        pats = {k: pats[k] for k in ("distinct", "pairs", "grid")}
    ns = range(1, N) if N == 13 else (2, 4, 6, 8, 10, 12, 14)
    worst_gap = 1.0
    for n in ns:
        cal = _subsets(N, n)
        ks = {0} | {k_critical(n, N, e, a, "rectified") for e in (0.1, 0.2, 0.3) for a in (0.05, 0.2)}
        for k in sorted(x for x in ks if x >= 0):
            g = _g_full(n, N, k)
            for name, score in pats.items():
                order = np.argsort(-score, kind="stable")
                for M in range(N + 1):
                    bad = np.zeros(N, bool)
                    bad[order[:M]] = True
                    U = _errors_vec(cal, bad, score, k)
                    _check_law(U, g, (N, n, k, name, M))
                    if name == "distinct" and M == N:
                        worst_gap = min(worst_gap, abs(U.mean() - g.sum()))
    assert worst_gap < 1e-9, "the all-bad workload should attain the mean bound exactly"


def test_exhaustive_random_bad_placement_with_ties():
    rng = np.random.default_rng(11)
    N = 13
    for _ in range(80):
        score = rng.integers(0, 5, N) / 4.0
        bad = rng.random(N) < rng.uniform(0.1, 0.9)
        n = int(rng.integers(1, N))
        k = int(rng.integers(0, n + 1))
        _check_law(_errors_vec(_subsets(N, n), bad, score, k), _g_full(n, N, k), (n, k))


# ----------------------------------------------------------------------------------------------- (c) Monte Carlo
def _mc_U(N, n, k, M, R, seed):
    """U for R uniform calibration draws, M bad items at the top (distinct scores): unlabelled bad items before the
    (k+1)-th labelled bad item (all unlabelled bad items if fewer than k+1 are labelled)."""
    rng = np.random.default_rng(seed)
    lab = np.argsort(rng.random((R, N)), axis=1)[:, :M] < n        # labelled flags of the M bad items (rank < n)
    c = np.cumsum(lab, axis=1)
    has = c[:, -1] >= k + 1
    pos = np.argmax(c >= k + 1, axis=1)                           # index of the (k+1)-th labelled bad item
    return np.where(has, pos - k, M - c[:, -1])


@pytest.mark.parametrize("N,n,k,e_ref", [(400, 117, 4, 0.02998), (1200, 199, 7, 0.03337)])
def test_monte_carlo_bench_sizes_attain_the_profile(N, n, k, e_ref):
    R = 2000
    m = int(math.floor(0.05 * N))
    M = 3 * (m + 1 + k)
    U = _mc_U(N, n, k, M, R, seed=N + k)
    g = _g_full(n, N, k)
    for e in (0.03, 0.05, 0.07):
        u = int(math.floor(e * N))
        p = float(np.mean(U > u))
        se = math.sqrt(g[u] * (1 - g[u]) / R)
        assert abs(p - g[u]) <= 3 * se, (e, p, g[u], se)
    se = U.std(ddof=1) / math.sqrt(R) / N
    assert abs(U.mean() / N - e_ref) <= 3 * se + 5e-6, (U.mean() / N, e_ref, se)
    assert risk_profile(n, N, k)["expected_eps_bound"] == pytest.approx(e_ref, abs=5e-6)
    # the vectorised simulation is the real rule: 300 draws through fixed_sequence on a workload with good items too
    rng = np.random.default_rng(7)
    score = rng.permutation(N).astype(float)
    bad = np.zeros(N, bool)
    bad[np.argsort(-score)[:M]] = True                              # same law: bad items occupy the top M scores
    Ureal = []
    for r in range(300):
        idx = rng.permutation(N)[:n]
        tau, _ = fixed_sequence(score[idx], bad[idx], N, 0.05, DELTA, "rectified")
        acc = score >= tau
        acc[idx] = False
        Ureal.append(int((acc & bad).sum()))
    assert k_critical(n, N, 0.05, DELTA, "rectified") == k
    assert abs(np.mean(Ureal) / N - e_ref) <= 3 * np.std(Ureal, ddof=1) / math.sqrt(300) / N


def test_monte_carlo_realistic_workload_is_close_to_the_bound():
    """Bad items spread over the score range (about 20% bad, more at low scores): the law depends only on the
    ordering of bad items, so the bound is (nearly) attained, never exceeded."""
    N, n, k, R = 400, 117, 4, 2000
    rng = np.random.default_rng(3)
    score = rng.random(N)
    bad = rng.random(N) < 0.02 + 0.5 * (1 - score) ** 3
    cal_rng = np.random.default_rng(4)
    U = []
    for r in range(R):
        idx = cal_rng.permutation(N)[:n]
        tau, _ = fixed_sequence(score[idx], bad[idx], N, 0.05, DELTA, "rectified")
        acc = score >= tau
        acc[idx] = False
        U.append(int((acc & bad).sum()))
    U = np.asarray(U)
    bound = risk_profile(n, N, k, 0.05, DELTA)
    se = U.std(ddof=1) / math.sqrt(R) / N
    assert U.mean() / N <= bound["expected_eps_bound"] + 3 * se
    assert U.mean() / N >= bound["expected_eps_bound"] - 3 * se      # attained (|B| >> m+1+k)
    assert np.mean(U > 20) <= bound["delta_spent"] + 3 * math.sqrt(DELTA * 0.9 / R)


# ----------------------------------------------------------------------------------------------- (d) harness
def _gold(wl):
    return {t[:400]: g for t, g in zip(wl.texts, wl.gold)}


def _run(wl, client=None, acc=None, **kw):
    kw = {"mode": "fused", "k": 20, "eps": 0.05, "seed": 3, **kw}
    cfg = BottleConfig(cheap="cheap", expert="expert", **kw)
    client = client or MockClient(_gold(wl), acc=acc or {"cheap": 0.8, "expert": 0.95})
    return asyncio.run(Bottler(client, SPEC, cfg).run(wl.texts))


@pytest.mark.parametrize("N,n,k,e", [(400, 117, 4, 0.0300), (1200, 199, 7, 0.0334)])
def test_default_certificate_reports_the_exact_profile(N, n, k, e):
    res = _run(toy_workload(N, seed=5))
    c = res.certificate
    r = c["risk"]
    assert c["n_cal"] == n and r["basis"] == "exact"
    assert r["looks"] == [{"n_cal": n, "alpha": DELTA, "k": k}]
    assert round(r["expected_eps_bound"], 4) == e
    assert r["delta_spent"] == pytest.approx(_rectified_levels(n, N, 0.05)[k], abs=1e-12)
    assert r["n_expert_unusable"] == 0 and r["unusable_share"] == 0.0
    assert f"<= {e * 100:.2f}%" in c["guarantee"] and "exact" in c["guarantee"]
    assert c["guarantee"].startswith("P[share of final answers != expert > 0.05] <= 0.1")
    json.dumps(c)                                                     # plain JSON types only
    assert _run(toy_workload(N, seed=5)).certificate == c             # deterministic, comparable with ==


def test_k_in_the_profile_is_the_k_actually_applied():
    """Single look: rebuilding the closed-form threshold from the calibration labels (spied) with the profile's k and
    n gives exactly the certificate's tau."""
    import bottlekit.harness as H
    from bottlekit.certify import disagreement
    rec = {}
    orig = H.certify_sequential

    async def spy(score, final, expert_fn, N, eps, delta, looks, *a, **kw):
        cert, labels = await orig(score, final, expert_fn, N, eps, delta, looks, *a, **kw)
        rec.update(score=np.asarray(score, float), final=list(final), labels=dict(labels), seed=kw["seed"])
        return cert, labels

    H.certify_sequential = spy
    try:
        res = _run(toy_workload(400, seed=7))
    finally:
        H.certify_sequential = orig
    c = res.certificate
    (look,) = c["risk"]["looks"]
    assert look["n_cal"] == c["n_cal"] == len(rec["labels"])
    idx = np.random.default_rng(rec["seed"]).permutation(400)[:look["n_cal"]]
    bad = np.array([disagreement(rec["final"][i], rec["labels"][int(i)]) for i in idx])
    bs = np.sort(rec["score"][idx][bad])[::-1]
    k = look["k"]
    tau = -np.inf if len(bs) <= k else float(np.nextafter(bs[k], np.inf))
    assert np.isfinite(tau) and c["tau"] == tau
    assert k == k_critical(look["n_cal"], 400, 0.05, DELTA, "rectified")


def test_staged_and_custom_looks_are_union_bounds_over_every_prespecified_look():
    wl = toy_workload(400, seed=5)
    for kw in ({"design": "staged"}, {"looks": (60, 120, 200), "spending": (1, 2, 3)},
               {"looks": (60, 120, 200, 300), "target_coverage": 0.05}):     # last one stops after an early look
        c = _run(wl, **kw).certificate
        r = c["risk"]
        assert r["basis"] == "union bound" and "union bound" in c["guarantee"]
        assert len(r["looks"]) >= len(c["looks"]) and len(r["looks"]) >= 2
        assert sum(p["alpha"] for p in r["looks"]) == pytest.approx(DELTA)
        for p in r["looks"]:
            assert p["k"] == k_critical(p["n_cal"], 400, 0.05, p["alpha"], "rectified")
            single = risk_profile(p["n_cal"], 400, p["k"], 0.05, DELTA)
            assert r["expected_eps_bound"] >= single["expected_eps_bound"] - 1e-12
            assert all(r["tail"][e] >= single["tail"][e] - 1e-12 for e in single["tail"])
        assert r["delta_spent"] <= DELTA + 1e-12
        if "target_coverage" in kw:
            assert len(c["looks"]) < len(r["looks"])                 # stopped early: still all looks in the bound
    # the reported union bound itself is the sum of the looks' tails (capped at 1) and matches the plan
    r = _run(wl, design="staged").certificate["risk"]
    ns, ks = [p["n_cal"] for p in r["looks"]], [p["k"] for p in r["looks"]]
    parts = [_g_full(a, 400, b) for a, b in zip(ns, ks)]
    tot = np.minimum(1.0, sum(parts))
    assert r["tail"]["0.05"] == pytest.approx(tot[20], abs=1e-12)
    assert r["expected_eps_bound"] == pytest.approx(tot.sum() / 400, rel=1e-9)


def test_fixed_hypergeometric_and_not_certified_paths():
    wl = toy_workload(400, seed=5)
    c = _run(wl, calibration="fixed", n_cal=150).certificate
    n = tune_size(150, 400, 0.05, DELTA, "rectified")
    assert c["risk"]["basis"] == "exact" and c["risk"]["looks"] == [
        {"n_cal": n, "alpha": DELTA, "k": k_critical(n, 400, 0.05, DELTA, "rectified")}]
    h = _run(wl, certificate="hypergeometric").certificate
    hk = h["risk"]["looks"][0]
    assert hk["k"] == k_critical(hk["n_cal"], 400, 0.05, DELTA, "hypergeometric")
    assert h["risk"]["expected_eps_bound"] <= risk_profile(hk["n_cal"], 400, k_critical(hk["n_cal"], 400, 0.05, DELTA,
                                                                                         "rectified"))[
        "expected_eps_bound"]
    tiny = _run(toy_workload(20, seed=2))
    t = tiny.certificate
    assert not t["certified"] and tiny.coverage() == 0.0
    assert t["risk"]["basis"] == "not certified" and t["risk"]["expected_eps_bound"] == 0.0
    assert all(v == 0.0 for v in t["risk"]["tail"].values()) and t["risk"]["delta_spent"] == 0.0
    assert "nothing bottled" in t["guarantee"]


@pytest.mark.parametrize("calibration", ["sequential", "fixed"])
def test_budget_path_reports_only_the_inversion(calibration):
    wl = toy_workload(600, seed=1)
    res = _run(wl, acc={"cheap": 0.7, "expert": 0.95}, eps=0.01, seed=2, expert_budget=40, calibration=calibration,
               n_cal=150)
    c = res.certificate
    assert c.get("budget_capped")
    r = c["risk"]
    # v0.9.1: every budget batch reports the Bonferroni-split profile (budget inversion at d/2 when binding)
    assert r["basis"] == "budget split" and r["budget_binding"] and r["part"] == "budget inversion"
    assert r["expected_eps_bound"] is None and r["tail"] is None and r["delta_spent"] is None and r["looks"] is None
    assert r["delta_split"] == {"k_rule": 0.05, "budget_inversion": 0.05}
    assert r["eps_at_delta"]["0.1"] == c["eps_achieved"]              # the inversion at delta/2 = 0.05
    vals = [r["eps_at_delta"][d] for d in ("0.5", "0.1", "0.05", "0.01")]
    assert all(a <= b for a, b in zip(vals, vals[1:]))               # smaller d -> larger bound
    assert "no expected-error bound under an expert budget" in c["guarantee"]


def test_unusable_expert_answers_are_reported_separately():
    wl = toy_workload(400, seed=4)
    client = NoneExpertClient(_gold(wl), wl.texts[::25], acc={"cheap": 0.75, "expert": 0.95})
    c = _run(wl, client=client, seed=1).certificate
    F = c["n_expert_unusable"]
    r = c["risk"]
    assert F > 0 and r["n_expert_unusable"] == F and r["unusable_share"] == F / 400
    assert r["expected_eps_bound"] == pytest.approx(expected_eps_closed_form(117, 400, 4))   # F not folded in
    assert f"plus {F}/400" in c["guarantee"] and "not part of the expectation" in c["guarantee"]
    assert "expert-routed" in c["guarantee"]


# ----------------------------------------------------------------------------------------------- (e) speed
def test_risk_profile_is_fast_at_large_N():
    """N = 100,000: n = round(1.25 sqrt(N/.05)) = 1768 and k ~ 77 (k_c there; the cost does not depend on the exact
    value). Best of 5 <= 50 ms; truncation at g < 1e-12 keeps the closed form to 1e-9."""
    n, N, k = 1768, 100_000, 77
    best = 1.0
    for _ in range(5):
        t0 = time.perf_counter()
        r = risk_profile(n, N, k, 0.05, DELTA)
        best = min(best, time.perf_counter() - t0)
    assert best <= 0.05, best
    g, rest = _tail_bound(n, N, k)
    assert (g.sum() + rest) / N == pytest.approx(expected_eps_closed_form(n, N, k), rel=1e-9)
    assert g[-1] < 1e-12 and len(g) < N - n
    assert r["delta_spent"] == pytest.approx(hypergeom.cdf(k, N, 5001 + k, n), rel=1e-9)


# ----------------------------------------------------------------------------------------------- max_expected_eps
def test_k_mean_cap_is_the_largest_k_under_the_cap():
    for N in (20, 100, 400, 1200, 10_000):
        for n in sorted({1, 5, N // 10, N // 4, N // 2, N - 1}):
            for cap in (0.001, 0.01, 0.02, 0.03, 0.05, 0.2):
                k = k_mean_cap(n, N, cap)
                assert -1 <= k <= n
                if k >= 0:
                    assert expected_eps_closed_form(n, N, k) <= cap + 1e-15
                if k < n:
                    assert (k + 2) * (N - n) / ((n + 1) * N) > cap
    assert k_mean_cap(10, 10, 0.001) == 10                          # nothing unlabelled: no cap needed
    with pytest.raises(ValueError):
        k_mean_cap(5, 10, -0.1)


@pytest.mark.parametrize("N", [12, 16])
def test_mean_cap_enumeration_both_constraints_hold(N):
    """k_used = min(k_c, k_mean_cap(n, N, cap)) <= k_c, from (n, N, eps, alpha, cap) only. Over every calibration
    subset and every |B| (bad prefix, tie patterns): P[U > floor(eps N)] <= alpha AND E[U]/N <= cap."""
    rng = np.random.default_rng(N)
    pats = _patterns(N, rng)
    pats = {k: pats[k] for k in ("distinct", "pairs", "grid")}
    hits = 0
    for n in ((2, 4, 6, 8, 10) if N == 12 else (4, 8, 12)):
        cal = _subsets(N, n)
        for eps, alpha in ((0.1, 0.3), (0.2, 0.2), (0.3, 0.1)):
            m = int(math.floor(eps * N))
            kc = k_critical(n, N, eps, alpha, "rectified")
            for cap in (0.05, 0.1, 0.2, 0.4):
                k = k_used(n, N, eps, alpha, "rectified", max_expected_eps=cap)
                if kc >= 0:
                    assert k <= kc and k == min(kc, k_mean_cap(n, N, cap))
                else:
                    assert k == -1
                hits += k < kc
                for name, score in pats.items():
                    order = np.argsort(-score, kind="stable")
                    for M in range(N + 1):
                        bad = np.zeros(N, bool)
                        bad[order[:M]] = True
                        U = _errors_vec(cal, bad, score, k)
                        assert np.mean(U > m) <= alpha + 1e-12, (n, eps, alpha, cap, name, M)
                        assert U.mean() / N <= cap + 1e-12, (n, eps, alpha, cap, name, M, U.mean() / N)
    assert hits > 0, "the cap never bound: the check would be vacuous"


def test_mean_cap_in_the_harness():
    wl = toy_workload(400, seed=5)
    base = _run(wl)
    loose = _run(wl, max_expected_eps=1.0)                           # non-binding cap: identical answers
    assert loose.answers == base.answers and loose.provenance == base.provenance
    assert loose.certificate["tau"] == base.certificate["tau"]
    assert loose.certificate["risk"]["mean_cap_met"] is True and "max_expected_eps" not in base.certificate["risk"]
    tight = _run(wl, max_expected_eps=0.02)
    r = tight.certificate["risk"]
    assert r["looks"][0]["k"] <= base.certificate["risk"]["looks"][0]["k"]
    assert r["looks"][0]["k"] == min(k_critical(117, 400, 0.05, DELTA, "rectified"), k_mean_cap(117, 400, 0.02))
    assert r["expected_eps_bound"] <= 0.02 and r["mean_cap_met"] is True
    assert r["delta_spent"] <= DELTA
    assert tight.coverage() <= base.coverage()
    st = _run(wl, design="staged", max_expected_eps=0.02).certificate["risk"]   # per look cap * share: sum <= cap
    assert st["basis"] == "union bound" and st["expected_eps_bound"] <= 0.02 + 1e-12 and st["mean_cap_met"]
    fx = _run(wl, calibration="fixed", n_cal=150, max_expected_eps=0.02).certificate["risk"]
    assert fx["expected_eps_bound"] <= 0.02 + 1e-12
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", max_expected_eps=0.0)
    from bottlekit.__main__ import _cfg, _common
    ap = argparse.ArgumentParser()
    _common(ap)
    assert _cfg(ap.parse_args(["--expert", "e"])).max_expected_eps is None
    assert _cfg(ap.parse_args(["--expert", "e", "--max-expected-eps", "0.03"])).max_expected_eps == 0.03


# ----------------------------------------------------------------------------------------------- edge cases
def test_tail_table_cap_raises_and_harness_falls_back(monkeypatch):
    """A huge (u, j) table raises RiskTooLarge; the harness reports the exact closed form for a single
    look instead of failing or allocating it."""
    cmod = sys.modules["bottlekit.certify"]      # the package attribute `certify` is the function, not the module
    from bottlekit.certify import RiskTooLarge
    with pytest.raises(RiskTooLarge):
        _tail_bound(5000, 2_000_000, 400)
    monkeypatch.setattr(cmod, "RISK_MAX_CELLS", 10)             # force the fallback on a small toy run
    wl = toy_workload(600)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    res = asyncio.run(Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=2)).run(wl.texts))
    risk = res.certificate["risk"]
    if res.certificate.get("certified"):
        assert risk["basis"] == "closed form" and risk["tail"] is None
        assert risk["expected_eps_bound"] == pytest.approx(risk["expected_eps_closed_form"])
    assert len(res.answers) == 600


def test_risk_failure_never_breaks_a_run(monkeypatch):
    """Any exception inside the report-only risk profile is caught; the run completes unchanged."""
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    wl = toy_workload(500)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    base = asyncio.run(Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=5)).run(wl.texts))
    monkeypatch.setattr(Bottler, "_risk", boom)
    client2 = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    res = asyncio.run(Bottler(client2, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=5)).run(wl.texts))
    assert res.certificate["risk"]["basis"] == "unavailable" and "synthetic failure" in res.certificate["risk"]["error"]
    assert res.answers == base.answers and res.provenance == base.provenance


def test_budget_set_but_not_binding_is_qualified():
    """With an expert budget that does not bind, the batch makes the
    k-rule's claim at delta/2 inside the delta split, the profile reports only the split quantiles (no expectation or
    fixed-eps' tail: showing the k-rule's law only when the budget does not bind is a label-dependent selection), and
    its design (looks) is the k-rule's at delta/2."""
    from bottlekit.certify import risk_profile
    wl = toy_workload(600)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    res = asyncio.run(Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=2,
                                                         expert_budget=10_000)).run(wl.texts))
    c, r = res.certificate, res.certificate["risk"]
    assert c["budget_capped"] is False and c["delta"] == 0.1
    assert "the budget did not bind, so this claim comes from the k-rule" in c["guarantee"]
    assert "k-rule at 0.05 + budget inversion at 0.05" in c["guarantee"]
    assert r["basis"] == "budget split" and not r["budget_binding"] and r["part"] == "k-rule"
    assert r["expected_eps_bound"] is None and r["tail"] is None and r["delta_spent"] is None
    (look,) = r["looks"]
    assert look["alpha"] == 0.05 and look["k"] == k_used(look["n_cal"], 600, 0.05, 0.05, "rectified")
    ref = risk_profile(look["n_cal"], 600, look["k"], 0.05, 0.05, deltas=(0.25, 0.05, 0.025, 0.005))
    assert r["eps_at_delta"] == {d: ref["eps_at_delta"][h] for d, h in
                                 (("0.5", "0.25"), ("0.1", "0.05"), ("0.05", "0.025"), ("0.01", "0.005"))}
    assert r["eps_at_delta"]["0.1"] <= 0.05                         # the k-rule at delta/2 certified eps at delta/2
    assert "no expected-error bound under an expert budget" in c["guarantee"]
    assert "budget fallback is not covered" not in c["guarantee"]


def test_new_certificate_arguments_are_keyword_only():
    """Guard: max_expected_eps and m must never bind positionally (a positional m once landed in the
    max_expected_eps slot silently)."""
    s, b = np.array([0.9, 0.5, 0.2]), np.array([False, True, False])
    with pytest.raises(TypeError):
        fixed_sequence(s, b, 100, 0.05, 0.1, "rectified", 0.04)
    with pytest.raises(TypeError):
        k_used(50, 100, 0.05, 0.1, "rectified", 0.04)
    assert fixed_sequence(s, b, 100, 0.05, 0.1, "rectified", m=5) == fixed_sequence(s, b, 100, 0.05, 0.1, "rectified", m=5)
