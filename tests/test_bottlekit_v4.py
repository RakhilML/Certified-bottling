"""Tests for the v0.4 default calibration design: ONE pre-specified look at the full delta of
n = round(design_c * sqrt(N / eps)) items (<= N/2, sawtooth-tuned), replacing the staged 1:4 default of v0.3.

  * single_look_design: shape, label-free monotonicity in N and eps, n <= N//2, small-N agreement with default_design
  * harness wiring: the default certificate has one look at share 1.0; design='staged' and explicit looks are still
    reachable and reproduce v0.3 certificates bit-for-bit (pinned values + reconstruction from the primitives)
  * the budget path gets the SAME tuned looks, spending and order seed as certify_sequential
  * validity: exact worst case (adversarial top workload: P[error] = v_{k_c} <= delta) on a grid of (N, eps), and a
    1000-draw Monte Carlo of the actual certify_sequential code with the harness-derived look
  * edge cases (N = 1, 2, tiny workloads where nothing can be certified), config validation and the CLI flag
"""

import argparse
import asyncio
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
from bottlekit import BottleConfig, Bottler, certified_eps  # noqa: E402
from bottlekit.certify import (certify_sequential, default_design, disagreement, k_critical,  # noqa: E402
                               single_look_design, spend, tuned_looks)
from bottlekit.labeling import sub_seed  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

DELTA = 0.1


def _harness_look(N, eps, delta=DELTA, c=1.25, method="rectified"):
    looks, w = single_look_design(N, eps, c)
    return tuned_looks(list(looks), spend(looks, w), N, eps, delta, method)


# ----------------------------------------------------------------------------------------------- the design
def test_single_look_design_shape_monotone_and_capped():
    Ns = list(range(1, 300)) + [400, 1000, 1200, 3000, 10_000, 30_000, 100_000]
    epss = (0.01, 0.02, 0.05, 0.1, 0.2)
    table = {}
    for eps in epss:
        for N in Ns:
            looks, w = single_look_design(N, eps)
            assert len(looks) == 1 and w == (1.0,)
            n = looks[0]
            assert 1 <= n and (n <= N // 2 or N < 2)
            table[eps, N] = n
    for eps in epss:                                                    # non-decreasing in N
        assert all(table[eps, a] <= table[eps, b] for a, b in zip(Ns, Ns[1:]))
    for N in Ns:                                                        # non-increasing in eps
        assert all(table[a, N] >= table[b, N] for a, b in zip(epss, epss[1:]))
    assert single_look_design(1200, 0.05) == ((194,), (1.0,))
    assert single_look_design(400, 0.05) == ((112,), (1.0,))
    assert single_look_design(400, 0.05, c=1.0) == ((89,), (1.0,))
    with pytest.raises(ValueError):
        single_look_design(400, 0.0)
    with pytest.raises(ValueError):
        single_look_design(400, 0.05, c=0)


def test_single_look_equals_small_n_default_and_edge_sizes():
    for N in range(1, 126):                                         # N <= 4 c^2 / eps = 125 at eps=.05, c=1.25
        assert single_look_design(N, 0.05)[0] == default_design(N)[0]
    assert single_look_design(1, 0.05) == ((1,), (1.0,))
    assert single_look_design(2, 0.05) == ((1,), (1.0,))
    # sizes are a function of (N, eps, delta, method, c) only: deterministic and identical for every workload
    assert _harness_look(400, 0.05) == _harness_look(400, 0.05) == [117]
    assert _harness_look(1200, 0.05) == [199]


# ----------------------------------------------------------------------------------------------- exact validity
@pytest.mark.parametrize("N", [20, 50, 100, 150, 400, 1200, 3000, 10_000])
@pytest.mark.parametrize("eps", [0.02, 0.05, 0.1])
def test_single_look_exact_worst_case_is_at_most_delta(N, eps):
    """With one look at level delta the certificate is the closed-form k-rule, whose worst case over every workload
    is v_{k_c} = HypCDF(k_c; N, m+1+k_c, n) (certify._rectified_levels). Check it with an INDEPENDENT computation for
    the tuned harness look: <= delta, and k_c+1 calibration errors would not be (the test is not over-conservative
    by a whole count)."""
    (n,) = _harness_look(N, eps)
    m = int(math.floor(eps * N))
    kc = k_critical(n, N, eps, DELTA, "rectified")
    v = lambda j: hypergeom.cdf(j, N, m + 1 + j, n) if m + 1 + j <= N else 0.0  # noqa: E731
    if kc >= 0:
        assert max(v(j) for j in range(kc + 1)) <= DELTA + 1e-12
    if kc + 1 <= n and m + 2 + kc <= N:
        assert v(kc + 1) > DELTA


def _run_seq(score, answer, N, eps, looks, spending, seed, method="rectified", delta=DELTA):
    async def expert(idx):
        return {i: "y" for i in idx}
    return asyncio.run(certify_sequential(score, answer, expert, N, eps, delta, looks, seed=seed, method=method,
                                          spending=spending))


def _closed_form_errors(score, bad, lab_idx, kc):
    """Unlabelled bad items bottled by the closed-form k-rule (independent re-implementation)."""
    N = len(score)
    lab = np.zeros(N, bool)
    lab[lab_idx] = True
    if kc < 0:
        return 0, np.inf
    bs = np.sort(score[lab & bad])[::-1]
    tau = -np.inf if len(bs) <= kc else float(np.nextafter(bs[kc], np.inf))
    return int(((score >= tau) & bad & ~lab).sum()), tau


def test_certify_sequential_with_one_look_is_the_closed_form_rule():
    """One look: certify_sequential == fixed_sequence on order[:n] at the full delta (no hidden second look, no
    Bonferroni split, the selection is trivial)."""
    rng = np.random.default_rng(3)
    for r in range(200):
        N = int(rng.choice([150, 400, 1200]))
        eps = float(rng.choice([0.02, 0.05, 0.1]))
        looks = _harness_look(N, eps)
        score = rng.random(N)
        bad = rng.random(N) < float(rng.uniform(0.02, 0.5)) * (1 - score) ** float(rng.uniform(0.5, 3))
        answer = np.where(bad, "x", "y").tolist()
        seed = int(rng.integers(1 << 30))
        cert, lab = _run_seq(score, answer, N, eps, looks, (1.0,), seed)
        order = np.random.default_rng(seed).permutation(N)
        assert sorted(lab) == sorted(order[:looks[0]].tolist())
        assert "L=1 looks" in cert.method and "[1.0]" in cert.method and len(cert.looks) == 1
        _, tau = _closed_form_errors(score, bad, order[:looks[0]], k_critical(looks[0], N, eps, DELTA, "rectified"))
        assert cert.tau == tau


def _mc_violation(N, eps, case, R, seed):
    looks = _harness_look(N, eps)
    n = looks[0]
    m = int(math.floor(eps * N))
    kc = k_critical(n, N, eps, DELTA, "rectified")
    rng = np.random.default_rng(seed)
    viol = 0
    for r in range(R):
        if case == "smooth":
            score = rng.random(N)
            bad = rng.random(N) < 0.02 + 0.5 * (1 - score) ** 3
        else:                                    # adversarial: |B| = m+1+k_c bad items at the very top
            score = np.linspace(1, 0, N)
            bad = np.zeros(N, bool)
            bad[: min(N, m + 1 + max(kc, 0))] = True
        order = rng.permutation(N)
        err, _ = _closed_form_errors(score, bad, order[:n], kc)
        viol += err > m
    return viol / R, kc


@pytest.mark.parametrize("N,eps", [(100, 0.05), (400, 0.02), (400, 0.05), (400, 0.1), (1200, 0.05), (3000, 0.02)])
@pytest.mark.parametrize("case", ["smooth", "top"])
def test_single_look_monte_carlo_validity(N, eps, case):
    """1000 draws per cell (the full grid N in {100, 400, 1200, 3000} x eps in {.02, .05, .1} is in
    sims/mc_validity.py): output violation rate <= delta + 3 sigma; the adversarial case really stresses the test."""
    R = 1000
    rate, kc = _mc_violation(N, eps, case, R, seed=N * 7 + int(eps * 1000) + len(case))
    assert rate <= DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / R), (rate, kc)
    if case == "top" and kc >= 0:
        assert rate > 0.04, "the adversarial case should stress the test (otherwise this check is vacuous)"


def test_monte_carlo_through_certify_sequential():
    """Same adversarial check, through the real certify_sequential code path (300 draws)."""
    N, eps = 400, 0.05
    looks = _harness_look(N, eps)
    m = int(math.floor(eps * N))
    kc = k_critical(looks[0], N, eps, DELTA, "rectified")
    score = np.linspace(1, 0, N)
    bad = np.zeros(N, bool)
    bad[: m + 1 + kc] = True
    answer = np.where(bad, "x", "y").tolist()
    R, viol = 300, 0
    for r in range(R):
        cert, lab = _run_seq(score, answer, N, eps, looks, (1.0,), 10_000 + r)
        acc = score >= cert.tau
        acc[list(lab)] = False
        viol += (acc & bad).sum() > m
    assert viol / R <= DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / R)


# ----------------------------------------------------------------------------------------------- harness wiring
def _gold(wl):
    return {t[:400]: g for t, g in zip(wl.texts, wl.gold)}


def _run(wl, **kw):
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, eps=0.05, seed=3, **kw)
    return asyncio.run(Bottler(MockClient(_gold(wl), acc={"cheap": 0.8, "expert": 0.95}), SPEC, cfg).run(wl.texts))


def test_harness_default_is_one_full_delta_look_and_staged_is_reachable():
    wl = toy_workload(400, seed=5)
    res = _run(wl)
    c = res.certificate
    assert BottleConfig(cheap="c", expert="e").design == "single"
    assert c["design"] == "single" and c["design_c"] == 1.25
    assert "L=1 looks" in c["method"] and "[1.0]" in c["method"]
    assert len(c["looks"]) == 1 and c["looks"][0]["n_cal"] == c["n_cal"] == 117 == _harness_look(400, 0.05)[0]
    assert sum(p.startswith("expert_calibration") for p in res.provenance) == 117
    st = _run(wl, design="staged")
    assert st.certificate["design"] == "staged" and "L=2 looks" in st.certificate["method"]
    cu = _run(wl, looks=(100, 200), spending=(1, 4))
    assert cu.certificate["design"] == "custom" and "L=2 looks" in cu.certificate["method"]
    fx = _run(wl, calibration="fixed", n_cal=150)
    assert fx.certificate["design"] == "fixed" and "design_c" not in fx.certificate


# v0.3 certificates on toy_workload(400, seed=5), MockClient(acc cheap .8 / expert .95), mode fused, eps .05, seed 3
# (captured with the unmodified v0.3 package). design='staged' and explicit looks must reproduce them exactly.
V03_PINNED = {
    "staged": {"tau": 1.0961203517443194, "n_cal": 217, "looks_n": [94, 217], "chosen_look": 1, "coverage": 0.3825},
    "custom": {"tau": 1.1052346994206281, "n_cal": 202, "looks_n": [94, 202], "chosen_look": 1, "coverage": 0.415},
}


@pytest.mark.parametrize("which", ["staged", "custom"])
def test_staged_and_custom_looks_reproduce_v03_bit_for_bit(which):
    wl = toy_workload(400, seed=5)
    res = _run(wl, design="staged") if which == "staged" else _run(wl, looks=(94, 202), spending=(1, 4), tune=False)
    c, pin = res.certificate, V03_PINNED[which]
    assert c["tau"] == pin["tau"] and c["n_cal"] == pin["n_cal"] and c["chosen_look"] == pin["chosen_look"]
    assert [h["n_cal"] for h in c["looks"]] == pin["looks_n"]
    assert res.coverage() == pin["coverage"]


def test_staged_harness_matches_direct_v03_reconstruction():
    """design='staged' == certify_sequential(tuned default_design(N), Bonferroni 1:4, seed sub_seed(seed, 1)) -- the
    v0.3 harness path rebuilt from the primitives."""
    wl = toy_workload(400, seed=7)
    rec, labels = {}, {}
    orig = H.certify_sequential

    async def spy(score, final, expert_fn, N, eps, delta, looks, *a, **kw):
        async def rec_expert(idx):
            out = await expert_fn(idx)
            labels.update(out)
            return out
        rec.update(score=np.array(score), final=list(final), looks=list(looks), kw=kw)
        return await orig(score, final, rec_expert, N, eps, delta, looks, *a, **kw)

    H.certify_sequential = spy
    try:
        res = _run(wl, design="staged")
    finally:
        H.certify_sequential = orig
    lk, w = default_design(400)
    assert rec["looks"] == tuned_looks(list(lk), spend(lk, w), 400, 0.05, 0.1, "rectified") == [94, 217]
    assert tuple(rec["kw"]["spending"]) == (1.0, 4.0) and rec["kw"]["seed"] == sub_seed(3, 1)

    async def replay(idx):
        return {i: labels[i] for i in idx}

    cert, _ = asyncio.run(orig(rec["score"], rec["final"], replay, 400, 0.05, 0.1, [94, 217], seed=sub_seed(3, 1),
                               method="rectified", spending=(1.0, 4.0)))
    assert cert.tau == (res.certificate["tau"] if res.certificate["tau"] is not None else np.inf)
    assert cert.n_cal == res.certificate["n_cal"]


def test_budget_path_gets_the_same_looks_spending_and_seed():
    """Budget-capped certificate: sequential_certified_eps must see the identical tuned single look, spending and
    order seed as certify_sequential, and its eps equals the direct one-look inversion. v0.9.1 (Bonferroni split under
    an expert budget): the look is tuned at delta/2 and the inversion runs at delta/2."""
    wl = toy_workload(600, seed=1)
    seen = {}
    o_seq, o_eps = H.certify_sequential, H.sequential_certified_eps

    async def spy_seq(score, final, expert_fn, N, eps, delta, looks, *a, **kw):
        seen["seq"] = (list(looks), kw.get("spending"), kw.get("seed"))
        return await o_seq(score, final, expert_fn, N, eps, delta, looks, *a, **kw)

    def spy_eps(score, final, labels, order_seed, looks, tau, N, delta, method="hypergeometric", spending=None):
        seen["eps"] = (list(looks), spending, order_seed)
        seen["args"] = (np.array(score), list(final), dict(labels), tau)
        return o_eps(score, final, labels, order_seed, looks, tau, N, delta, method=method, spending=spending)

    H.certify_sequential, H.sequential_certified_eps = spy_seq, spy_eps
    try:
        cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, eps=0.01, delta=0.1, seed=2,
                           expert_budget=40)
        res = asyncio.run(Bottler(MockClient(_gold(wl), acc={"cheap": 0.7, "expert": 0.95}), SPEC, cfg).run(wl.texts))
    finally:
        H.certify_sequential, H.sequential_certified_eps = o_seq, o_eps
    assert res.certificate.get("budget_capped") and "eps" in seen
    assert seen["seq"] == seen["eps"]
    looks, spending, seed = seen["eps"]
    assert looks == _harness_look(600, 0.01, delta=0.05) and tuple(spending) == (1.0,) and seed == sub_seed(2, 1)
    score, final, labels, tau = seen["args"]
    idx = np.random.default_rng(seed).permutation(600)[:looks[0]]
    k_end = sum(1 for i, lab in labels.items() if score[i] >= tau and disagreement(final[i], lab))
    direct = certified_eps(score[idx], [disagreement(final[i], labels[int(i)]) for i in idx], tau, 600, 0.05,
                           "rectified", k_end=k_end)
    assert res.certificate["eps_achieved"] == direct
    assert res.certificate["delta"] == 0.1 and res.certificate["delta_split"] == {"k_rule": 0.05,
                                                                                  "budget_inversion": 0.05}


def test_tiny_workloads_nothing_certifiable_and_n1_n2():
    wl = toy_workload(20, seed=2)
    res = _run(wl)
    assert not res.certificate["certified"] and res.coverage() == 0.0          # k_c = -1: nothing certified
    assert all(p.startswith("expert") for p in res.provenance)
    for N in (1, 2):
        (n,) = _harness_look(N, 0.05)
        assert 1 <= n <= N


def test_config_validation_and_cli_flag():
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", design="uniform")
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", design_c=0.0)
    from bottlekit.__main__ import _cfg, _common
    ap = argparse.ArgumentParser()
    _common(ap)
    assert _cfg(ap.parse_args(["--expert", "e"])).design == "single"
    assert _cfg(ap.parse_args(["--expert", "e", "--design", "staged"])).design == "staged"
    with pytest.raises(SystemExit):
        ap.parse_args(["--expert", "e", "--design", "bogus"])
