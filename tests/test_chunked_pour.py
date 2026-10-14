"""Right-sized pours: a large pour split into label-free geometric chunks certified in sequence, with
the whole-batch guarantee P[U > floor(eps*N)] <= delta_batch from the exact convolution of the per-chunk worst-case
laws (certify.batch_tail). No servers needed.

Covers: batch_tail numerics (pmf complete and non-negative, truncation, critic values, 10k-draw Monte Carlo), an
EXHAUSTIVE tiny-N check with an ADAPTIVE adversary (chunk 2's bad set chosen from chunk 1's calibration sample and
labels, over every calibration subset), a >= 300-draw rule-level adaptive simulation at bench sizes, a 300-draw
harness simulation where chunk-1 expert labels make the chunk-2 bottle worse, the label-free plan (deterministic, no
split at bench sizes => byte-identical), stitching/ledger/streams, and every documented fallback.
"""

import asyncio
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
from bottlekit import BottleConfig, Bottler  # noqa: E402
from bottlekit.certify import (_tail_bound, batch_tail, chunk_law, fixed_sequence, k_used,  # noqa: E402
                               min_batch_for_tax, single_look_size)
from bottlekit.labeling import sub_seed  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

EPS, DELTA = 0.05, 0.1


def _client(wl, acc=None, cls=MockClient):
    return cls({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc=acc or {"cheap": 0.85, "expert": 0.95})


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=7, chunking="auto")   # opt-in
    base.update(kw)
    return BottleConfig(**base)


def _out(res):
    return json.dumps({"answers": res.answers, "provenance": res.provenance, "certificate": res.certificate,
                       "ledger": res.ledger}, sort_keys=True, default=str)


def _des(N, eps=EPS, delta=DELTA):
    n = single_look_size(N, eps, delta)
    return (n, N, k_used(n, N, eps, delta, "rectified"))


# ----------------------------------------------------------------------------------------------- batch_tail numerics
@pytest.mark.parametrize("n,N,k", [(3, 8, 1), (117, 400, 4), (169, 800, 6), (40, 60, 3), (5, 6, 5), (1, 2, 0)])
def test_chunk_law_is_a_complete_nonnegative_pmf_matching_the_tail(n, N, k):
    p, over = chunk_law(n, N, k)
    assert over == 0.0 and (p >= 0).all() and abs(p.sum() - 1) < 1e-12 and len(p) == N - n + 1
    g, _ = _tail_bound(n, N, k, u_stop=N - n - 1)
    surv = 1 - np.cumsum(p)                               # P[V > u]
    assert np.allclose(surv[:-1], g, atol=1e-12) and abs(surv[-1]) < 1e-12


def test_chunk_law_trivial_cases():
    for n, N, k in ((5, 5, 2), (3, 10, -1), (10, 8, 0)):
        p, over = chunk_law(n, N, k)
        assert p.tolist() == [1.0] and over == 0.0


def test_truncated_law_matches_full_law():
    full = batch_tail([(117, 400, 4), (169, 800, 6), (243, 1600, 9)], eps=EPS, survival=True)
    p_full, _ = chunk_law(243, 1600, 9, full=True)
    p_cut, over = chunk_law(243, 1600, 9, full=False)
    assert 0 < over < 1e-11 and np.allclose(p_full[:len(p_cut)], p_cut, atol=1e-15)
    assert abs(p_cut.sum() + over - 1) < 1e-12
    assert full["overflow"] == 0.0


@pytest.mark.parametrize("plan,expected", [((400, 400), 0.0441), ((400, 800), 0.0487), ((300, 600), 0.0509),
                                           ((200,) * 4, 0.0119), ((400,) * 3, 0.0209),
                                           ((800, 1600, 3200), 0.0268)])
def test_batch_tail_reproduces_the_critic_values(plan, expected):
    r = batch_tail([_des(c) for c in plan], eps=EPS)
    assert abs(r["delta_batch"] - expected) < 5e-4      # 3 decimals
    assert r["delta_batch"] <= DELTA


def test_single_chunk_batch_tail_is_the_single_look_profile():
    from bottlekit.certify import risk_profile
    for N in (400, 1200):
        n, _, k = _des(N)
        bt = batch_tail([(n, N, k)], eps=EPS)
        rp = risk_profile(n, N, k, EPS, DELTA)
        assert abs(bt["delta_batch"] - rp["delta_spent"]) < 1e-12
        assert abs(bt["expected_eps_bound"] - rp["expected_eps_bound"]) < 1e-9
        for e in rp["tail"]:
            assert abs(bt["tail"][e] - rp["tail"][e]) < 1e-12


def _mc_allbad(designs, D, rng):
    """All-bad workload (attains every g_j(u) at once): U_j = #unlabelled before the (k_j+1)-th labelled item."""
    tot = np.zeros(D, np.int64)
    for n, N, k in designs:
        pos = np.sort(np.argsort(rng.random((D, N)), axis=1)[:, :n], axis=1)
        tot += pos[:, k] - k
    return tot


@pytest.mark.parametrize("designs", [[(117, 400, 4), (117, 400, 4)], [(117, 400, 4), (169, 800, 6)],
                                     [(60, 200, 2), (60, 200, 2), (60, 200, 2)]])
def test_batch_tail_agrees_with_10k_draw_monte_carlo(designs):
    rng = np.random.default_rng(123)
    D = 10_000
    U = _mc_allbad(designs, D, rng)
    bt = batch_tail(designs, eps=EPS, survival=True)
    s = np.asarray(bt["survival"])
    Ntot = sum(N for _, N, _ in designs)
    for u in range(0, int(0.08 * Ntot)):
        emp, ex = float((U > u).mean()), float(s[u]) if u < len(s) else 0.0
        assert abs(emp - ex) <= 4.5 * math.sqrt(max(ex * (1 - ex), 1e-4) / D) + 2e-3, (u, emp, ex)
    mean_ex = bt["expected_eps_bound"] * Ntot
    assert abs(U.mean() - mean_ex) <= 4.5 * U.std() / math.sqrt(D)


# ----------------------------------------------------------------------------------------------- exhaustive, adaptive
def _rule_U(B, S, k):
    """Closed-form k-rule, items 0..N-1 in DESCENDING score order: bottles the unlabelled items strictly above the
    (k+1)-th highest labelled bad item (everything if <= k labelled bad). Returns #unlabelled bad items bottled."""
    lab_bad = sorted(i for i in S if i in B)
    if len(lab_bad) <= k:
        return sum(1 for b in B if b not in S)
    cut = lab_bad[k]
    return sum(1 for b in B if b not in S and b < cut)


@pytest.mark.parametrize("n,eps,delta", [(3, 0.25, 0.5), (4, 0.25, 0.4), (5, 0.25, 0.5)])
def test_rule_helper_matches_fixed_sequence(n, eps, delta):
    N = 8
    k = k_used(n, N, eps, delta, "rectified")
    assert k >= 1
    scores = -np.arange(N, dtype=float)
    for r in range(N + 1):
        for B in itertools.combinations(range(N), r):
            Bs = set(B)
            bad = np.array([i in Bs for i in range(N)])
            for S in itertools.combinations(range(N), n):
                S = list(S)
                tau, _ = fixed_sequence(scores[S], bad[S], N, eps, delta, "rectified")
                inS = np.zeros(N, bool)
                inS[S] = True
                assert int(((scores >= tau) & bad & ~inS).sum()) == _rule_U(Bs, S, k)


def _subsets(N):
    return [frozenset(c) for r in range(N + 1) for c in itertools.combinations(range(N), r)]


@pytest.mark.parametrize("d1,d2", [((3, 8, 1), (3, 8, 1)), ((3, 8, 0), (4, 8, 2)), ((2, 8, 1), (3, 8, 2)),
                                   ((4, 8, 1), (2, 8, 0)), ((2, 6, 1), (3, 10, 1))])
def test_exhaustive_adaptive_two_chunks(d1, d2):
    """Every bad set B1 of chunk 1; chunk 2's bad set chosen ADAPTIVELY from chunk 1's calibration sample S1 and its
    labels (the adversary also knows B1, hence U1): the best response maximises P[U2 > t - U1] for each S1. Over all
    calibration subsets S1, S2: max over B1 of P[U1 + U2 > t] <= convolved tail for EVERY t, with equality (the
    all-bad workload attains it)."""
    (n1, N1, k1), (n2, N2, k2) = d1, d2
    S1s = [list(c) for c in itertools.combinations(range(N1), n1)]
    S2s = [list(c) for c in itertools.combinations(range(N2), n2)]
    T = N1 + N2
    # chunk 2: for every bad set, the exact law of U2 over the calibration draw -> best adaptive response per need
    best2 = np.zeros(T + 2)                                  # best2[v] = max_B2 P[U2 > v] (v = t - U1)
    for B2 in _subsets(N2):
        cnt = np.bincount([_rule_U(B2, S, k2) for S in S2s], minlength=N2 + 1) / len(S2s)
        surv = 1 - np.cumsum(cnt)
        best2[:N2 + 1] = np.maximum(best2[:N2 + 1], surv)
    bt = batch_tail([d1, d2], survival=True)
    s = np.asarray(bt["survival"] + [0.0] * (T + 2))
    worst = np.zeros(T + 1)
    for B1 in _subsets(N1):
        U1 = np.array([_rule_U(B1, S, k1) for S in S1s])
        for t in range(T + 1):
            v = t - U1                                       # need U2 > v
            p = np.where(v < 0, 1.0, best2[np.clip(v, 0, T + 1)])
            worst[t] = max(worst[t], float(p.mean()))
    assert (worst <= s[:T + 1] + 1e-12).all(), (worst, s[:T + 1])
    assert np.allclose(worst, s[:T + 1], atol=1e-12)        # tight: the bound is attained


def test_rule_level_adaptive_simulation_at_bench_sizes():
    """>= 300 draws (here 20,000) at 400 + 400: chunk 1 = worst-case workload (all bad); chunk 2's bad set is chosen
    after seeing chunk 1's outcome, as the best response for the remaining allowance (the u+1+k top items bad, which
    attains g_2(u)). The joint violation rate must be <= delta_batch (+ MC error) and is ~equal to it."""
    rng = np.random.default_rng(5)
    d1, d2 = _des(400), _des(400)
    D, t = 20_000, math.floor(EPS * 800)
    U1 = _mc_allbad([d1], D, rng)
    n2, N2, k2 = d2
    viol = 0
    for u1 in U1:
        need = t - int(u1)                                   # chunk 2 must bottle > need bad items
        if need < 0:
            viol += 1
            continue
        nb = min(N2, need + 1 + k2)                          # best response: the top need+1+k2 items bad
        S = rng.choice(N2, n2, replace=False)
        viol += _rule_U(set(range(nb)), S.tolist(), k2) > need
    rate = viol / D
    db = batch_tail([d1, d2], eps=EPS)["delta_batch"]
    se = math.sqrt(db * (1 - db) / D)
    assert rate <= db + 4 * se and abs(rate - db) <= 4 * se + 2e-3, (rate, db)
    assert db < DELTA


# ----------------------------------------------------------------------------------------------- the plan
def test_min_batch_for_tax_is_label_free_and_meets_the_tax():
    n_min = min_batch_for_tax(EPS, DELTA)
    assert n_min == min_batch_for_tax(EPS, DELTA)
    assert single_look_size(n_min, EPS, DELTA) / n_min <= 0.30
    assert single_look_size(n_min - 1, EPS, DELTA) / (n_min - 1) > 0.30
    assert 300 <= n_min <= 450
    assert 150 <= min_batch_for_tax(0.1, DELTA) <= 250
    assert min_batch_for_tax(EPS, DELTA, tax=0.2) > n_min


def test_unified_min_size_helper_is_the_exact_smallest_n():
    """ONE exact helper. min_batch_for_tax is a thin wrapper over certify.min_certified_size (moved from
    session.py; the session name is the same function): 355 at eps .05 (a bisection can return 400, a later
    crossing of the sawtooth), 190 at .1, and every N in [scan start, N_min) has share > tax."""
    import importlib
    C = importlib.import_module("bottlekit.certify")      # (the package attribute 'certify' is the function)
    S = importlib.import_module("bottlekit.session")
    assert S.min_certified_size is C.min_certified_size
    assert min_batch_for_tax(EPS, DELTA) == C.min_certified_size(EPS, DELTA) == 355
    assert min_batch_for_tax(0.1, DELTA) == C.min_certified_size(0.1, DELTA) == 190
    for N in range(int(4 * 1.25 ** 2 / EPS), 355):           # exact: no smaller N in the unclipped regime
        assert single_look_size(N, EPS, DELTA) / N > 0.30
    with pytest.raises(ValueError):                          # one tax domain (0, 0.45] for both helpers
        min_batch_for_tax(EPS, DELTA, tax=0.5)
    with pytest.raises(ValueError):
        C.min_certified_size(EPS, DELTA, tax=0.6)
    assert _bottler().chunk_min_size() == 355


def _bottler(**kw):
    wl = toy_workload(50, seed=1)
    return Bottler(_client(wl), SPEC, _cfg(**kw))


def test_plan_sizes_rule():
    b = _bottler()
    n_min = b.chunk_min_size()
    # N_min comes from the exact helper, 355 at eps .05 (not 400, a bisection's
    # later sawtooth crossing). Plans whose first chunk is max(N_min, M0) with M0 >= 400 are unchanged (800 into 400,
    # 4,800 into 400); plans into small bottles shrink (1,000 into 200: 400 + 600 -> 355 + 645), and the smallest
    # 2-chunk pour into a 400-item bottle is 755.
    assert n_min == 355
    assert b.chunk_plan(800, 400)["sizes"] == [400, 400]
    assert b.chunk_plan(1000, 200)["sizes"] == [355, 645]
    p = b.chunk_plan(4800, 400)
    assert p["sizes"] == [400, 800, 1600, 2000] and p["chunks"] == 4 and p["delta_batch"] <= DELTA
    assert sum(p["sizes"]) == 4800 and min(p["sizes"]) >= n_min
    assert _bottler(max_chunks=2).chunk_plan(4800, 400)["sizes"] == [400, 4400]
    assert _bottler(chunk_growth=0.5).chunk_plan(2000, 800)["sizes"] == [400, 600, 1000]
    assert b.chunk_plan(754, 400)["chunks"] == 1          # 400 + 354: the rest would be below N_min
    assert b.chunk_plan(755, 400)["sizes"] == [400, 355]  # (was 799 -> one chunk at N_min = 400)
    assert b.chunk_plan(5000, 0)["sizes"][0] == n_min     # empty bottle: first chunk N_min
    for N, M0 in ((800, 400), (4800, 400), (1000, 200)):  # deterministic and label-free
        assert b.chunk_plan(N, M0) == _bottler(seed=99).chunk_plan(N, M0)


@pytest.mark.parametrize("M0", [300, 350, 380, 400, 420, 600, 800, 1000, 1200])
def test_no_split_at_bench_sizes(M0):
    """bench pour1 (N=400, M0~400) and pour2 (N=400, M0~800) never split => byte-identical by construction."""
    p = _bottler().chunk_plan(400, M0)
    assert p["chunks"] == 1


@pytest.mark.parametrize("kw,reason", [({"chunking": "off"}, "off"), ({"memo": True}, "memo"),
                                       ({"expert_budget": 50}, "expert_budget"), ({"design": "staged"}, "design"),
                                       ({"looks": (100, 200)}, "looks"), ({"calibration": "fixed"}, "calibration"),
                                       ({"export": False}, "export"), ({"certificate": "hypergeometric"}, "certificate")])
def test_ineligible_configurations_never_split(kw, reason):
    p = _bottler(**kw).chunk_plan(4800, 400)
    assert p["chunks"] == 1 and reason in p["reason"]


def test_plan_falls_back_when_delta_batch_exceeds_delta(monkeypatch):
    real = H.batch_tail

    def fake(*a, **k):
        r = real(*a, **k)
        r["delta_batch"] = 0.2
        return r
    monkeypatch.setattr(H, "batch_tail", fake)
    p = _bottler().chunk_plan(800, 400)
    assert p["chunks"] == 1 and "delta_batch" in p["reason"]


def test_planned_chunk_design_equals_the_certificate_design():
    """_chunk_design(N) is exactly the (n, k) the ordinary pour applies (label-free), incl. a mean-risk cap."""
    for kw in ({}, {"max_expected_eps": 0.03}, {"tune": False}, {"eps": 0.1}):
        b = _bottler(**kw)
        for N in (37, 120, 400, 401, 650, 800, 1600, 2000, 3333):
            n, _, k = b._chunk_design(N)
            assert n == b._planned_n(N)
            cfg = b.cfg
            assert k == k_used(n, N, cfg.eps, cfg.delta, "rectified", max_expected_eps=cfg.max_expected_eps)


def test_partition_stream_is_distinct_from_every_other_stream():
    for seed in (0, 7, 1000):
        for v, M in ((0, 400), (3, 1200)):
            p = sub_seed(seed, H.PARTITION_STREAM_KEY, v + 1, M)
            others = {sub_seed(seed, 1), sub_seed(seed, 1, v + 1, M), sub_seed(seed, 2), sub_seed(seed, 7),
                      *[sub_seed(seed, 11, c) for c in range(5)], *[sub_seed(seed, 1, v + j, M + 400 * j) for j in range(1, 6)]}
            assert p not in others


# ----------------------------------------------------------------------------------------------- end to end (mock)
def _setup(n_run=400, n_tot=1200, seed=21, cls=MockClient, acc=None, **kw):
    wl = toy_workload(n_tot, seed=seed)
    client = _client(wl, acc=acc or {"cheap": 0.8, "expert": 0.95}, cls=cls)
    b = Bottler(client, SPEC, _cfg(**kw))
    r0 = asyncio.run(b.run(wl.texts[:n_run]))
    return wl, client, b, r0


def test_chunked_pour_equals_the_manual_chain_and_stitches_in_input_order():
    wl, client, b, r0 = _setup()
    T = wl.texts[400:1200]
    snap = client.ledger.snapshot()
    res = asyncio.run(b.pour(r0.artifact, T, votes=False))
    led = client.ledger.since(snap)
    c = res.certificate
    assert c["chunked"] and c["plan"] == [400, 400] and res.diagnostics["chunked"]
    assert res.diagnostics["mode"] == "pour" and c["batch_claim"] and c["plan_consistent"]
    assert c["delta_batch"] <= DELTA and abs(c["delta_batch"] - 0.0441) < 5e-4
    assert res.artifact.version == r0.artifact.version + 2
    assert res.ledger == led
    # manual replication of the label-free partition and the chain of ordinary pours
    M0, v0 = len(r0.artifact.texts), r0.artifact.version
    perm = np.random.default_rng(sub_seed(b.cfg.seed, 13, v0 + 1, M0)).permutation(len(T))
    b2 = Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.95}), SPEC, _cfg())
    art, off, ans, prov, n_cal = r0.artifact, 0, [None] * len(T), [None] * len(T), 0
    for csz in c["plan"]:
        idx = np.sort(perm[off:off + csz])
        off += csz
        r = asyncio.run(b2.pour(art, [T[i] for i in idx], votes=False, _chunk=True))
        for jj, i in enumerate(idx):
            ans[i], prov[i] = r.answers[jj], r.provenance[jj]
        n_cal += r.certificate["n_cal"]
        art = r.artifact
    assert ans == res.answers and prov == res.provenance and n_cal == c["n_cal"]
    assert art.texts == res.artifact.texts and art.labels == res.artifact.labels
    # streams: every chunk's calibration key is fresh (distinct from each other and from run's (seed, 1))
    keys = [tuple(k) for k in c["cal_stream"]]
    assert len(set(keys)) == len(keys) and tuple(r0.certificate["cal_stream"]) not in keys
    assert c["partition_stream"] == [b.cfg.seed, 13, v0 + 1, M0]
    # per-chunk certificates are ordinary single-look certificates at (eps, delta)
    for cd, csz in zip(c["chunks"], c["plan"]):
        assert cd["N"] == csz and cd["delta"] == DELTA and cd["design"] == "single"
    assert c["eps_certified"] == EPS + c["n_expert_unusable"] / len(T)
    assert "whole batch" in c["guarantee"] and "each chunk" in c["guarantee"]
    assert res.coverage() == pytest.approx(sum(p == "bottle" for p in prov) / len(T))


def test_chunking_off_equals_v06_single_pour_and_small_pours_are_unchanged():
    wl, client, b, r0 = _setup()
    off = Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.95}), SPEC, _cfg(chunking="off"))
    on = Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.95}), SPEC, _cfg())
    for T, votes in ((wl.texts[400:800], False), (wl.texts[400:800], True), (wl.texts[400:1200], False)):
        a = asyncio.run(off.pour(r0.artifact, T, votes=votes))
        assert "chunked" not in a.certificate
        if len(T) == 400:                                   # one-chunk plan: the identical single pour
            c = asyncio.run(on.pour(r0.artifact, T, votes=votes))
            assert _out(a) == _out(c)
            assert c.diagnostics["chunking"]["chunks"] == 1
            assert a.artifact.texts == c.artifact.texts and a.artifact.version == c.artifact.version


def test_votes_auto_and_warm_chunks_and_mixed_mode():
    wl, client, b, r0 = _setup()
    T = wl.texts[400:1200]
    w = asyncio.run(b.pour(r0.artifact, T, votes=True))
    assert w.certificate["chunked"] and w.diagnostics["mode"] == "pour_warm" and "cheap_vote" in w.ledger
    a = asyncio.run(b.pour(r0.artifact, T, votes="auto"))
    assert a.certificate["chunked"]
    for ch in a.diagnostics["chunks"]:
        assert ch["votes_decision"] is not None and ch["mode"] in ("pour", "pour_warm")
    modes = {ch["mode"] for ch in a.diagnostics["chunks"]}
    assert a.diagnostics["mode"] == (modes.pop() if len(modes) == 1 else "pour_mixed")


def _emb(texts, d=8):
    out = []
    for t in texts:
        r = np.random.default_rng(abs(hash(t)) % (2 ** 32))
        v = r.normal(size=d)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, np.float32)


def test_dense_chunked_pour_and_emb_validation():
    wl = toy_workload(1200, seed=4)
    E = _emb(wl.texts)
    b = Bottler(_client(wl), SPEC, _cfg())
    r0 = asyncio.run(b.run(wl.texts[:400], emb=E[:400]))
    res = asyncio.run(b.pour(r0.artifact, wl.texts[400:], votes=False, emb=E[400:]))
    assert res.certificate["chunked"] and res.artifact.dense and res.diagnostics["dense"] == "dense"
    assert all(ch["dense"] == "dense" for ch in res.diagnostics["chunks"])
    with pytest.raises(ValueError):
        asyncio.run(b.pour(r0.artifact, wl.texts[400:], votes=False, emb=E[400:1100]))


def test_unusable_expert_answers_are_added_over_the_whole_batch():
    class Flaky(MockClient):
        def _answer(self, alias, text, salt, pos=None, k=None):
            if alias == "expert" and hash(text) % 15 == 0:
                return "I am not sure."
            return super()._answer(alias, text, salt, pos, k)
    wl, client, b, r0 = _setup(cls=Flaky)
    res = asyncio.run(b.pour(r0.artifact, wl.texts[400:1200], votes=False))
    c = res.certificate
    F = sum(cd["n_expert_unusable"] for cd in c["chunks"])
    assert c["n_expert_unusable"] == F > 0
    assert c["eps_certified"] == pytest.approx(EPS + F / 800)
    assert c["risk"]["unusable_share"] == pytest.approx(F / 800)
    assert f"{c['eps_certified']:.4f}" in c["guarantee"]


def test_memo_budget_and_staged_pours_are_the_single_pour():
    wl, client, b, r0 = _setup()
    for kw in ({"memo": True}, {"expert_budget": 100}, {"design": "staged"}):
        x = asyncio.run(Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.95}), SPEC, _cfg(**kw)).pour(
            r0.artifact, wl.texts[400:1200], votes=False))
        y = asyncio.run(Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.95}), SPEC, _cfg(chunking="off", **kw)).pour(
            r0.artifact, wl.texts[400:1200], votes=False))
        assert "chunked" not in x.certificate
        assert x.answers == y.answers and x.provenance == y.provenance
        assert json.dumps(x.certificate, sort_keys=True, default=str) == json.dumps(y.certificate, sort_keys=True,
                                                                                     default=str)


def test_chunking_default():
    """The default is the 'cold' rule. 'off' keeps the single pour and 'auto' the per-chunk planner."""
    cfg = BottleConfig(cheap="c", expert="e")
    assert cfg.chunking == "cold" and cfg.chunk_cold_ratio == 4.0 == H.CHUNK_COLD_RATIO
    b = Bottler(_client(toy_workload(50, seed=1)), SPEC, BottleConfig(cheap="cheap", expert="expert"))
    assert b.chunk_plan(4800, 400)["sizes"] == [400, 800, 1600, 2000]     # a cold pour of 12x memory is planned
    assert b.chunk_plan(1599, 400)["chunks"] == 1 and "'cold' rule" in b.chunk_plan(1599, 400)["reason"]
    off = Bottler(_client(toy_workload(50, seed=1)), SPEC, BottleConfig(cheap="cheap", expert="expert", chunking="off"))
    assert off.chunk_plan(4800, 400) == {"chunks": 1, "reason": "chunking='off'"}


def test_cli_chunking_flag():
    import argparse
    from bottlekit.__main__ import _cfg as cli_cfg
    ns = argparse.Namespace(cheap="c", expert="e", mode="auto", k=20, contexts=3, warmup=0, eps=0.05, delta=0.1,
                            calibration="sequential", n_cal=300, seed=0, no_balance=False, no_weights=False, fusion=None,
                            n_build=None, patience=None, target_coverage=None, looks=None, budget=None, spending=None,
                            no_tune=False)
    default = BottleConfig(cheap="c", expert="e").chunking
    assert cli_cfg(ns).chunking == default               # run subcommand has no flag: the config default
    ns.chunking = None                                   # pour without --chunking: the config default
    assert cli_cfg(ns).chunking == default
    for v in ("auto", "off", "cold"):
        ns.chunking = v
        assert cli_cfg(ns).chunking == v


def test_config_validation():
    for kw in ({"chunking": "on"}, {"chunk_tax": 0.0}, {"chunk_tax": 1.0}, {"chunk_tax": 0.5}, {"chunk_growth": 0},
               {"max_chunks": 0}, {"chunk_cold_ratio": 0}, {"chunk_cold_ratio": -1.0}):
        with pytest.raises(ValueError):
            _cfg(**kw)
    assert _cfg(chunk_tax=0.45).chunk_tax == 0.45


# ----------------------------------------------------------------------------------------------- adversarial harness
class Trap(MockClient):
    """Expert that makes the flywheel ANTI-informative: on items containing 'trap', it answers gold shifted by the
    item's chunk index (0 = the run batch, 1 = chunk 1, 2 = chunk 2). The chunk assignment comes from the label-free
    partition, fixed before any label of the batch, so every item's expert answer is fixed in advance; what the bottle
    learns from chunk 1's expert answers (shift 1) is wrong on chunk 2 (shift 2)."""
    chunk_of: dict = {}

    def _answer(self, alias, text, salt, pos=None, k=None):
        if alias == "expert" and salt == "single" and " trap" in text:
            g = self.gold[text[:400]]
            i = SPEC.names.index(g)
            return SPEC.names[(i + self.chunk_of.get(text, 0)) % len(SPEC.names)]
        return super()._answer(alias, text, salt, pos, k)


@pytest.mark.parametrize("chunking", ["auto", "cold"])
def test_adversarial_flywheel_harness_simulation_300_draws(chunking):
    """chunking='cold': the same simulation through the pour-level cold decision (forced-cold chunks); its
    ratio is lowered to 2.0 here only so that this 400-item pour into a 200-item bottle splits (the ratio decides
    WHETHER to split, never the per-chunk law)."""
    eps = 0.1                                               # N_min = 190: plan 200 + 200 keeps the sim fast
    ckw = {"chunking": chunking, **({"chunk_cold_ratio": 2.0} if chunking == "cold" else {})}
    wl = toy_workload(600, seed=77)
    texts = [t + (" trap" if i % 3 == 0 else "") for i, t in enumerate(wl.texts)]
    gold = {t[:400]: g for t, g in zip(texts, wl.gold)}
    client = Trap(gold, acc={"cheap": 0.85, "expert": 0.97})
    client.chunk_of = {}
    b0 = Bottler(client, SPEC, _cfg(eps=eps, seed=0))
    r0 = asyncio.run(b0.run(texts[:200]))
    T = texts[200:600]
    M0, v0 = len(r0.artifact.texts), r0.artifact.version
    plan = Bottler(client, SPEC, _cfg(eps=eps, seed=1, **ckw)).chunk_plan(len(T), M0)
    assert plan["chunks"] == 2
    db = plan["delta_batch"]
    viol, viol2, worse, D = 0, 0, 0, 300
    for d in range(D):
        cfg = _cfg(eps=eps, seed=1000 + d, **ckw)
        b = Bottler(client, SPEC, cfg)
        sizes = b.chunk_plan(len(T), M0)["sizes"]
        perm = np.random.default_rng(sub_seed(cfg.seed, 13, v0 + 1, M0)).permutation(len(T))
        client.chunk_of = {}
        off = 0
        for j, c in enumerate(sizes):
            for i in perm[off:off + c]:
                client.chunk_of[T[int(i)]] = j + 1
            off += c
        res = asyncio.run(b.pour(r0.artifact, T, votes=False))
        assert res.certificate.get("chunked") and res.diagnostics["mode"] == "pour"
        ex = [client._answer("expert", t, "single") for t in T]
        bad = [a != e for a, e in zip(res.answers, ex)]
        viol += float(np.mean(bad)) > eps
        c2 = set(perm[sizes[0]:].tolist())
        bad2 = [bad[i] for i in c2]
        viol2 += float(np.mean(bad2)) > eps
        # the adversary bites: the bottle after chunk 1 agrees less with chunk 2's expert than the initial bottle
        if d < 20:
            idx2 = sorted(c2)
            p0, _ = r0.artifact.predict([T[i] for i in idx2])
            mid = asyncio.run(Bottler(client, SPEC, cfg).pour(r0.artifact, [T[int(i)] for i in np.sort(perm[:sizes[0]])],
                                                             votes=False, _chunk=True)).artifact
            p1, _ = mid.predict([T[i] for i in idx2])
            worse += np.mean([a == ex[i] for a, i in zip(p1, idx2)]) <= np.mean([a == ex[i] for a, i in zip(p0, idx2)])
    rate, rate2 = viol / D, viol2 / D
    assert worse >= 15                                      # chunk-1 labels made the chunk-2 bottle worse
    assert rate <= db + 3.5 * math.sqrt(db * (1 - db) / D) + 1e-9, (rate, db)
    assert rate2 <= 0.1 + 3.5 * math.sqrt(0.09 / D), rate2


# ----------------------------------------------------------------------------------------------- 'cold' rule
def _art(a):
    return json.dumps({"v": a.version, "texts": a.texts, "labels": a.labels, "src": list(a.src), "fp": a.fingerprint,
                       "forecast": a.forecast}, sort_keys=True, default=str)


def _cold_setup(n_run=200, n_tot=1400, seed=31, **kw):
    """A run() bottle of n_run items; mk(**cfg) gives a Bottler with a FRESH mock client (ledgers comparable)."""
    wl = toy_workload(n_tot, seed=seed)

    def mk(**k):
        return Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.95}), SPEC, _cfg(**{**kw, **k}))
    r0 = asyncio.run(mk(chunking="off").run(wl.texts[:n_run]))
    return wl, mk, r0


@pytest.mark.parametrize("votes,kw", [(True, {}), ("auto", {"gain_prior": 10.0})])
def test_cold_rule_never_splits_warm_pours(votes, kw):
    """votes=True and votes='auto' deciding warm (gain_prior 10: the prior rule picks votes): a 6x-memory pour that
    v0.7's 'auto' would split runs as the single pour, byte-identical to chunking='off' (decision included)."""
    wl, mk, r0 = _cold_setup()
    T = wl.texts[200:1400]
    M0 = len(r0.artifact.texts)
    assert len(T) >= 4 * M0 and mk(chunking="auto").chunk_plan(len(T), M0)["chunks"] >= 2
    c = asyncio.run(mk(chunking="cold", **kw).pour(r0.artifact, T, votes=votes))
    o = asyncio.run(mk(chunking="off", **kw).pour(r0.artifact, T, votes=votes))
    assert "chunked" not in c.certificate and c.diagnostics["mode"] == "pour_warm" and "cheap_vote" in c.ledger
    ch = c.diagnostics["chunking"]
    assert ch["chunks"] == 1 and ch["decision"] == "warm" and "never split" in ch["reason"]
    assert _out(c) == _out(o) and _art(c.artifact) == _art(o.artifact)
    assert json.dumps(c.diagnostics["votes_decision"], sort_keys=True, default=str) == \
        json.dumps(o.diagnostics["votes_decision"], sort_keys=True, default=str)


def test_cold_rule_auto_cold_split_equals_the_forced_cold_chain(monkeypatch):
    """votes='auto' deciding cold (gain_prior 0): the decision is made ONCE before planning, every chunk is a forced
    cold pour, and the result equals (i) the forced-cold pour, (ii) v0.7's planner with votes=False and (iii) the
    manual chain of ordinary forced-cold chunk pours over the label-free partition."""
    wl, mk, r0 = _cold_setup()
    T = wl.texts[200:1400]
    calls = []
    real = H.Bottler._votes_pay

    def spy(self, *a, **k):
        out = real(self, *a, **k)
        calls.append(bool(out[0]))
        return out
    monkeypatch.setattr(H.Bottler, "_votes_pay", spy)
    b = mk(chunking="cold", gain_prior=0.0)
    a = asyncio.run(b.pour(r0.artifact, T, votes="auto"))
    assert calls == [False]                                 # one pour-level decision, never re-decided per chunk
    c = a.certificate
    assert c["chunked"] and c["plan"] == [355, 845] and c["batch_claim"] and c["delta_batch"] <= DELTA
    assert a.diagnostics["mode"] == "pour" and a.diagnostics["votes_decision"]["basis"] == "prior"
    assert all(ch["mode"] == "pour" and ch["votes_decision"] == {"reason": "fixed by caller"}
               for ch in a.diagnostics["chunks"])
    assert "cheap_vote" not in a.ledger
    info = a.diagnostics["chunking"]
    assert info["rule"] == "cold" and info["decision"] == "cold" and info["ratio"] == 4.0 and info["sizes"] == c["plan"]
    f = asyncio.run(mk(chunking="cold", gain_prior=0.0).pour(r0.artifact, T, votes=False))
    assert _out(f) == _out(a) and _art(f.artifact) == _art(a.artifact)
    v7 = asyncio.run(mk(chunking="auto", gain_prior=0.0).pour(r0.artifact, T, votes=False))
    assert _out(v7) == _out(a)
    M0, v0 = len(r0.artifact.texts), r0.artifact.version
    perm = np.random.default_rng(sub_seed(b.cfg.seed, 13, v0 + 1, M0)).permutation(len(T))
    bm = mk(chunking="cold", gain_prior=0.0)
    art, off, ans, prov = r0.artifact, 0, [None] * len(T), [None] * len(T)
    for csz in c["plan"]:
        idx = np.sort(perm[off:off + csz])
        off += csz
        r = asyncio.run(bm.pour(art, [T[i] for i in idx], votes=False, _chunk=True))
        for jj, i in enumerate(idx):
            ans[i], prov[i] = r.answers[jj], r.provenance[jj]
        art = r.artifact
    assert ans == a.answers and prov == a.provenance and _art(art) == _art(a.artifact)
    # the stored forecast is the last chunk's, recorded for the returned bottle's own version (v0.6 rule)
    assert a.artifact.version == v0 + len(c["plan"]) == a.artifact.forecast["artifact_version"]


@pytest.mark.parametrize("votes", [False, True, "auto"])
def test_cold_rule_below_ratio_is_byte_identical_to_off(votes):
    """N < 4 x memory (2x and 2.5x: v0.7's 'auto' splits both), and a >= 4x pour whose plan has one chunk: answers,
    provenance, certificate, ledger, bottle (forecast included) and every diagnostic but 'chunking' equal 'off'."""
    wl, mk, r0 = _cold_setup(n_run=400)
    M0 = len(r0.artifact.texts)
    cases = [(r0, wl.texts[400:1200]), (r0, wl.texts[400:1400])]
    small = asyncio.run(mk(chunking="off").run(wl.texts[:100]))      # 4x memory, but 400 < 355 + 355: one chunk
    cases.append((small, wl.texts[1000:1400]))
    for base, T in cases:
        m0 = len(base.artifact.texts)
        if base is r0:
            assert len(T) < 4 * M0 and mk(chunking="auto").chunk_plan(len(T), M0)["chunks"] >= 2
        else:
            assert len(T) >= 4 * m0 and mk(chunking="cold").chunk_plan(len(T), m0)["chunks"] == 1
        c = asyncio.run(mk(chunking="cold").pour(base.artifact, T, votes=votes))
        o = asyncio.run(mk(chunking="off").pour(base.artifact, T, votes=votes))
        assert "chunked" not in c.certificate
        assert _out(c) == _out(o) and _art(c.artifact) == _art(o.artifact)
        dc, do = dict(c.diagnostics), dict(o.diagnostics)
        assert dc.pop("chunking")["chunks"] == 1 and do.pop("chunking")["reason"] == "chunking='off'"
        dc.pop("wall_time_s", None), do.pop("wall_time_s", None)   # wall time is never byte-identical
        assert json.dumps(dc, sort_keys=True, default=str) == json.dumps(do, sort_keys=True, default=str)


@pytest.mark.parametrize("kw,reason", [({"memo": True}, "memo"), ({"expert_budget": 50}, "expert_budget"),
                                       ({"design": "staged"}, "design"), ({"looks": (100, 200)}, "looks"),
                                       ({"calibration": "fixed"}, "calibration"), ({"export": False}, "export"),
                                       ({"certificate": "hypergeometric"}, "certificate")])
def test_cold_rule_ineligible_configurations_never_split(kw, reason):
    b = _bottler(chunking="cold", **kw)
    p = b.chunk_plan(4800, 400)
    assert p["chunks"] == 1 and reason in p["reason"] and b._cold_gate(4800, 400) is not None


class _CrashAt(MockClient):
    """Raises on the crash_at-th expert call (once)."""

    def __init__(self, *a, crash_at=None, **kw):
        super().__init__(*a, **kw)
        self.crash_at, self.n_exp = crash_at, 0

    async def chat(self, alias, messages, **kw):
        if alias == "expert":
            self.n_exp += 1
            if self.crash_at is not None and self.n_exp == self.crash_at:
                raise RuntimeError("injected crash")
        return await super().chat(alias, messages, **kw)


def test_session_flush_of_a_large_submission_is_chunked_with_lineage_and_resume(tmp_path):
    """BottleSession (min_batch=0, votes=False) over 200 -> 1,000 -> 200 items with the default 'cold' rule: the
    1,000-item flush into the 200-item bottle is chunked (355 + 645); the lineage guard records one key per chunk (all
    distinct over the stream; up to v0.7 session.cert_streams crashed on a chunked certificate); the batch equals the
    manual run -> pour chain; a crash inside chunk 2 followed by resume() + flush() equals the uninterrupted stream
    (same reserved key, same chunk keys, the interrupted chunks are not applied twice)."""
    from bottlekit import BottleSession
    from bottlekit.session import _norm, cert_streams
    wl = toy_workload(1400, seed=41)
    gold = {t[:400]: g for t, g in zip(wl.texts, wl.gold)}
    acc = {"cheap": 0.82, "expert": 0.95}
    cfg = _cfg(chunking="cold", seed=17)
    subs = [wl.texts[:200], wl.texts[200:1200], wl.texts[1200:1400]]

    def j(o):
        return json.dumps(_norm(o), sort_keys=True)

    def view(rec):
        return j({k: rec[k] for k in ("answers", "provenance", "certificate", "ledger", "kind", "N", "streams")})
    cu = MockClient(gold, acc=acc)
    u = BottleSession(cu, SPEC, cfg, state_path=tmp_path / "u.json", min_batch=0, votes=False)
    for t in subs:
        asyncio.run(u.submit(t))
    b1 = u.batches[1]
    c1 = b1["certificate"]
    M0 = u.batches[0]["memory_out"]
    assert c1["chunked"] and c1["plan"] == [355, 645] and b1["kind"] == "pour_cold"
    keys = cert_streams(c1)
    assert keys == [tuple(k) for k in b1["streams"]] and len(keys) == 2 and keys[0] == (17, 1, 1, M0)
    every = [k for rec in u.batches for k in cert_streams(rec["certificate"])]
    assert len(every) == len(set(every)) == 4 and sorted(every) == sorted(u.used_streams)
    assert u.batches[2]["streams"][0] == [17, 1, 1 + len(c1["plan"]), u.batches[1]["memory_out"]]
    assert u.report()["rows"][1]["delta"] == pytest.approx(c1["delta_batch"])
    # the manual chain
    bm = Bottler(MockClient(gold, acc=acc), SPEC, cfg)
    r0 = asyncio.run(bm.run(subs[0]))
    r1 = asyncio.run(bm.pour(r0.artifact, subs[1], votes=False))
    assert j(r1.certificate) == j(c1) and r1.answers == b1["answers"] and r1.provenance == b1["provenance"]
    # crash inside chunk 2 of batch 1 (after chunk 1's expert calls), then resume + flush
    prov0 = u.batches[0]["provenance"]
    idx1 = b1["diagnostics"]["chunk_index"][0]
    calls_b0 = sum(p != "bottle" for p in prov0)
    calls_ch1 = sum(b1["provenance"][i] != "bottle" for i in idx1)
    cc = _CrashAt(gold, acc=acc, crash_at=calls_b0 + calls_ch1 + 5)
    p = tmp_path / "c.json"
    s = BottleSession(cc, SPEC, cfg, state_path=p, min_batch=0, votes=False)
    asyncio.run(s.submit(subs[0]))
    with pytest.raises(RuntimeError, match="injected crash"):
        asyncio.run(s.submit(subs[1]))
    assert len(s.batches) == 1 and s._inflight is not None and cc.n_exp > calls_b0 + calls_ch1
    del s
    r = BottleSession.resume(p, MockClient(gold, acc=acc), SPEC, cfg)
    assert tuple(r._inflight["key"]) == keys[0] and not set(keys) & set(r.used_streams)
    rows = asyncio.run(r.flush())
    assert len(rows) == 1 and rows[0]["batch"] == 1
    asyncio.run(r.submit(subs[2]))
    assert len(r.batches) == 3
    for a_, b_ in zip(u.batches, r.batches):
        assert view(a_) == view(b_)
    assert _art(r.artifact) == _art(u.artifact)
