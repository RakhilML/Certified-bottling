"""Batch-spent chunk plans (cfg.chunk_spend='batch'): every chunk's k is raised to a common per-chunk
level alpha* >= delta, at the planned calibration sizes, chosen label-free by exact search so that the whole-batch
tail delta_batch (exact convolution of the per-chunk worst-case laws, certify.batch_tail) stays <= delta.

Covers: the feasibility numbers (4,800 into 400), exactness/maximality/monotonicity of alpha*, the
stored plan (delta_batch <= delta, k_j = k_used at alpha*, the mean-risk cap), 'chunk' bit-identical to the code without this feature
(loaded from $BOTTLEKIT_PRE_CHUNKSPEND_SNAPSHOT when set), the end-to-end pour (chunk certificates at alpha*, realized ==
planned, guarantee text, risk profile, forecast at the standard design), the private chunk-design kwarg, a plan
mismatch withdrawing the whole-batch claim, chunking='auto' designs independent of votes, the stream ledger, an
EXHAUSTIVE adaptive enumeration (2-3 chunks, N_j <= 10) at the inflated k, a rule-level Monte Carlo (all-bad and
adaptive best-response workloads) and >= 300-draw harness simulations with adaptive adversarial experts.
"""

import asyncio
import importlib.util
import itertools
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
from bottlekit import BottleConfig, Bottler  # noqa: E402
from bottlekit.certify import (_pvalues, batch_spent_alpha, batch_tail, k_mean_cap, k_used,  # noqa: E402
                               single_look_size)
from bottlekit.labeling import sub_seed  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

EPS, DELTA = 0.05, 0.1
PLAN_4800 = [400, 800, 1600, 2000]


def _client(wl, acc=None, cls=MockClient):
    return cls({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc=acc or {"cheap": 0.8, "expert": 0.95})


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=7, chunk_spend="batch")
    base.update(kw)
    return BottleConfig(**base)


def _bottler(**kw):
    return Bottler(_client(toy_workload(50, seed=1)), SPEC, _cfg(**kw))


def _out(res):
    return json.dumps({"answers": res.answers, "provenance": res.provenance, "certificate": res.certificate,
                       "ledger": res.ledger}, sort_keys=True, default=str)


def _art(a):
    return json.dumps({"v": a.version, "texts": a.texts, "labels": a.labels, "src": list(a.src), "fp": a.fingerprint,
                       "forecast": a.forecast}, sort_keys=True, default=str)


def _ns(sizes, eps=EPS, delta=DELTA):
    return [(single_look_size(c, eps, delta), c) for c in sizes]


# ----------------------------------------------------------------------------------------------- batch_spent_alpha
def test_feasibility_numbers_4800_into_400():
    """4,800 items into a 400-item bottle: plan [400, 800, 1600, 2000], (n, k) = (117,4) (169,6) (243,9) (269,10) at
    delta (delta_batch 0.0142) -> k 5, 7, 11, 12 with delta_batch 0.0943 <= 0.1. alpha* is reported as the SMALLEST
    common level giving that k-vector (0.2341); a first estimate, 0.258, is the top of the same k-vector's interval
    [0.2341, 0.2576): one more step (k_2 = 8) gives delta_batch 0.1127 > delta."""
    ds = _ns(PLAN_4800)
    assert [n for n, _ in ds] == [117, 169, 243, 269]
    alpha, designs, info = batch_spent_alpha(ds, 4800, EPS, DELTA)
    assert info["k_chunk_level"] == [4, 6, 9, 10] and abs(info["delta_batch_chunk_level"] - 0.0142) < 1e-4
    assert [d[2] for d in designs] == [5, 7, 11, 12] and [d[:2] for d in designs] == [tuple(x) for x in ds]
    assert 0.2340 < alpha < 0.2342 and alpha < 0.258
    assert abs(info["delta_batch"] - 0.0943) < 1e-4 and info["delta_batch"] <= DELTA
    assert batch_tail(designs, 4800, EPS)["delta_batch"] == pytest.approx(info["delta_batch"], abs=0)
    for (n, N, k) in designs:                              # every chunk's k is exactly the rule's k at alpha*
        assert k == k_used(n, N, EPS, alpha, "rectified")
    # the same plan through the harness
    p = _bottler().chunk_plan(4800, 400)
    assert p["sizes"] == PLAN_4800 and p["spend"] == "batch" and p["alpha_chunk"] == alpha
    assert [tuple(d) for d in p["designs"]] == designs and p["delta_batch"] == pytest.approx(info["delta_batch"])
    assert p["k_raise"] == [1, 1, 2, 2] and p["delta_batch_chunk_level"] == pytest.approx(0.0141760868766117)
    assert [list(d) for d in p["designs_chunk_level"]] == [[117, 400, 4], [169, 800, 6], [243, 1600, 9], [269, 2000, 10]]


@pytest.mark.parametrize("sizes,eps", [(PLAN_4800, EPS), ([355, 645], EPS), ([400, 800, 1600, 3200], EPS),
                                       ([127, 227, 146], 0.15), ([100, 200, 100], 0.2), ([400, 400], EPS)])
def test_alpha_star_is_exact_maximal_and_monotone(sizes, eps):
    """Along the candidate levels (delta and every chunk p-value in (delta, 0.5]) the k-vector and delta_batch are
    non-decreasing; the returned k-vector is the largest passing one, and the next candidate fails."""
    ds = _ns(sizes, eps)
    N = sum(sizes)
    alpha, designs, info = batch_spent_alpha(ds, N, eps, DELTA)
    kv = [d[2] for d in designs]
    cands = sorted({DELTA, *(float(p) for n, c in ds for p in np.unique(_pvalues(n, c, eps, "rectified"))
                             if DELTA < p <= 0.5)})
    prev_k, prev_db, passed = None, -1.0, []
    for a in cands:
        k = [k_used(n, c, eps, a, "rectified") for n, c in ds]
        db = batch_tail([(n, c, kk) for (n, c), kk in zip(ds, k)], N, eps)["delta_batch"]
        if prev_k is not None:
            assert all(x >= y for x, y in zip(k, prev_k)) and db >= prev_db - 1e-15
        prev_k, prev_db = k, db
        if db <= DELTA:
            passed.append(k)
    assert kv == passed[-1]                                 # the largest passing k-vector
    assert alpha >= DELTA and [k_used(n, c, eps, alpha, "rectified") for n, c in ds] == kv
    assert alpha == max([DELTA] + [float(_pvalues(n, c, eps, "rectified")[k]) for (n, c), k in zip(ds, kv) if k >= 0])
    # monotone in delta: a larger batch budget never lowers alpha* or any k_j
    out = [batch_spent_alpha(ds, N, eps, d) for d in (0.05, 0.1, 0.2)]
    for (a1, d1, _), (a2, d2, _) in zip(out, out[1:]):
        assert a2 >= a1 and all(y[2] >= x[2] for x, y in zip(d1, d2))
    for d, (a, dd, inf) in zip((0.05, 0.1, 0.2), out):
        assert inf["delta_batch"] <= d and batch_tail(dd, N, eps)["delta_batch"] <= d


def test_failing_delta_level_plan_is_returned_unchanged():
    ds = [(3, 8), (3, 8)]
    alpha, designs, info = batch_spent_alpha(ds, None, 0.3, 0.01)
    if info["ok"]:
        assert info["delta_batch"] <= 0.01
    else:
        assert alpha == 0.01 and [d[2] for d in designs] == info["k_chunk_level"]
    with pytest.raises(ValueError):
        batch_spent_alpha([], None, EPS, DELTA)
    with pytest.raises(ValueError):
        batch_spent_alpha(ds, None, EPS, 0.6)


# ----------------------------------------------------------------------------------------------- the stored plan
@pytest.mark.parametrize("kw", [{}, {"max_expected_eps": 0.03}, {"max_expected_eps": 0.045}, {"eps": 0.1},
                                {"tune": False}, {"chunk_growth": 0.5}, {"chunking": "auto"}])
@pytest.mark.parametrize("N,M0", [(4800, 400), (1000, 200), (2000, 300), (12000, 600), (755, 0)])
def test_stored_plan_has_delta_batch_le_delta_and_rule_k(kw, N, M0):
    b = _bottler(**kw)
    p = b.chunk_plan(N, M0)
    if p.get("chunks", 1) < 2:
        return
    cfg = b.cfg
    designs = [tuple(int(x) for x in d) for d in p["designs"]]
    assert p["delta_batch"] <= cfg.delta
    assert batch_tail(designs, N, cfg.eps)["delta_batch"] <= cfg.delta                 # exact re-check
    assert designs == [(c["n_cal"], c["N"], c["k"]) for c in p["risk"]["chunks"]]
    base = [b._chunk_design(c) for c in p["sizes"]]
    assert [d[:2] for d in designs] == [d[:2] for d in base]                            # n_j never re-tuned
    if p.get("spend") == "batch":
        a = p["alpha_chunk"]
        assert cfg.delta <= a <= 0.5
        for (n, Nj, k), (_, _, k0) in zip(designs, base):
            assert k == k_used(n, Nj, cfg.eps, a, cfg.certificate, max_expected_eps=cfg.max_expected_eps) >= k0
            if cfg.max_expected_eps is not None:
                assert k <= k_mean_cap(n, Nj, cfg.max_expected_eps)
        assert any(p["k_raise"])
    else:
        assert designs == base and "spend_note" in p                                    # nothing could be raised


def test_truncated_chunk_law_overflow_stays_in_delta_batch():
    """20,000 items into 400: the last chunk (7,600 items) has N_j - n_j > BATCH_FULL_SUPPORT_MAX, so its law is
    truncated. The raised plan's delta_batch still carries the truncated mass (a union bound), is <= delta by the exact
    re-check, and batch_spent_alpha's search used the same tail (overflow included)."""
    from bottlekit.certify import BATCH_FULL_SUPPORT_MAX, chunk_law
    b = _bottler()
    p = b.chunk_plan(20000, 400)
    assert p["spend"] == "batch" and any(Nj - n > BATCH_FULL_SUPPORT_MAX for n, Nj, _ in p["designs"])
    designs = [tuple(d) for d in p["designs"]]
    over = sum(chunk_law(n, Nj, k)[1] for n, Nj, k in designs)
    assert over > 0 and p["risk"]["overflow"] == pytest.approx(over, rel=1e-12)
    bt = batch_tail(designs, 20000, EPS)
    assert p["delta_batch"] == bt["delta_batch"] <= DELTA and p["delta_batch"] >= over
    alpha, ds, info = batch_spent_alpha([(n, Nj) for n, Nj, _ in p["designs_chunk_level"]], 20000, EPS, DELTA)
    assert ds == designs and info["delta_batch"] == bt["delta_batch"] and alpha == p["alpha_chunk"]


def test_mean_risk_cap_binds_every_chunk_so_nothing_is_raised():
    """max_expected_eps=0.03 caps every chunk below its delta-level k: no k can be raised, the plan is v0.8's."""
    b = _bottler(max_expected_eps=0.03)
    p = b.chunk_plan(4800, 400)
    assert "spend" not in p and "spend_note" in p and [list(d) for d in p["designs"]] == \
        [list(b._chunk_design(c)) for c in PLAN_4800]


def test_chunk_spend_validation_and_cli_default():
    assert BottleConfig(cheap="c", expert="e").chunk_spend == H.BottleConfig.chunk_spend
    assert H.CHUNK_SPENDS == ("chunk", "batch")
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", chunk_spend="both")
    p = _bottler(chunk_spend="chunk").chunk_plan(4800, 400)
    assert "spend" not in p and "alpha_chunk" not in p and "spend_note" not in p
    import argparse
    from bottlekit.__main__ import _cfg as cli_cfg
    ns = argparse.Namespace(cheap="c", expert="e", mode="auto", k=20, contexts=3, warmup=0, eps=0.05, delta=0.1,
                            calibration="sequential", n_cal=300, seed=0, no_balance=False, no_weights=False, fusion=None,
                            n_build=None, patience=None, target_coverage=None, looks=None, budget=None, spending=None,
                            no_tune=False)
    default = BottleConfig(cheap="c", expert="e").chunk_spend
    assert cli_cfg(ns).chunk_spend == default
    for v in ("chunk", "batch"):
        ns.chunk_spend = v
        assert cli_cfg(ns).chunk_spend == v


# ----------------------------------------------------------------------------------------------- 'chunk' == v0.8
def _v08_snapshot_dir():
    """The reference bottlekit WITHOUT batch-spent chunk plans: $BOTTLEKIT_PRE_CHUNKSPEND_SNAPSHOT (a dir holding
    bottlekit/: the code just before this feature, with forecast v2, the generality knobs and the health report);
    None if absent."""
    cands = [Path(p) for p in [os.environ.get("BOTTLEKIT_PRE_CHUNKSPEND_SNAPSHOT")] if p]
    for c in cands:
        if (c / "bottlekit" / "__init__.py").exists():
            return c / "bottlekit"
    return None


def _load_v08():
    base = _v08_snapshot_dir()
    if base is None:
        pytest.skip("the code without batch-spent plans is not present (set $BOTTLEKIT_PRE_CHUNKSPEND_SNAPSHOT)")
    name = "bottlekit_v08_snapshot"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, base / "__init__.py", submodule_search_locations=[str(base)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_chunk_spend_chunk_is_bit_identical_to_v08():
    """chunk_spend='chunk' (and a 'batch' plan where nothing can be raised) is the pour of the code without this
    feature (see _v08_snapshot_dir): answers, provenance, certificate, ledger, bottle and forecast, for a chunked
    cold pour, an auto pour and a warm pour."""
    v08 = _load_v08()
    wl = toy_workload(1400, seed=31)
    acc = {"cheap": 0.8, "expert": 0.95}
    kw = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=7)
    r_new = asyncio.run(Bottler(_client(wl, acc), SPEC, BottleConfig(**kw, chunking="off")).run(wl.texts[:200]))
    r_old = asyncio.run(v08.Bottler(_client(wl, acc), SPEC, v08.BottleConfig(**kw, chunking="off")).run(wl.texts[:200]))
    assert _out(r_new) == _out(r_old)
    T = wl.texts[200:1400]
    for votes in (False, "auto", True):
        a = asyncio.run(Bottler(_client(wl, acc), SPEC, BottleConfig(**kw, chunk_spend="chunk")).pour(
            r_new.artifact, T, votes=votes))
        o = asyncio.run(v08.Bottler(_client(wl, acc), SPEC, v08.BottleConfig(**kw)).pour(r_old.artifact, T, votes=votes))
        assert _out(a) == _out(o) and _art(a.artifact) == _art(o.artifact)
        if votes is False:
            assert a.certificate["chunked"]
    # a 'batch' plan with nothing to raise ([200, 200] at eps .1 into 200: k 4, 4 either way) is the v0.8 pour too
    kw2 = dict(kw, eps=0.1, chunk_cold_ratio=2.0)
    r2n = asyncio.run(Bottler(_client(wl, acc), SPEC, BottleConfig(**kw2, chunking="off")).run(wl.texts[:200]))
    r2o = asyncio.run(v08.Bottler(_client(wl, acc), SPEC, v08.BottleConfig(**kw2, chunking="off")).run(wl.texts[:200]))
    b = Bottler(_client(wl, acc), SPEC, BottleConfig(**kw2, chunk_spend="batch"))
    p = b.chunk_plan(400, len(r2n.artifact.texts))
    assert p["sizes"] == [200, 200] and "spend_note" in p
    a = asyncio.run(b.pour(r2n.artifact, wl.texts[200:600], votes=False))
    o = asyncio.run(v08.Bottler(_client(wl, acc), SPEC, v08.BottleConfig(**kw2)).pour(r2o.artifact, wl.texts[200:600],
                                                                                    votes=False))
    assert _out(a) == _out(o) and _art(a.artifact) == _art(o.artifact)


# ----------------------------------------------------------------------------------------------- end to end
def _setup(n_run=200, n_tot=1400, seed=31, eps=EPS, **kw):
    wl = toy_workload(n_tot, seed=seed)

    def mk(**k):
        return Bottler(_client(wl), SPEC, _cfg(**{"eps": eps, **kw, **k}))
    r0 = asyncio.run(mk(chunking="off").run(wl.texts[:n_run]))
    return wl, mk, r0


def test_batch_spent_pour_end_to_end():
    wl, mk, r0 = _setup()
    T = wl.texts[200:1400]
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no plan-mismatch warning
        res = asyncio.run(mk().pour(r0.artifact, T, votes=False))
    ref = asyncio.run(mk(chunk_spend="chunk").pour(r0.artifact, T, votes=False))
    c, cr = res.certificate, ref.certificate
    a = c["alpha_chunk"]
    assert c["chunked"] and c["plan_consistent"] and c["batch_claim"] and c["chunk_spend"] == "batch"
    assert c["plan"] == cr["plan"] == [355, 845] and c["partition_stream"] == cr["partition_stream"]
    assert c["delta_batch"] <= DELTA and c["delta_batch"] > cr["delta_batch"]
    assert c["delta_batch_chunk_level"] == pytest.approx(cr["delta_batch"])
    assert c["chunk_designs_chunk_level"] == cr["chunk_designs"]
    assert [d[:2] for d in c["chunk_designs"]] == [d[:2] for d in cr["chunk_designs"]]
    assert all(x[2] >= y[2] for x, y in zip(c["chunk_designs"], cr["chunk_designs"]))
    assert c["n_cal"] == cr["n_cal"]                        # the same calibration labels per chunk
    assert c["cal_stream"] == cr["cal_stream"]
    for j, (cd, (n, Nj, k)) in enumerate(zip(c["chunks"], c["chunk_designs"])):
        assert cd["delta"] == a and cd["N"] == Nj and cd["n_cal"] == n
        assert cd["chunk_design"]["k"] == cd["chunk_design"]["k_planned"] == k and cd["chunk_design"]["alpha"] == a
        r = cd["risk"]
        assert r["looks"] == [{"n_cal": n, "alpha": a, "k": k}] and r["delta"] == a
        assert r["delta_spent"] <= a + 1e-12
        assert f"{a:.4f}" in cd["guarantee"] and "batch-spent" in cd["guarantee"] and f"<= {a}" in cd["guarantee"]
    assert f"{c['delta_batch']:.4f}" in c["guarantee"] and f"{a:.4f}" in c["guarantee"]
    # chunk 1 starts from the same bottle with the same calibration sample: a larger k never bottles less there
    i1, j1 = res.diagnostics["chunk_index"][0], ref.diagnostics["chunk_index"][0]
    assert i1 == j1 and sum(res.provenance[i] == "bottle" for i in i1) >= sum(ref.provenance[i] == "bottle" for i in i1)
    # per-chunk risk sums to the pour's worst-case expectation (session's _expected_errors)
    from bottlekit.session import _cert_delta, _expected_errors
    ee = _expected_errors(c)
    assert ee == pytest.approx(c["risk"]["expected_eps_bound"] * len(T), rel=1e-9)
    assert _cert_delta(c, 0.5) == c["delta_batch"]
    # stream design of the batch: the INFLATED plan
    sd = H.stream_design(c, len(T))
    assert sd["ok"] and sd["designs"] == [[[[n, k]], Nj] for n, Nj, k in c["chunk_designs"]]


def test_batch_spent_pour_equals_the_manual_chain_of_designed_chunks():
    wl, mk, r0 = _setup()
    T = wl.texts[200:1400]
    b = mk()
    res = asyncio.run(b.pour(r0.artifact, T, votes=False))
    c = res.certificate
    M0, v0 = len(r0.artifact.texts), r0.artifact.version
    perm = np.random.default_rng(sub_seed(b.cfg.seed, 13, v0 + 1, M0)).permutation(len(T))
    bm = mk()
    art, off, ans, prov = r0.artifact, 0, [None] * len(T), [None] * len(T)
    for j, (n, Nj, k) in enumerate(c["chunk_designs"]):
        idx = np.sort(perm[off:off + Nj])
        off += Nj
        d = H._ChunkDesign(n, Nj, k, c["alpha_chunk"], c["delta_batch"], j, len(c["plan"]), token=H._CHUNK_TOKEN)
        r = asyncio.run(bm.pour(art, [T[i] for i in idx], votes=False, _chunk=True, _chunk_design=d))
        for jj, i in enumerate(idx):
            ans[i], prov[i] = r.answers[jj], r.provenance[jj]
        art = r.artifact
    assert ans == res.answers and prov == res.provenance and _art(art) == _art(res.artifact)


def test_forecast_of_a_batch_spent_chunk_is_at_the_standard_design():
    """The returned bottle's forecast (used by the next votes='auto' decision) is the EXACT standard-design (eps,
    delta) outcome on the last chunk's calibration sample, never the chunk-level (alpha*) coverage."""
    wl, mk, r0 = _setup()
    T = wl.texts[200:1400]
    res = asyncio.run(mk().pour(r0.artifact, T, votes=False))
    f = res.artifact.forecast
    last = res.certificate["chunks"][-1]
    cold = f["arms"]["cold"]
    assert f["chunk_level"] == res.certificate["alpha_chunk"] and cold["basis"] == "exact (standard design)"
    assert cold["n_cal"] == last["n_cal"]
    idx = res.diagnostics["chunk_index"][-1]
    realized = sum(res.provenance[i] == "bottle" for i in idx) / len(idx)
    assert cold["bottled"] <= realized + 1e-12               # k at delta <= k at alpha*: never more bottled
    assert cold["tau"] is None or last["tau"] is None or cold["tau"] >= last["tau"]
    # the standard-design arm equals the v0.8 (delta-level) chunk pour from the same bottle: same scores, same
    # calibration key and sample, k at delta
    art_before, b = None, mk()
    M0, v0 = len(r0.artifact.texts), r0.artifact.version
    perm = np.random.default_rng(sub_seed(b.cfg.seed, 13, v0 + 1, M0)).permutation(len(T))
    art, off = r0.artifact, 0
    c = res.certificate
    for j, (n, Nj, k) in enumerate(c["chunk_designs"]):
        idx_j = np.sort(perm[off:off + Nj])
        off += Nj
        if j == len(c["chunk_designs"]) - 1:
            art_before = art
            break
        d = H._ChunkDesign(n, Nj, k, c["alpha_chunk"], c["delta_batch"], j, len(c["plan"]), token=H._CHUNK_TOKEN)
        art = asyncio.run(b.pour(art, [T[i] for i in idx_j], votes=False, _chunk=True, _chunk_design=d)).artifact
    std = asyncio.run(mk(chunk_spend="chunk").pour(art_before, [T[i] for i in idx_j], votes=False, _chunk=True))
    assert std.certificate["delta"] == DELTA and std.certificate["n_cal"] == cold["n_cal"]
    assert cold["bottled"] == pytest.approx(std.coverage(), abs=1e-12)
    assert cold["tau"] == std.certificate["tau"]


def test_private_chunk_design_cannot_be_set_by_a_public_pour():
    wl, mk, r0 = _setup()
    T = wl.texts[200:600]
    b = mk()
    good = dict(n=single_look_size(400, EPS, DELTA), N=400, k=5, alpha=0.2, delta_batch=0.09, j=0, J=1)
    for bad_kw in ({"_chunk_design": object(), "_chunk": True},
                   {"_chunk_design": H._ChunkDesign(**good), "_chunk": True},                     # no token
                   {"_chunk_design": H._ChunkDesign(**good, token=H._CHUNK_TOKEN)},               # not a chunk
                   {"_chunk_design": H._ChunkDesign(**{**good, "alpha": 0.6}, token=H._CHUNK_TOKEN), "_chunk": True},
                   {"_chunk_design": H._ChunkDesign(**{**good, "alpha": 0.05}, token=H._CHUNK_TOKEN), "_chunk": True},
                   {"_chunk_design": H._ChunkDesign(**{**good, "N": 401}, token=H._CHUNK_TOKEN), "_chunk": True}):
        with pytest.raises(ValueError, match="private"):
            asyncio.run(b.pour(r0.artifact, T, votes=False, **bad_kw))
    with pytest.raises(ValueError, match="private"):       # never under chunk_spend='chunk'
        asyncio.run(mk(chunk_spend="chunk").pour(r0.artifact, T, votes=False, _chunk=True,
                                                 _chunk_design=H._ChunkDesign(**good, token=H._CHUNK_TOKEN)))


def test_plan_mismatch_withdraws_the_whole_batch_claim(monkeypatch):
    """If a chunk applied another k than the plan (simulated by a plan whose stored k is off by one), the whole-batch
    claim is withdrawn (warning, delta_batch None, no risk) and the stream ledger refuses the batch."""
    wl, mk, r0 = _setup()
    T = wl.texts[200:1400]
    real = H.Bottler.chunk_plan

    def drift(self, N, M0):
        p = real(self, N, M0)
        if p.get("spend") == "batch":
            p["designs"] = [(n, Nj, k + 1) if j == 0 else (n, Nj, k) for j, (n, Nj, k) in enumerate(p["designs"])]
        return p
    monkeypatch.setattr(H.Bottler, "chunk_plan", drift)
    with pytest.warns(UserWarning, match="differ from the plan"):
        res = asyncio.run(mk().pour(r0.artifact, T, votes=False))
    c = res.certificate
    assert not c["plan_consistent"] and not c["batch_claim"] and c["delta_batch"] is None
    assert c["risk"]["basis"] == "unavailable" and "per-chunk only" in c["guarantee"] and "its own certificate" in c["guarantee"]
    assert not H.stream_design(c, len(T))["ok"]
    # only per-chunk claims at alpha* remain: the pour's 'delta' says so, and the session books it at alpha* (never at
    # cfg.delta)
    from bottlekit.session import _cert_delta
    assert c["delta"] == c["alpha_chunk"] > DELTA and _cert_delta(c, DELTA) == c["alpha_chunk"]


def test_chunk_alpha_max_ceiling():
    """cfg.chunk_alpha_max caps alpha*: validated in (0, 1); a ceiling <= delta leaves the v0.8 plan; a binding
    ceiling gives a smaller (still exact, still <= delta) k-vector; pour() refuses a chunk design above the ceiling."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            BottleConfig(cheap="c", expert="e", chunk_alpha_max=bad)
    assert BottleConfig(cheap="c", expert="e").chunk_alpha_max == 0.5
    free = _bottler().chunk_plan(4800, 400)
    p0 = _bottler(chunk_alpha_max=DELTA).chunk_plan(4800, 400)
    p1 = _bottler(chunk_alpha_max=0.05).chunk_plan(4800, 400)
    for p in (p0, p1):
        assert "spend" not in p and "spend_note" in p and [list(d) for d in p["designs"]] == free["designs_chunk_level"]
    p2 = _bottler(chunk_alpha_max=0.2).chunk_plan(4800, 400)
    a2 = p2["alpha_chunk"]
    assert DELTA <= a2 <= 0.2 < free["alpha_chunk"]
    assert p2["delta_batch"] <= free["delta_batch"] <= DELTA
    for (n, Nj, k), (_, _, kf), (_, _, k0) in zip(p2["designs"], free["designs"], free["designs_chunk_level"]):
        assert k0 <= k <= kf and k == k_used(n, Nj, EPS, a2, "rectified")
    alpha, designs, _ = batch_spent_alpha([(n, Nj) for n, Nj, _ in free["designs_chunk_level"]], 4800, EPS, DELTA,
                                          alpha_max=0.2)
    assert alpha == a2 and designs == [tuple(d) for d in p2["designs"]]
    # the private design is refused above the configured ceiling
    wl, mk, r0 = _setup()
    T = wl.texts[200:600]
    d = H._ChunkDesign(single_look_size(400, EPS, DELTA), 400, 5, 0.3, 0.09, 0, 1, token=H._CHUNK_TOKEN)
    with pytest.raises(ValueError, match="private"):
        asyncio.run(mk(chunk_alpha_max=0.2).pour(r0.artifact, T, votes=False, _chunk=True, _chunk_design=d))
    # CLI
    import argparse
    from bottlekit.__main__ import _cfg as cli_cfg
    ns = argparse.Namespace(cheap="c", expert="e", mode="auto", k=20, contexts=3, warmup=0, eps=0.05, delta=0.1,
                            calibration="sequential", n_cal=300, seed=0, no_balance=False, no_weights=False, fusion=None,
                            n_build=None, patience=None, target_coverage=None, looks=None, budget=None, spending=None,
                            no_tune=False, chunk_alpha_max=0.3)
    assert cli_cfg(ns).chunk_alpha_max == 0.3
    ns.chunk_alpha_max = None
    assert cli_cfg(ns).chunk_alpha_max == 0.5


def test_chunking_auto_batch_spend_designs_do_not_depend_on_votes():
    wl, mk, r0 = _setup(chunking="auto")
    T = wl.texts[200:1400]
    outs = [asyncio.run(mk(chunking="auto").pour(r0.artifact, T, votes=v)) for v in (False, True, "auto")]
    for o in outs:
        c = o.certificate
        assert c["chunked"] and c["plan_consistent"] and c["chunk_spend"] == "batch"
        assert c["chunk_designs"] == outs[0].certificate["chunk_designs"]
        assert c["alpha_chunk"] == outs[0].certificate["alpha_chunk"]
        assert all(cd["delta"] == c["alpha_chunk"] for cd in c["chunks"])
    assert outs[1].diagnostics["mode"] == "pour_warm"
    f = outs[1].artifact.forecast                           # warm chunk: both arms at the standard design
    assert f["source"] == "pour_warm" and f["arms"]["warm"]["basis"] == "exact (standard design)"
    assert f["arms"]["cold"]["basis"] == "exact" and f["chunk_level"] == outs[1].certificate["alpha_chunk"]


def test_session_stream_ledger_stays_exact_with_the_inflated_k(tmp_path):
    from bottlekit import BottleSession
    from bottlekit.certify import StreamLaw
    wl = toy_workload(1400, seed=41)
    gold = {t[:400]: g for t, g in zip(wl.texts, wl.gold)}
    cfg = _cfg(seed=17)
    s = BottleSession(MockClient(gold, acc={"cheap": 0.82, "expert": 0.95}), SPEC, cfg, state_path=tmp_path / "s.json",
                      min_batch=0, votes=False)
    for t in (wl.texts[:200], wl.texts[200:1200], wl.texts[1200:1400]):
        asyncio.run(s.submit(t))
    c1 = s.batches[1]["certificate"]
    assert c1["chunked"] and c1["chunk_spend"] == "batch" and c1["plan"] == [355, 645]
    assert any(x[2] > y[2] for x, y in zip(c1["chunk_designs"], c1["chunk_designs_chunk_level"]))
    rep = s.report()
    sr = rep["totals"]["stream_risk"]
    assert sr["status"] == "exact" and sr["batches_included"] == 3
    assert s._stream[1]["d"] == [[[[n, k]], Nj] for n, Nj, k in c1["chunk_designs"]]       # the inflated k
    law = StreamLaw()
    for e in s._stream:
        for looks, Nj in e["d"]:
            law.add_design(looks, Nj)
    assert sr["delta_stream"] == pytest.approx(law.tail(math.floor(EPS * 1400)))
    assert rep["rows"][1]["delta"] == pytest.approx(c1["delta_batch"])
    assert rep["totals"]["expected_failing_batches_bound"] == pytest.approx(
        sum(r["delta"] for r in rep["rows"]))
    # resume rebuilds the same ledger
    r = BottleSession.resume(tmp_path / "s.json", MockClient(gold), SPEC, cfg)
    assert r.report()["totals"]["stream_risk"]["delta_stream"] == pytest.approx(sr["delta_stream"])


# ----------------------------------------------------------------------------------------------- exhaustive, adaptive
def _rule_U(B, S, k):
    """Closed-form k-rule, items 0..N-1 in DESCENDING score order: #unlabelled bad items strictly above the (k+1)-th
    highest labelled bad item (all unlabelled bad if <= k labelled bad)."""
    lab_bad = sorted(i for i in S if i in B)
    if len(lab_bad) <= k:
        return sum(1 for b in B if b not in S)
    cut = lab_bad[k]
    return sum(1 for b in B if b not in S and b < cut)


@pytest.mark.parametrize("ds,eps,delta", [([(5, 10), (5, 10)], 0.3, 0.1), ([(5, 10), (5, 10)], 0.25, 0.2),
                                          ([(3, 8), (4, 10)], 0.3, 0.2), ([(3, 8), (3, 8), (3, 8)], 0.3, 0.1),
                                          ([(2, 6), (3, 8), (3, 8)], 0.3, 0.2), ([(4, 10), (4, 10)], 0.25, 0.1)])
def test_exhaustive_adaptive_enumeration_at_the_inflated_k(ds, eps, delta):
    """Exact worst case over ADAPTIVE adversaries (chunk j's bad set chosen after seeing everything before it; the
    optimal choice depends only on the remaining allowance r): V_{J+1}(r) = 1[r < 0], V_j(r) = max over EVERY bad set
    B_j of the mean over EVERY calibration subset S_j of V_{j+1}(r - U_j). The result V_1(m) is <= the convolution
    tail P[V_1 + ... + V_J > m] at the inflated k_j for every m, with equality (the all-bad workload attains it), and
    at m = floor(eps N) it is <= delta."""
    alpha, designs, info = batch_spent_alpha(ds, None, eps, delta)
    assert info["ok"] and [d[2] for d in designs] != info["k_chunk_level"]      # the spend raised some k
    T = sum(N for _, N, _ in designs)
    laws = []                                               # per chunk: {bad set size pattern: U distribution}
    for n, N, k in designs:
        Ss = [list(c) for c in itertools.combinations(range(N), n)]
        dists = set()
        for r in range(N + 1):
            for B in itertools.combinations(range(N), r):
                Bs = set(B)
                cnt = np.bincount([_rule_U(Bs, S, k) for S in Ss], minlength=N + 1) / len(Ss)
                dists.add(tuple(np.round(cnt, 15)))
        laws.append([np.array(d) for d in dists])
    V = lambda r: 1.0 if r < 0 else 0.0                     # noqa: E731
    vals = {r: V(r) for r in range(-T - 1, T + 1)}
    for dist_list in reversed(laws):
        new = {}
        for r in range(-T - 1, T + 1):
            new[r] = max(float(sum(p * vals.get(r - u, 1.0 if r - u < 0 else 0.0) for u, p in enumerate(d) if p))
                         for d in dist_list)
        vals = new
    bt = batch_tail(designs, eps=eps, survival=True)
    s = list(bt["survival"]) + [0.0] * (T + 2)
    for m in range(T + 1):
        assert vals[m] <= s[m] + 1e-12, (m, vals[m], s[m])
        assert abs(vals[m] - s[m]) <= 1e-12                 # attained
    m = math.floor(eps * T)
    assert vals[m] <= delta and abs(bt["delta_batch"] - vals[m]) <= 1e-12


# ----------------------------------------------------------------------------------------------- Monte Carlo
def _labelled_positions(rng, D, n, N):
    return np.sort(np.argsort(rng.random((D, N)), axis=1)[:, :n], axis=1)


def test_rule_level_monte_carlo_on_the_inflated_4800_plan():
    """>= 300 draws (4,000) of the inflated plan k = 5, 7, 11, 12 under (a) the all-bad workload (attains every g_j
    at once: violation = delta_batch) and (b) an ADAPTIVE best response, chunk j's bad set = the top need+1+k_j items
    for the allowance left after chunks 1..j-1. Whole-batch violation <= delta + 2 se and matches delta_batch."""
    ds = _ns(PLAN_4800)
    alpha, designs, info = batch_spent_alpha(ds, 4800, EPS, DELTA)
    t = math.floor(EPS * 4800)
    D = 4000
    rng = np.random.default_rng(2026)
    pos = [_labelled_positions(rng, D, n, N) for n, N, _ in designs]
    U_all = sum(p[:, k] - k for p, (_, _, k) in zip(pos, designs))
    rate_a = float((U_all > t).mean())
    used = np.zeros(D, np.int64)
    for p, (n, N, k) in zip(pos, designs):
        need = t - used
        b = np.clip(need + 1 + k, 0, N)
        Lb = (p < b[:, None]).sum(axis=1)
        U = np.where(Lb <= k, b - Lb, np.minimum(p[:, min(k, n - 1)], b) - k)
        U = np.where(need < 0, 0, U)
        used = used + U
    rate_b = float((used > t).mean())
    se = math.sqrt(DELTA * (1 - DELTA) / D)
    db = info["delta_batch"]
    for rate in (rate_a, rate_b):
        assert rate <= DELTA + 2 * se, (rate, db)
        assert abs(rate - db) <= 4 * math.sqrt(db * (1 - db) / D) + 1e-3, (rate, db)


class _Adv(MockClient):
    """Adaptive adversarial EXPERT for chunked pours. Before chunk j starts, the test's pour hook records the chunk's
    bottle predictions (a function of F_{j-1} only: the bottle exported by chunk j-1). The expert then disagrees with
    the bottle on a bad set chosen from that chunk's own score order:
      'all'   : every item (attains the worst-case law of every chunk at once);
      'top'   : the top b_j = floor(eps N_j) + 1 + k_j items by bottle confidence (the per-chunk attaining size);
      'below' : the next b_j items just below them (good items on top, the bad block sits where tau falls)."""
    rule = "all"
    eps = 0.15
    pred: dict = {}

    def _answer(self, alias, text, salt, pos=None, k=None):
        if alias == "expert" and salt == "single" and text in self.pred:
            p, rank, Nj, kj = self.pred[text]
            b = math.floor(self.eps * Nj) + 1 + kj
            bad = (self.rule == "all" or (self.rule == "top" and rank < b)
                   or (self.rule == "below" and b <= rank < 2 * b))
            if bad:
                return next(n for n in SPEC.names if n != p)
            return p
        return super()._answer(alias, text, salt, pos, k)


@pytest.mark.parametrize("rule", ["all", "below"])
def test_harness_simulation_300_draws_adaptive_adversarial_expert(rule, monkeypatch):
    """300 chunked batch-spent pours (eps .15: 500 items into a 100-item bottle, plan [127, 227, 146], k 4, 6, 4 ->
    5, 8, 5 at alpha* 0.2497, delta_batch 0.0961), each with its own seed (fresh partition and calibration draws), and
    an adaptive adversarial expert. Whole-batch violation (share of final answers != expert > eps) <= delta + 2 se,
    and <= delta_batch + 3 se."""
    eps = 0.15
    wl = toy_workload(600, seed=88)
    gold = {t[:400]: g for t, g in zip(wl.texts, wl.gold)}
    client = _Adv(gold, acc={"cheap": 0.85, "expert": 0.97})
    client.rule, client.eps, client.pred = rule, eps, {}
    r0 = asyncio.run(Bottler(client, SPEC, _cfg(eps=eps, seed=0, chunking="off")).run(wl.texts[:100]))
    T = wl.texts[100:600]
    real = H.Bottler.pour

    async def hook(self, artifact, texts, votes="auto", emb=None, **kw):
        cd = kw.get("_chunk_design")
        if kw.get("_chunk") and cd is not None:
            pred, conf = artifact.predict(texts)
            order = np.argsort(-np.asarray(conf, float), kind="stable")
            rank = {int(i): r for r, i in enumerate(order)}
            for i, (t, p) in enumerate(zip(texts, pred)):
                client.pred[t] = (p, rank[i], len(texts), int(cd.k))
        return await real(self, artifact, texts, votes=votes, emb=emb, **kw)
    monkeypatch.setattr(H.Bottler, "pour", hook)
    plan = Bottler(client, SPEC, _cfg(eps=eps, seed=1)).chunk_plan(len(T), len(r0.artifact.texts))
    assert plan["sizes"] == [127, 227, 146] and [d[2] for d in plan["designs"]] == [5, 8, 5]
    db = plan["delta_batch"]
    assert 0.09 < db <= DELTA
    D, viol, u_sum = 300, 0, 0.0
    for d in range(D):
        client.pred = {}
        res = asyncio.run(Bottler(client, SPEC, _cfg(eps=eps, seed=5000 + d)).pour(r0.artifact, T, votes=False))
        c = res.certificate
        assert c["chunked"] and c["plan_consistent"] and c["delta_batch"] == db
        ex = [client._answer("expert", t, "single") for t in T]
        bad = [a != e for a, e in zip(res.answers, ex)]
        viol += float(np.mean(bad)) > eps
        u_sum += sum(1 for a, p, e in zip(res.answers, res.provenance, ex) if p == "bottle" and a != e) / len(T)
    rate = viol / D
    print(f"[harness sim rule={rule}] violations {viol}/{D} = {rate:.4f}; delta_batch {db:.4f}; mean U/N "
          f"{u_sum / D:.4f} vs worst-case expected {c['risk']['expected_eps_bound']:.4f}")
    assert rate <= DELTA + 2 * math.sqrt(DELTA * (1 - DELTA) / D), rate
    assert rate <= db + 3 * math.sqrt(db * (1 - db) / D), (rate, db)
    assert u_sum / D <= c["risk"]["expected_eps_bound"] + 0.01
    if rule == "all":                                       # the worst case is attained (up to ties): rate ~ db
        assert rate >= db - 3.5 * math.sqrt(db * (1 - db) / D), (rate, db)
