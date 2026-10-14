"""Tests for recurrence-aware bottling (BottleConfig.memo=True; opt-in). No servers needed.

  (a) the population-restricted rectified rule with an EXPLICIT allowance m: exhaustive enumeration (every memo subset
      of a small batch, every calibration subset of the rest, bad sets, score orders, ties), the implementation
      against the vectorised rule, the budget inversion with a batch-level denominator, and Monte Carlo at bench sizes
  (b) propagating calibration answers to identical prompts never increases output errors and leaves tau unchanged
  (c) the memo is off for a fingerprint mismatch, src='bottle', '+fallback' and NONE answers, old pickles, memo=False
  (d) src and the fingerprint survive save/load, dedupe (newest wins), the max_memory cap, updated(), with_embeddings()
  (e) expert-call dedupe under a deterministic mock expert: same answers, one call per unique routed prompt
  (f) N_R = 0 (every item from memory, cold and warm), tiny N_R, the trivial allowance m >= N_R
  plus group-aware cross-fitting (singletons reproduce the ungrouped folds; copies never scored by a twin-trained model),
  the calibration permutation over R's own index space, and an end-to-end validity check on a recurring mock stream.
"""

import asyncio
import hashlib
import itertools
import math
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
from bottlekit import NONE, BottleArtifact, BottleConfig, Bottler, ModelSpec, TfidfLR, cross_fit  # noqa: E402
from bottlekit.certify import (_pvalues, certified_eps, fixed_sequence, k_critical, single_look_design,  # noqa: E402
                               tune_size)
from bottlekit.harness import expert_key  # noqa: E402
from bottlekit.labeling import sub_seed  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402
from test_bottlekit_v3 import _errors_vec, _subsets, _tie_patterns  # noqa: E402

DELTA = 0.1


def _client(wl, acc=None):
    return MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc=acc or {"cheap": 0.8, "expert": 0.95})


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=5, memo=True)
    base.update(kw)
    return BottleConfig(**base)


def _expert_of(client, texts):
    return [client._answer("expert", expert_key(SPEC, t), "single") for t in texts]


def _calls(res, phases=("expert_calibration", "expert_escalation")):
    return sum(res.ledger.get(p, {}).get("calls", 0) for p in phases)


# ================================================================================================ (a) certificate
def test_explicit_m_reproduces_floor_eps_n_and_is_monotone_in_m():
    for (n, N, eps) in [(5, 12, 0.2), (117, 400, 0.05), (66, 229, 0.05), (199, 1200, 0.05), (16, 16, 0.3)]:
        m = int(math.floor(eps * N))
        np.testing.assert_array_equal(_pvalues(n, N, eps, "rectified"), _pvalues(n, N, 0.77, "rectified", m))
        np.testing.assert_array_equal(_pvalues(n, N, eps, "hypergeometric"), _pvalues(n, N, 0.77, "hypergeometric", m))
        assert k_critical(n, N, eps, DELTA, "rectified") == k_critical(n, N, 0.77, DELTA, "rectified", m)
        assert tune_size(n, N, eps, DELTA, "rectified") == tune_size(n, N, 0.77, DELTA, "rectified", m=m)
        ks = [k_critical(n, N, eps, DELTA, "rectified", mm) for mm in range(0, min(N, m + 6))]
        assert ks == sorted(ks)                                      # a larger allowance never certifies less
    with pytest.raises(ValueError):
        _pvalues(10, 40, 0.1, "binomial", 3)
    with pytest.raises(ValueError):
        _pvalues(10, 40, 0.1, "rectified", -1)


@pytest.mark.parametrize("N_R", [7, 10, 12])
def test_exhaustive_explicit_allowance_exact_validity(N_R):
    """P[more than m unlabelled bad items of the population bottled] <= alpha EXACTLY for every explicit m (not tied to
    any eps), every calibration size, every calibration subset, every number of bad items and every tie pattern; and the
    worst case attains the closed form (non-vacuous)."""
    rng = np.random.default_rng(N_R)
    pats = _tie_patterns(N_R, rng)
    worst_ratio = 0.0
    for n in range(1, N_R):
        cal = _subsets(N_R, n)
        for m in range(0, N_R):
            for method in ("rectified", "hypergeometric"):
                for alpha in (0.05, 0.1, 0.3):
                    k = k_critical(n, N_R, 0.5, alpha, method, m)
                    if k < 0:
                        continue
                    for name, score in pats.items():
                        order = np.argsort(-score, kind="stable")
                        worst = 0.0
                        for M in range(N_R + 1):
                            bad = np.zeros(N_R, bool)
                            bad[order[:M]] = True
                            p_err = float(np.mean(_errors_vec(cal, bad, score, k) > m))
                            assert p_err <= alpha + 1e-12, (N_R, n, m, method, alpha, name, M, p_err)
                            worst = max(worst, p_err)
                        if name == "distinct" and method == "rectified":
                            assert worst == pytest.approx(_pvalues(n, N_R, 0.5, "rectified", m)[k], abs=1e-9)
                        worst_ratio = max(worst_ratio, worst / alpha)
    assert worst_ratio > 0.5


def test_exhaustive_over_every_memo_subset_of_a_small_batch():
    """Batch of 8 items. For EVERY memo subset (2^8), the remaining population R is certified with the harness's
    allowance rule ('novel': floor(eps*N_R); 'batch': floor(eps*N)) at the harness's single-look size, over every
    calibration subset of R: P[total output errors > m] <= alpha, where memo items add 0 errors (deterministic expert),
    and the batch share bound floor(eps*N)/N holds too. Bad sets: the top items of R, a random set, and all of R."""
    N, rng = 8, np.random.default_rng(3)
    score_full = rng.integers(0, 4, N) / 3.0                       # ties
    checked = 0
    for r in range(N + 1):
        for memo in itertools.combinations(range(N), r):
            R = [i for i in range(N) if i not in memo]
            N_R = len(R)
            if N_R < 2:
                continue
            score = score_full[R]
            order = np.argsort(-score, kind="stable")
            bads = [np.zeros(N_R, bool), np.ones(N_R, bool), rng.random(N_R) < 0.5]
            for M in range(1, N_R):
                b = np.zeros(N_R, bool)
                b[order[:M]] = True
                bads.append(b)
            for eps in (0.15, 0.3):
                for rule in ("novel", "batch"):
                    m = int(math.floor(eps * (N if rule == "batch" else N_R)))
                    eps_design = max(eps, m / N_R)
                    for alpha in (0.1, 0.3):
                        if m >= N_R:
                            continue                                # trivial: no calibration (tested separately)
                        n = single_look_design(N_R, eps_design)[0][0]
                        n = tune_size(n, N_R, eps_design, alpha, "rectified", m=m)
                        k = k_critical(n, N_R, eps_design, alpha, "rectified", m)
                        if k < 0 or n >= N_R:
                            continue
                        cal = _subsets(N_R, n)
                        for bad in bads:
                            err = _errors_vec(cal, bad, score, k)   # memo items: 0 output errors
                            assert np.mean(err > m) <= alpha + 1e-12, (memo, eps, rule, alpha, bad)
                            if rule == "batch":
                                assert np.mean(err > math.floor(eps * N)) <= alpha + 1e-12
                            checked += 1
    assert checked > 1000


def test_fixed_sequence_with_explicit_m_matches_the_vectorised_rule():
    rng = np.random.default_rng(1)
    N = 9
    for name, score in _tie_patterns(N, rng).items():
        for n in (3, 5):
            cal = _subsets(N, n)
            for M in (0, 3, 6, 9):
                bad = np.zeros(N, bool)
                bad[rng.choice(N, M, replace=False)] = True
                for m, alpha in ((0, 0.3), (2, 0.1), (4, 0.2)):
                    k = k_critical(n, N, 0.5, alpha, "rectified", m)
                    vec = _errors_vec(cal, bad, score, k) if k >= 0 else np.zeros(len(cal), int)
                    for r in range(len(cal)):
                        idx = np.nonzero(cal[r])[0]
                        tau, _ = fixed_sequence(score[idx], bad[idx], N, 0.5, alpha, "rectified", m=m)
                        acc = score >= tau
                        acc[idx] = False
                        assert int((acc & bad).sum()) == int(vec[r])


@pytest.mark.parametrize("N_tot,N_R,rule", [(400, 229, "novel"), (400, 229, "batch"), (400, 88, "batch"),
                                             (1200, 500, "batch")])
def test_restricted_rule_monte_carlo_adversarial(N_tot, N_R, rule):
    """Bench-sized populations with the harness's allowance and look size; adversarial workloads (|B| = m+1+j bad
    items at the top for several j, worst case of the closed form)."""
    eps = 0.05
    m = int(math.floor(eps * (N_tot if rule == "batch" else N_R)))
    eps_design = max(eps, m / N_R)
    n = tune_size(single_look_design(N_R, eps_design)[0][0], N_R, eps_design, DELTA, "rectified", m=m)
    k = k_critical(n, N_R, eps_design, DELTA, "rectified", m)
    rng = np.random.default_rng(N_R)
    R = 4000
    for j in (0, k // 2, k):
        Mb = m + 1 + j
        if Mb > N_R:
            continue
        bad = np.zeros(N_R, bool)
        bad[:Mb] = True
        score = np.linspace(1, 0, N_R)
        viol = 0
        for _ in range(R):
            idx = rng.choice(N_R, n, replace=False)
            tau, _ = fixed_sequence(score[idx], bad[idx], N_R, eps, DELTA, "rectified", m=m)
            acc = score >= tau
            acc[idx] = False
            viol += int((acc & bad).sum()) > m
        exact = float(hypergeom.cdf(j, N_R, Mb, n))                  # = v_j: the exact error probability here
        assert exact <= DELTA + 1e-12
        assert viol / R <= exact + 4 * math.sqrt(exact * (1 - exact) / R) + 1e-3, (j, viol / R, exact)


def test_budget_inversion_with_batch_denominator_covers_the_batch_share():
    """certified_eps(denom=N_tot) bounds (output errors in R)/N_tot (memo items add none) with prob >= 1 - delta."""
    rng = np.random.default_rng(7)
    N_R, N_tot, n, tau = 200, 330, 60, 0.4
    score = rng.random(N_R)
    for M in (5, 20, 60):
        bad = np.zeros(N_R, bool)
        bad[rng.choice(N_R, M, replace=False)] = True
        cover, R = 0, 1500
        for _ in range(R):
            idx = rng.choice(N_R, n, replace=False)
            e = certified_eps(score[idx], bad[idx], tau, N_R, DELTA, "rectified", denom=N_tot)
            acc = score >= tau
            acc[idx] = False
            cover += int((acc & bad).sum()) / N_tot <= e + 1e-12
        assert cover / R >= 1 - DELTA - 3 * math.sqrt(DELTA * (1 - DELTA) / R)
    assert certified_eps([0.5], [False], 0.1, 10, 0.1, "hypergeometric") == certified_eps(
        [0.5], [False], 0.1, 10, 0.1, "hypergeometric", denom=10)
    with pytest.raises(ValueError):
        certified_eps([0.5], [False], 0.1, 10, 0.1, "rectified", denom=9)


# ================================================================================================ helpers: workloads
def _zipf_batch(texts, n_slots, n_unique, rng, s=1.0):
    pool = [texts[i] for i in rng.choice(len(texts), n_unique, replace=False)]
    w = 1.0 / np.arange(1, n_unique + 1) ** s
    return [pool[i] for i in rng.choice(n_unique, n_slots, p=w / w.sum())]


# ================================================================================================ (b) propagation
def test_propagation_never_increases_output_errors_and_keeps_tau():
    wl = toy_workload(600, seed=21)
    for s in range(4):
        rng = np.random.default_rng(s)
        Z = _zipf_batch(wl.texts, 300, 120, rng)
        out = {}
        for prop in (False, True):
            client = _client(wl, acc={"cheap": 0.7, "expert": 0.9})
            b = Bottler(client, SPEC, _cfg(seed=30 + s))
            b._propagate = prop
            out[prop] = asyncio.run(b.run(Z))
        a, p = out[False], out[True]
        assert a.certificate["tau"] == p.certificate["tau"] and a.certificate["n_cal"] == p.certificate["n_cal"]
        exp = _expert_of(client, Z)
        err = {k: sum(x != e for x, e in zip(v.answers, exp)) for k, v in out.items()}
        assert err[True] <= err[False]
        dup = [i for i, q in enumerate(p.provenance) if q == "expert_calibration_dup"]
        assert dup and all(p.answers[i] == exp[i] for i in dup)
        for i in range(len(Z)):
            if i not in dup:
                assert p.answers[i] == a.answers[i] and p.provenance[i] == a.provenance[i]
        cal_keys = {expert_key(SPEC, Z[i]) for i, q in enumerate(p.provenance) if q == "expert_calibration"}
        assert all(expert_key(SPEC, Z[i]) in cal_keys for i in dup)
        assert p.certificate["n_dup_propagated"] == len(dup)
        assert "deterministic expert" in p.certificate["guarantee"]


# ================================================================================================ (c) memo off
def _run_art(wl, n=300, **kw):
    client = _client(wl)
    r0 = asyncio.run(Bottler(client, SPEC, _cfg(**kw)).run(wl.texts[:n]))
    return client, r0


def test_memo_answers_only_expert_sourced_items_with_matching_fingerprint():
    wl = toy_workload(500, seed=22)
    client, r0 = _run_art(wl)
    art = r0.artifact
    assert art.fingerprint and len(art.src) == len(art.texts)
    prov_of = {t: p for t, p in zip(wl.texts[:300], r0.provenance)}
    for t, s in zip(art.texts, art.src):
        assert s == ("expert" if prov_of[t] in ("expert_calibration", "expert_escalation") else "bottle")
    rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=6)).pour(art, wl.texts[:300], votes=False))
    memo = [i for i, p in enumerate(rp.provenance) if p == "memo"]
    assert set(memo) == {i for i, p in enumerate(r0.provenance) if p in ("expert_calibration", "expert_escalation")}
    assert all(rp.answers[i] == r0.answers[i] for i in memo)
    assert rp.certificate["n_memo"] == len(memo) and rp.certificate["N_population"] == 300 - len(memo)
    assert rp.memo_share() == pytest.approx(len(memo) / 300) and rp.coverage() <= 1 - rp.memo_share()
    assert "deterministic expert" in rp.certificate["guarantee"]


def test_memo_off_on_fingerprint_mismatch_old_pickle_and_config(tmp_path):
    wl = toy_workload(500, seed=23)
    client, r0 = _run_art(wl)
    art, T = r0.artifact, wl.texts[:300]
    # instruction changed -> different expert prompt
    spec2 = H.TaskSpec(SPEC.name, SPEC.instruction + " Be careful.", SPEC.labels)
    rp = asyncio.run(Bottler(client, spec2, _cfg(seed=6)).pour(art, T, votes=False))
    assert "memo" not in rp.provenance and rp.diagnostics["memo"].startswith("off: prompt fingerprint mismatch")
    # expert model identity changed (same alias, other instance id / base_url / extra_body / max_chars)
    for change in ({"ids": ["expert-v2"]}, {"base_url": "http://other/v1"}, {"extra_body": {"reasoning_effort": "high"}}):
        c2 = _client(wl)
        c2.models["expert"] = ModelSpec("expert", **{"ids": ["expert"], "price_in": 2.0, "price_out": 8.0, **change})
        rp = asyncio.run(Bottler(c2, SPEC, _cfg(seed=6)).pour(art, T, votes=False))
        assert "memo" not in rp.provenance, change
    spec3 = H.TaskSpec(SPEC.name, SPEC.instruction, SPEC.labels, max_chars=300)
    rp = asyncio.run(Bottler(client, spec3, _cfg(seed=6)).pour(art, T, votes=False))
    assert "memo" not in rp.provenance
    # memo=False: v0.4 path, no memo, no extra certificate fields
    rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=6, memo=False)).pour(art, T, votes=False))
    assert "memo" not in rp.provenance and "n_memo" not in rp.certificate and "memo" not in rp.diagnostics
    # old pickle (no src, no fingerprint) -> memo off
    d = {"spec": {"name": SPEC.name, "instruction": SPEC.instruction, "labels": SPEC.labels, "max_chars": SPEC.max_chars},
         "texts": art.texts, "labels": art.labels, "C": 4.0, "max_memory": 200_000, "version": 2}
    path = tmp_path / "old.pkl"
    path.write_bytes(pickle.dumps(d))
    old = BottleArtifact.load(path)
    assert old.fingerprint is None and set(old.src) == {"bottle"}
    rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=6)).pour(old, T, votes=False))
    assert "memo" not in rp.provenance and rp.diagnostics["memo"].startswith("off: bottle has no prompt fingerprint")


class _FlakyExpert(MockClient):
    """Expert gives an unusable reply for every 4th distinct item (deterministic per text)."""

    async def chat(self, alias, messages, max_tokens=256, temperature=0.0, phase="default", seed=None):
        r = await super().chat(alias, messages, max_tokens, temperature, phase, seed)
        if alias == "expert" and int(hashlib.md5(messages[-1]["content"].encode()).hexdigest()[:6], 16) % 4 == 0:
            r.text = "I am not sure."
        return r


def test_fallback_and_none_answers_never_become_memo():
    wl = toy_workload(400, seed=24)
    client = _FlakyExpert({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.8, "expert": 0.95})
    b = Bottler(client, SPEC, _cfg(seed=7, max_expert_none=0.99))   # R concentrates the failed items
    r0 = asyncio.run(b.run(wl.texts[:300]))
    fb = [i for i, p in enumerate(r0.provenance) if p.endswith("+fallback")]
    none_items = [i for i, a in enumerate(r0.answers) if a == NONE]
    assert fb or none_items
    src = dict(zip(r0.artifact.texts, r0.artifact.src))
    assert all(src.get(wl.texts[i]) == "bottle" for i in fb)
    assert all(wl.texts[i] not in src for i in none_items)
    rp = asyncio.run(b.pour(r0.artifact, wl.texts[:300], votes=False))
    assert all(rp.provenance[i] != "memo" for i in fb + none_items)
    from bottlekit.harness import src_of
    assert src_of("expert_calibration+fallback") == "bottle" and src_of("expert_escalation+fallback") == "bottle"
    assert src_of("bottle") == "bottle" and src_of("memo") == "expert" and src_of("expert_calibration_dup") == "expert"


def test_conflicting_expert_answers_for_one_prompt_are_not_memoised():
    a = BottleArtifact(SPEC, ["x  one", "x one", "y two"], ["alpha", "beta", "gamma"], src=["expert"] * 3,
                       fingerprint="f")
    tab, n_conf = a.memo_table(SPEC)
    assert n_conf == 1 and expert_key(SPEC, "x one") not in tab and tab[expert_key(SPEC, "y two")] == "gamma"


# ================================================================================================ (d) provenance storage
def test_src_and_fingerprint_survive_dedupe_cap_update_embeddings_and_save_load(tmp_path):
    texts = ["a", "b", "c", "b", "d"]
    labels = ["alpha", "beta", "gamma", "delta_x", NONE]
    src = ["expert", "expert", "bottle", "bottle", "expert"]
    art = BottleArtifact(SPEC, texts, labels, src=src, fingerprint="fp1")
    assert art.texts == ["a", "b", "c"] and art.labels == ["alpha", "delta_x", "gamma"]
    assert art.src == ["expert", "bottle", "bottle"]                    # newest 'b' wins with ITS src; NONE dropped
    capped = BottleArtifact(SPEC, texts, labels, src=src, fingerprint="fp1", max_memory=2)
    assert capped.texts == ["b", "c"] and capped.src == ["bottle", "bottle"]
    up = art.updated(["c", "e"], ["alpha", "beta"], src=["expert", "expert"])
    assert up.fingerprint == "fp1" and up.version == 1
    assert dict(zip(up.texts, zip(up.labels, up.src))) == {"a": ("alpha", "expert"), "b": ("delta_x", "bottle"),
                                                             "c": ("alpha", "expert"), "e": ("beta", "expert")}
    up2 = up.updated(["f"], ["gamma"], src=["expert"], fingerprint="fp2")   # another prompt/model: old answers demoted
    assert up2.fingerprint == "fp2" and dict(zip(up2.texts, up2.src)) == {"a": "bottle", "b": "bottle", "c": "bottle",
                                                                           "e": "bottle", "f": "expert"}
    E = np.random.default_rng(0).normal(size=(len(up.texts), 3))
    we = up.with_embeddings(E, "e1")
    assert we.src == up.src and we.fingerprint == up.fingerprint and we.dense
    for a in (art, up, up2, we):
        p = tmp_path / "b.pkl"
        a.save(p)
        b = BottleArtifact.load(p)
        assert (b.texts, b.labels, b.src, b.fingerprint, b.version) == (a.texts, a.labels, a.src, a.fingerprint,
                                                                        a.version)
    with pytest.raises(ValueError):
        BottleArtifact(SPEC, ["a"], ["alpha"], src=["oracle"])
    with pytest.raises(ValueError):
        BottleArtifact(SPEC, ["a", "b"], ["alpha", "beta"], src=["expert"])


def test_memo_items_are_not_re_added_and_src_is_not_corrupted_by_repours():
    wl = toy_workload(500, seed=25)
    client, r0 = _run_art(wl)
    b = Bottler(client, SPEC, _cfg(seed=8))
    r1 = asyncio.run(b.pour(r0.artifact, wl.texts[150:450], votes=False))
    r2 = asyncio.run(b.pour(r1.artifact, wl.texts[150:450], votes=False))
    mem = {t: (l, s) for t, l, s in zip(r2.artifact.texts, r2.artifact.labels, r2.artifact.src)}
    exp = dict(zip(wl.texts[150:450], _expert_of(client, wl.texts[150:450])))
    for t, (l, s) in mem.items():
        if s == "expert" and t in exp:
            assert l == exp[t]                              # an expert-sourced memory label is the expert's answer
    memo2 = [i for i, p in enumerate(r2.provenance) if p == "memo"]
    assert len(memo2) >= len([p for p in r1.provenance if p == "memo"])
    assert len(r2.artifact.texts) == len(r1.artifact.texts)  # recurring texts: no new memory rows


# ================================================================================================ (e) dedupe
def test_expert_dedupe_same_answers_and_one_call_per_unique_routed_prompt():
    wl = toy_workload(500, seed=26)
    for s in range(3):
        Z = _zipf_batch(wl.texts, 300, 100, np.random.default_rng(100 + s))
        res = {}
        for dd in (False, True):
            client = _client(wl)
            b = Bottler(client, SPEC, _cfg(seed=40 + s))
            b._dedupe_expert, b._propagate = dd, False
            res[dd] = asyncio.run(b.run(Z))
        a, d = res[False], res[True]
        assert a.answers == d.answers and a.provenance == d.provenance
        routed = [i for i, p in enumerate(d.provenance) if p != "bottle"]
        assert _calls(a) == len(routed)
        assert _calls(d) == len({expert_key(SPEC, Z[i]) for i in routed}) < len(routed)
        assert d.diagnostics["expert_calls"] == _calls(d)
        assert d.certificate["n_expert_dedup"] == len(routed) - _calls(d)


def test_unusable_answers_count_for_every_routed_copy():
    wl = toy_workload(400, seed=27)
    client = _FlakyExpert({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.8, "expert": 0.95})
    Z = _zipf_batch(wl.texts, 300, 90, np.random.default_rng(5))
    b = Bottler(client, SPEC, _cfg(seed=9, max_expert_none=0.6))
    r = asyncio.run(b.run(Z))
    routed_none = sum(1 for p in r.provenance if p.startswith("expert_") and p.endswith("+fallback")) + \
        sum(1 for a, p in zip(r.answers, r.provenance) if a == NONE and p.startswith("expert_"))
    assert r.certificate["n_expert_unusable"] == routed_none > 0
    cal = [i for i, p in enumerate(r.provenance) if p.startswith("expert_calibration") and "dup" not in p]
    rate = np.mean([r.provenance[i].endswith("+fallback") or r.answers[i] == NONE for i in cal])
    assert r.diagnostics["expert_none_rate_calibration"] == pytest.approx(rate)


# ================================================================================================ (f) edge cases
def test_all_memo_pour_cold_and_warm_has_no_calls_and_no_certificate():
    wl = toy_workload(400, seed=28)
    client, r0 = _run_art(wl)
    T = [t for t, s in zip(r0.artifact.texts, r0.artifact.src) if s == "expert"][:60]
    for votes in (False, True, "auto"):
        rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=11)).pour(r0.artifact, T, votes=votes))
        assert rp.memo_share() == 1.0 and rp.ledger == {}
        c = rp.certificate
        assert c["no_calibration_needed"] and c["N_population"] == 0 and c["N"] == 60 and c["n_cal"] == 0
        assert rp.artifact.version == r0.artifact.version + 1 and rp.artifact.texts == r0.artifact.texts
        exp = _expert_of(client, T)
        assert rp.answers == exp


def test_tiny_population_and_trivial_allowance():
    wl = toy_workload(400, seed=29)
    client, r0 = _run_art(wl)
    exp_texts = [t for t, s in zip(r0.artifact.texts, r0.artifact.src) if s == "expert"][:80]
    novel = wl.texts[350:353]
    for n_new in (1, 2, 3):
        T = exp_texts + novel[:n_new]
        for rule in ("novel", "batch"):
            rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=12, memo_allowance=rule)).pour(r0.artifact, T,
                                                                                            votes=False))
            c = rp.certificate
            assert c["N_population"] == n_new and c["n_memo"] == 80
            m = math.floor(0.05 * (len(T) if rule == "batch" else n_new))
            assert c["allowance_m"] == m and c["eps_population"] == pytest.approx(m / n_new)
            if m >= n_new:
                assert c["no_calibration_needed"] and c["n_cal"] == 0 and _calls(rp, ("expert_calibration",)) == 0
            else:
                assert not c["no_calibration_needed"] and 1 <= c["n_cal"] <= n_new
            assert all(p == "memo" for p in rp.provenance[:80])


def test_calibration_permutation_is_uniform_over_R_from_the_fresh_stream():
    wl = toy_workload(500, seed=30)
    client, r0 = _run_art(wl)
    T = wl.texts[100:400]
    rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=13)).pour(r0.artifact, T, votes=False))
    R = [i for i, p in enumerate(rp.provenance) if p != "memo"]
    key = [13, 1, r0.artifact.version + 1, len(r0.artifact.texts)]
    assert rp.certificate["cal_stream"] == key
    n = rp.certificate["n_cal"]
    perm = np.random.default_rng(sub_seed(*key)).permutation(len(R))[:n]
    assert {i for i, p in enumerate(rp.provenance) if p.startswith("expert_calibration") and "dup" not in p} == \
        {R[j] for j in perm}
    # the look size is a function of (N_R, eps, delta, m) only
    N_R, m = len(R), rp.certificate["allowance_m"]
    n0 = single_look_design(N_R, max(0.05, m / N_R))[0][0]
    assert n == tune_size(n0, N_R, max(0.05, m / N_R), 0.1, "rectified", m=m)


def test_memo_pour_with_emb_and_budget_paths():
    """Dense pour: emb rows are sliced to R; budget-capped memo pour reports eps_achieved over the whole batch."""
    from test_dense_bottle import hashed_embed
    wl = toy_workload(500, seed=31)
    client = _client(wl)
    E = hashed_embed(wl.texts)
    r0 = asyncio.run(Bottler(client, SPEC, _cfg(seed=14)).run(wl.texts[:300], emb=E[:300]))
    assert r0.artifact.dense
    rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=15)).pour(r0.artifact, wl.texts[100:400], votes=False,
                                                                emb=E[100:400]))
    assert rp.certificate["n_memo"] > 0 and rp.artifact.dense and rp.diagnostics["dense"] == "dense"
    with pytest.raises(ValueError):
        asyncio.run(Bottler(client, SPEC, _cfg(seed=15)).pour(r0.artifact, wl.texts[100:400], votes=False,
                                                               emb=E[100:399]))
    rb = asyncio.run(Bottler(client, SPEC, _cfg(seed=15, eps=0.01, expert_budget=10)).pour(
        r0.artifact, wl.texts[100:400], votes=False))
    c = rb.certificate
    if c.get("budget_capped"):
        assert 0 <= c["eps_achieved"] <= c["N_population"] / c["N"] + 1e-12
        assert c["eps_certified"] >= c["eps_achieved"]


# ================================================================================================ group-aware folds
def test_group_folds_with_singletons_reproduce_v04_folds():
    wl = toy_workload(300, seed=32)
    labels = list(wl.gold)
    labels[5] = None
    for seed in (0, 3):
        a = cross_fit(wl.texts, labels, list(range(300)), seed=seed, query=labels)
        b = cross_fit(wl.texts, labels, list(range(300)), seed=seed, query=labels, groups=list(range(300)))
        assert a[0] == b[0]
        np.testing.assert_array_equal(a[1], b[1])
        np.testing.assert_array_equal(a[2], b[2])


class _SpyLR(TfidfLR):
    log: list = []

    def fit(self, texts, labels, idx):
        self._train = set(texts)
        return super().fit(texts, labels, idx)

    def predict(self, texts, idx, query=None, dense=None):
        _SpyLR.log.append((self._train, list(texts)))
        return super().predict(texts, idx, query, dense)


def test_group_folds_never_score_a_copy_with_a_twin_trained_student():
    wl = toy_workload(300, seed=33)
    Z = _zipf_batch(wl.texts, 300, 80, np.random.default_rng(1))
    gold = dict(zip(wl.texts, wl.gold))
    labels = [gold[t] for t in Z]
    _SpyLR.log = []
    cross_fit(Z, labels, list(range(len(Z))), seed=2, factory=_SpyLR, groups=Z)
    assert _SpyLR.log and all(not (set(scored) & train) for train, scored in _SpyLR.log)
    _SpyLR.log = []
    cross_fit(Z, labels, list(range(len(Z))), seed=2, factory=_SpyLR)            # v0.4 folds: twins leak
    assert any(set(scored) & train for train, scored in _SpyLR.log)


# ================================================================================================ end-to-end validity
def test_recurring_stream_validity_mock():
    """run(A) then a cold pour of a half-recurring batch, both allowances, 20 seeds: memo items never disagree with the
    (deterministic) expert, and the batch violation rate stays within delta + 3 sigma."""
    wl = toy_workload(800, seed=34)
    A = wl.texts[:400]
    B = wl.texts[200:600]
    R_seeds = 20
    viol = Counter()
    for s in range(R_seeds):
        client = _client(wl, acc={"cheap": 0.75, "expert": 0.95})
        r0 = asyncio.run(Bottler(client, SPEC, _cfg(seed=200 + s)).run(A))
        exp = _expert_of(client, B)
        for rule in ("novel", "batch"):
            rp = asyncio.run(Bottler(client, SPEC, _cfg(seed=300 + s, memo_allowance=rule, export=False)).pour(
                r0.artifact, B, votes=False))
            dis = [a != e for a, e in zip(rp.answers, exp)]
            assert not any(d for d, p in zip(dis, rp.provenance) if p == "memo")
            viol[rule] += np.mean(dis) > 0.05
    for rule in ("novel", "batch"):
        assert viol[rule] / R_seeds <= DELTA + 3 * math.sqrt(DELTA * (1 - DELTA) / R_seeds), (rule, viol)


def test_config_validation():
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", memo_allowance="all")
    assert BottleConfig(cheap="c", expert="e").memo is False                     # opt-in


# ----------------------------------------------------------------------------------------------- edge cases
def test_grouped_cross_fit_falls_back_when_one_group_dominates():
    """A batch where one prompt-group covers (almost) every usable item must still be fully scored."""
    from bottlekit.students import cross_fit
    texts = ["same prompt"] * 6
    labels = ["alpha", "beta", "alpha", "beta", "alpha", "beta"]
    pred, conf = cross_fit(texts, labels, list(range(6)), groups=texts)
    assert all(p is not None for p in pred)
    texts2 = ["dup"] * 9 + ["other one"]
    labels2 = ["alpha"] * 5 + ["beta"] * 5
    pred2, _ = cross_fit(texts2, labels2, list(range(10)), groups=texts2)
    assert all(p is not None for p in pred2)


def test_fingerprint_records_temperature_and_parser_version():
    import bottlekit.harness as hmod
    from bottlekit.tasks import PARSER_VERSION
    wl = toy_workload(50)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert"))
    fp = b.fingerprint()
    old_parser, old_temp = PARSER_VERSION, hmod.EXPERT_TEMPERATURE
    try:
        hmod.EXPERT_TEMPERATURE = 0.3
        assert b.fingerprint() != fp
    finally:
        hmod.EXPERT_TEMPERATURE = old_temp
    hmod.PARSER_VERSION = "999"
    try:
        assert b.fingerprint() != fp
    finally:
        hmod.PARSER_VERSION = old_parser
