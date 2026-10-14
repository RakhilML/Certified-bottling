"""Tests for the fresh per-batch calibration stream (v0.4 fix, independent of the calibration design).

v0.3 drew every batch's calibration permutation from sub_seed(cfg.seed, 1): a pour of the SAME items (or a batch that
is position-aligned with an earlier one) calibrated on exactly the positions whose expert answers the bottle had
memorized, so the calibration sample was no longer independent of the scores (replay: emotion half-overlap re-pour,
violation 10/20 vs 0/20 with a fresh stream). Now:
  * run() keeps sub_seed(cfg.seed, 1) (bit-identical calibration draws);
  * pour() uses sub_seed(cfg.seed, 1, artifact.version + 1, len(artifact.texts)), distinct from every ancestor's key;
  * the one s_cal drives the sequential order, the fixed draw and the budget inversion; certificate['cal_stream'].
"""

import asyncio
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
from bottlekit import BottleArtifact, BottleConfig, Bottler  # noqa: E402
from bottlekit.certify import tune_size  # noqa: E402
from bottlekit.labeling import sub_seed  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402


def _client(wl, **kw):
    return MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc=kw.get("acc", {"cheap": 0.8, "expert": 0.95}))


def _cal_set(res):
    return {i for i, p in enumerate(res.provenance) if p.startswith("expert_calibration")}


class _Spy:
    """Record the order seed given to certify_sequential and sequential_certified_eps."""

    def __init__(self):
        self.seq, self.eps = [], []

    def __enter__(self):
        self.o_seq, self.o_eps = H.certify_sequential, H.sequential_certified_eps

        async def seq(*a, **kw):
            self.seq.append(kw.get("seed"))
            return await self.o_seq(*a, **kw)

        def eps(score, final, labels, order_seed, *a, **kw):
            self.eps.append(order_seed)
            return self.o_eps(score, final, labels, order_seed, *a, **kw)

        H.certify_sequential, H.sequential_certified_eps = seq, eps
        return self

    def __exit__(self, *exc):
        H.certify_sequential, H.sequential_certified_eps = self.o_seq, self.o_eps


def test_run_calibration_stream_is_unchanged():
    wl = toy_workload(400, seed=3)
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=11)
    with _Spy() as spy:
        res = asyncio.run(Bottler(_client(wl), SPEC, cfg).run(wl.texts))
    assert spy.seq == [sub_seed(11, 1)]
    assert res.certificate["cal_stream"] == [11, 1]
    order = np.random.default_rng(sub_seed(11, 1)).permutation(400)
    assert _cal_set(res) == set(order[:res.certificate["n_cal"]].tolist())


def test_repour_of_the_same_items_draws_a_fresh_calibration_sample():
    """run() then a cold pour of the SAME items in the SAME order with the same seed: v0.3 calibrated on the identical
    positions (every calibration item a memorized expert answer: a vacuous certificate); now the sets differ and
    overlap only by chance (~n^2/N)."""
    wl = toy_workload(400, seed=4)
    b = Bottler(_client(wl), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=5))
    r0 = asyncio.run(b.run(wl.texts))
    rp = asyncio.run(b.pour(r0.artifact, wl.texts, votes=False))
    s_run, s_pour = _cal_set(r0), _cal_set(rp)
    n_run, n_pour = len(s_run), len(s_pour)
    v03 = np.random.default_rng(sub_seed(5, 1)).permutation(400)              # v0.3: one stream for both batches
    assert s_run == set(v03[:n_run].tolist())
    v03_pour = set(v03[:n_pour].tolist())
    assert v03_pour <= s_run or s_run <= v03_pour                               # v0.3: nested (identical if n equal)
    key = [5, 1, r0.artifact.version + 1, len(r0.artifact.texts)]
    assert rp.certificate["cal_stream"] == key
    assert s_pour == set(np.random.default_rng(sub_seed(*key)).permutation(400)[:n_pour].tolist())
    assert s_pour != v03_pour
    overlap = len(s_run & s_pour)
    assert overlap < 0.5 * min(n_run, n_pour) + 10, (overlap, n_run, n_pour)   # chance level ~ n_run*n_pour/N


def test_lineage_keys_are_distinct_from_ancestors_and_other_streams():
    """run -> v0 -> pour -> v1 -> pour -> v2: every calibration stream differs from all ancestors' and from the other
    sub_seed streams (2: cross-fit, 11: voting contexts)."""
    wl = toy_workload(600, seed=6)
    b = Bottler(_client(wl), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=7))
    with _Spy() as spy:
        r0 = asyncio.run(b.run(wl.texts[:200]))
        r1 = asyncio.run(b.pour(r0.artifact, wl.texts[200:400], votes=False))
        r2 = asyncio.run(b.pour(r1.artifact, wl.texts[400:600], votes=True))
    keys = [r.certificate["cal_stream"] for r in (r0, r1, r2)]
    assert keys[0] == [7, 1] and keys[1][:3] == [7, 1, 1] and keys[2][:3] == [7, 1, 2]
    seeds = [sub_seed(*k) for k in keys]
    assert spy.seq == seeds and len(set(seeds)) == 3
    others = {sub_seed(7, 2)} | {sub_seed(7, 11, c) for c in range(3)} | {7}
    assert not set(seeds) & others
    perms = [tuple(np.random.default_rng(s).permutation(200)[:50]) for s in seeds]
    assert len(set(perms)) == 3


def test_budget_inversion_and_fixed_draw_use_the_pour_stream():
    wl = toy_workload(400, seed=8)
    b0 = Bottler(_client(wl), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=9))
    art = asyncio.run(b0.run(wl.texts[:200])).artifact
    key = (9, 1, art.version + 1, len(art.texts))
    # sequential + budget: certify_sequential and sequential_certified_eps get the same pour stream
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=9, eps=0.01, expert_budget=20)
    with _Spy() as spy:
        res = asyncio.run(Bottler(_client(wl), SPEC, cfg).pour(art, wl.texts[200:400], votes=False))
    assert res.certificate.get("budget_capped")
    assert spy.seq == spy.eps == [sub_seed(*key)]
    # fixed calibration: the uniform draw comes from the same stream
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=9, calibration="fixed", n_cal=80)
    res = asyncio.run(Bottler(_client(wl), SPEC, cfg).pour(art, wl.texts[200:400], votes=False))
    n_cal = tune_size(80, 200, cfg.eps, cfg.delta, cfg.certificate)
    want = set(np.random.default_rng(sub_seed(*key)).choice(200, size=n_cal, replace=False).tolist())
    assert _cal_set(res) == want and res.certificate["cal_stream"] == list(key)


def test_old_pickles_and_hand_built_artifacts_pour(tmp_path):
    """A v0.3 pickle (version present, no new fields) loads and pours; a hand-built version-0 bottle gets the key
    (seed, 1, 1, memory size), distinct from run()'s (seed, 1)."""
    wl = toy_workload(300, seed=10)
    d = {"spec": {"name": SPEC.name, "instruction": SPEC.instruction, "labels": SPEC.labels, "max_chars": SPEC.max_chars},
         "texts": wl.texts[:150], "labels": wl.gold[:150], "C": 4.0, "max_memory": 200_000, "version": 3}
    path = tmp_path / "old.pkl"
    path.write_bytes(pickle.dumps(d))
    art = BottleArtifact.load(path)
    b = Bottler(_client(wl), SPEC, BottleConfig(cheap="cheap", expert="expert", seed=12))
    res = asyncio.run(b.pour(art, wl.texts[150:], votes=False))
    assert res.certificate["cal_stream"] == [12, 1, 4, 150] and res.artifact.version == 4
    hand = BottleArtifact(SPEC, wl.texts[:150], wl.gold[:150])
    res2 = asyncio.run(b.pour(hand, wl.texts[150:], votes=False))
    assert res2.certificate["cal_stream"] == [12, 1, 1, 150]
    assert sub_seed(12, 1, 1, 150) != sub_seed(12, 1)


def test_aligned_half_overlap_repour_validity_mock():
    """Small MockClient version of the replay check (sims/repour_mix_check.py has >= 20 seeds x 3 tasks on real
    replay data): run on A, then a cold pour of a batch whose positions hold A's items (half) or new items (half),
    position-aligned. With the fresh stream the output violation rate stays within delta + 3 sigma."""
    wl = toy_workload(800, seed=13)
    A, NEW = wl.texts[:400], wl.texts[400:]
    mix = np.random.default_rng(99).random(400) < 0.5
    B = [NEW[i] if mix[i] else A[i] for i in range(400)]
    R, viol = 24, 0
    for s in range(R):
        client = _client(wl, acc={"cheap": 0.75, "expert": 0.95})
        b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=100 + s))
        r0 = asyncio.run(b.run(A))
        rp = asyncio.run(b.pour(r0.artifact, B, votes=False))
        expert = [client._answer("expert", t, "single") for t in B]
        dis = np.mean([a != e for a, e in zip(rp.answers, expert)])
        viol += dis > 0.05
    assert viol / R <= 0.1 + 3 * np.sqrt(0.09 / R)
