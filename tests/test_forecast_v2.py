"""Tests for forecast v2 of pour(votes='auto'). No servers needed.

  (1) vote-expert agreement a, measured on each voted batch's uniform calibration sample, carried through cold pours;
  (2) the can-pay ceiling economics.votes_can_pay (cap = (1 - n/N)*a + e_new);
  (3) the run-time cross-fitted cold arm for the first pour;
  (4) the next-bottle cross-fitted cold arm for later pours (_resolve_xfit), from a slim block (hashes, raw calibration
      answers) recorded in every bottle.
Validity: the decision reads earlier batches only, draws only from sub_seed(seed, 7, version), never from a
calibration stream, and never changes the batch that records it.
"""

import asyncio
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
from bottlekit import BottleArtifact, BottleConfig, Bottler  # noqa: E402
from bottlekit.certify import NONE, expected_eps_closed_form, k_used  # noqa: E402
from bottlekit.economics import DEFAULT_HYSTERESIS, votes_can_pay  # noqa: E402
from bottlekit.labeling import sub_seed  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402


def _client(wl, acc=None):
    return MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc=acc or {"cheap": 0.85, "expert": 0.95})


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=7)
    base.update(kw)
    return BottleConfig(**base)


def _out(res):
    return json.dumps({"answers": res.answers, "provenance": res.provenance, "certificate": res.certificate},
                      sort_keys=True, default=str)


def _v2(why):
    """The v2 decision info, wherever it sits (top level, or under 'forecast' when the prior rule decided)."""
    return why if "forecast_v2" in why else (why.get("forecast") or {})


def _rec(why):
    """A decision record as bytes. It holds no wall time, so two resolutions compare exactly."""
    return json.dumps(why, sort_keys=True, default=str)


def _keys(o):
    """Every dict key in a nested structure."""
    if isinstance(o, dict):
        return set(o) | set().union(*[_keys(v) for v in o.values()])
    if isinstance(o, (list, tuple)):
        return set().union(*[_keys(v) for v in o]) if o else set()
    return set()


def _cal_set(res):
    return {i for i, p in enumerate(res.provenance) if p.startswith("expert_calibration") and p != "expert_calibration_dup"}


# ----------------------------------------------------------------------------------------------- (2) the ceiling
def test_votes_can_pay_unit():
    # cap = (1 - 117/400)*0.88 + 0.03 = 0.6526; cold 0.62: best warm saves 0.0326 of c_exp; votes cost 0.178 -> cold
    can, info = votes_can_pay(0.178, 1.0, 0.62, 0.88, 117, 400, 0.03)
    assert not can and abs(info["cap"] - (0.7075 * 0.88 + 0.03)) < 1e-12
    assert abs(info["cost_warm_best"] - (0.178 + 1 - info["cap"])) < 1e-12 and abs(info["cost_cold"] - 0.38) < 1e-12
    # weak bottle, cheap votes: best warm saves a lot -> votes can pay
    assert votes_can_pay(0.035, 1.0, 0.16, 0.80, 117, 400, 0.03)[0]
    # boundary: c_vote == c_exp*(cap - cov_cold) + h*c_exp -> cannot pay (ties go cold)
    cap = 0.7075 * 0.8 + 0.03
    c = 1.0 * (cap - 0.3) + DEFAULT_HYSTERESIS
    assert not votes_can_pay(c, 1.0, 0.3, 0.8, 117, 400, 0.03)[0]
    assert votes_can_pay(c - 1e-9, 1.0, 0.3, 0.8, 117, 400, 0.03)[0]
    # monotone: dearer votes never turn 'cannot' into 'can'; units cancel
    for cv in np.linspace(0, 0.5, 26):
        a = votes_can_pay(cv, 1.0, 0.4, 0.7, 117, 400, 0.03)[0]
        b = votes_can_pay(cv + 0.01, 1.0, 0.4, 0.7, 117, 400, 0.03)[0]
        assert a or not b
        assert a == votes_can_pay(cv * 1e-5, 1e-5, 0.4, 0.7, 117, 400, 0.03)[0]
    # the cap is a share: clipped to [0, 1]
    assert votes_can_pay(0.1, 1.0, 0.5, 1.0, 0, 400, 0.5)[1]["cap"] == 1.0
    assert votes_can_pay(0.1, 1.0, 0.5, 0.0, 117, 400, 0.0)[1]["cap"] == 0.0


def test_decision_ceiling_uses_measured_a_and_the_planned_design():
    wl = toy_workload(900, seed=61)
    b = Bottler(_client(wl), SPEC, _cfg(seed=3, forecast_v2="canpay"))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=True))
    nxt = wl.texts[600:]
    use, why = b._votes_pay(nxt, w.artifact, dense_next=False)
    cp = why["can_pay"]
    ag = w.artifact.forecast["agree"]
    N, n = len(nxt), b._planned_n(len(nxt))
    k = k_used(n, N, b.cfg.eps, b.cfg.delta, "rectified")
    e = expected_eps_closed_form(n, N, k)
    assert abs(cp["e_new"] - e) < 1e-15 and abs(cp["a"] - ag["a"]) < 1e-15
    assert abs(cp["cap"] - min(1.0, (1 - n / N) * ag["a"] + e)) < 1e-12
    if why["basis"] == "measured":
        assert why["cov_warm"] <= cp["cap"] + 1e-15


# ----------------------------------------------------------------------------------------------- (1) agreement
def test_agreement_is_measured_on_the_raw_calibration_answers_and_carried(monkeypatch):
    wl = toy_workload(1200, seed=62)

    class Flaky(MockClient):         # ~7% unusable expert answers: excluded from a's counts
        def _answer(self, alias, text, salt, pos=None, k=None):
            if alias == "expert" and hash(text) % 14 == 0:
                return "I am not sure."
            return super()._answer(alias, text, salt, pos, k)

    client = Flaky({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.8, "expert": 0.95})
    seen = []
    real = H.Bottler._agreement

    def spy(kind, fc, ctx, parent, key):
        out = real(kind, fc, ctx, parent, key)
        seen.append((kind, dict(ctx["cal"]), list(ctx["final"]), out))
        return out
    monkeypatch.setattr(H.Bottler, "_agreement", staticmethod(spy))
    b = Bottler(client, SPEC, _cfg(seed=5))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=True))
    c = asyncio.run(b.pour(w.artifact, wl.texts[600:900], votes=False))
    c2 = asyncio.run(b.pour(c.artifact, wl.texts[900:], votes=False))
    for kind, cal, final, out in seen[:2]:
        use = [i for i, a in cal.items() if a != NONE]
        assert len(use) < len(cal)                                      # some raw NONE answers were excluded
        assert out["n"] == len(use) and out["n_cal"] == len(cal) and out["age"] == 0 and out["basis"] == "measured"
        assert abs(out["a"] - np.mean([final[i] == cal[i] for i in use])) < 1e-15
    assert r0.artifact.forecast["agree"]["source"] == "run"
    assert w.artifact.forecast["agree"]["source"] == "pour_warm"
    g1, g2 = c.artifact.forecast["agree"], c2.artifact.forecast["agree"]
    assert g1["a"] == w.artifact.forecast["agree"]["a"] and g1["age"] == 1 and g1["basis"] == "carried"
    assert g2["a"] == g1["a"] and g2["age"] == 2


def test_student_mode_run_records_no_agreement_and_decides_as_v07():
    wl = toy_workload(900, seed=63)
    cfgs = {v: _cfg(seed=2, mode="student", n_build=200, forecast_v2=v) for v in ("off", "full")}
    b = Bottler(_client(wl), SPEC, cfgs["full"])
    r0 = asyncio.run(b.run(wl.texts[:600]))
    assert "agree" not in r0.artifact.forecast
    u_full, why = b._votes_pay(wl.texts[600:], r0.artifact, dense_next=False)
    u_off, _ = Bottler(b.client, SPEC, cfgs["off"])._votes_pay(wl.texts[600:], r0.artifact, dense_next=False)
    assert why["basis"] == "prior" and u_full == u_off


# ----------------------------------------------------------------------------------------------- slim block
def test_slim_block_holds_no_texts_and_maps_onto_memory():
    wl = toy_workload(700, seed=64)
    b = Bottler(_client(wl), SPEC, _cfg(seed=6))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=True))
    for res, n_items in ((r0, 300), (w, 300)):
        blk = res.artifact.forecast["xfit"]
        assert len(blk["hashes"]) == n_items and blk["n_cal"] == res.certificate["n_cal"] == len(blk["cal"])
        assert blk["version"] == res.artifact.version and blk["memory"] == len(res.artifact.texts)
        assert set(blk["cal"]) == _cal_set(res)
        flat = json.dumps(blk)
        assert not any(t in flat for t in wl.texts[:600])               # no texts stored
        assert all(isinstance(h, int) and 0 <= h < 2 ** 64 for h in blk["hashes"])
        assert {H._text_hash(t) for t in res.artifact.texts} >= set(blk["hashes"])


# ----------------------------------------------------------------------------------------------- validity
def test_resolver_draws_only_from_its_own_key_and_never_certifies_with_the_pipeline(monkeypatch):
    """The decision (incl. both cross-fits) uses sub_seed(seed, 7, version) only: no calibration stream (1, ...), no
    student stream (2), no context stream (11, c); and never calls the pipeline's own certify/certify_sequential."""
    wl = toy_workload(900, seed=65)
    b = Bottler(_client(wl), SPEC, _cfg(seed=21))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    c = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=False))
    keys = []
    real_sub = H.sub_seed

    def spy_sub(*a):
        keys.append(tuple(int(x) for x in a))
        return real_sub(*a)
    rngs = []
    real_rng = np.random.default_rng

    def spy_rng(seed=None):
        rngs.append(seed)
        return real_rng(seed)

    def forbidden(*a, **k):
        raise AssertionError("the decision called a pipeline certificate function")
    monkeypatch.setattr(H, "sub_seed", spy_sub)
    monkeypatch.setattr(H.np.random, "default_rng", spy_rng)
    monkeypatch.setattr(H, "certify", forbidden)
    monkeypatch.setattr(H, "certify_sequential", forbidden)
    b2 = Bottler(b.client, SPEC, _cfg(seed=99))                          # another Bottler seed decides
    u1, w1 = b2._votes_pay(wl.texts[300:600], r0.artifact, dense_next=False)
    u2, w2 = b2._votes_pay(wl.texts[600:], c.artifact, dense_next=False)
    assert "next-bottle" in _v2(w2)["arms_basis"]["cold"] and "crossfit cold" in _v2(w1)["arms_basis"]["cold"]
    assert keys == [(21, 7, 0), (21, 7, 1)]                               # the recording batch's seed and version
    assert rngs == [real_sub(21, 7, 0), real_sub(21, 7, 1)]
    assert real_sub(21, 7, 0) == real_sub(21, 7)                          # run(): the v0.6 forecast stream
    cal_keys = {real_sub(21, 1), real_sub(21, 1, 1, 300), real_sub(21, 1, 2, 600), real_sub(21, 2)}
    assert not cal_keys & set(rngs)


def test_cross_fit_never_scores_a_row_with_a_model_that_saw_its_text(monkeypatch):
    """Leakage check: duplicates inside the batch and batch texts already in the old memory map to ONE memory row,
    which is held out together; every fold's training texts are disjoint from the texts it scores."""
    wl = toy_workload(900, seed=66)
    b = Bottler(_client(wl), SPEC, _cfg(seed=4))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    batch = wl.texts[300:560] + wl.texts[300:330] + wl.texts[:40]        # in-batch duplicates + recurring old texts
    w = asyncio.run(b.pour(r0.artifact, batch, votes=True))
    pairs = []
    real_fit, real_pred = H.TfidfLR.fit, H.TfidfLR.predict

    def fit(self, texts, labels, idx):
        self._train = set(texts)
        return real_fit(self, texts, labels, idx)

    def pred(self, texts, idx, query=None, dense=None):
        pairs.append((getattr(self, "_train", set()), set(texts)))
        return real_pred(self, texts, idx, query, dense)
    monkeypatch.setattr(H.TfidfLR, "fit", fit)
    monkeypatch.setattr(H.TfidfLR, "predict", pred)
    _, why = b._votes_pay(wl.texts[600:], w.artifact, dense_next=False)
    x = _v2(why).get("xfit")
    assert isinstance(x, dict) and x["n_rows"] == len(set(batch)) and x["n_other_rows"] == len(w.artifact.texts) - len(set(batch))
    assert len(pairs) == x["folds"]
    for train, scored in pairs:
        assert scored and not (train & scored)
    assert set().union(*[s for _, s in pairs]) == set(batch)            # every batch row scored exactly once
    assert sum(len(s) for _, s in pairs) == len(set(batch))


def test_auto_equals_forced_and_calibration_sample_is_choice_free_in_every_mode():
    wl = toy_workload(1200, seed=67)
    for v in ("canpay", "pour1", "full"):
        b = Bottler(_client(wl), SPEC, _cfg(seed=8, forecast_v2=v))
        r0 = asyncio.run(b.run(wl.texts[:300]))
        a1 = asyncio.run(b.pour(r0.artifact, wl.texts[300:700], votes="auto"))
        f1 = asyncio.run(b.pour(r0.artifact, wl.texts[300:700], votes=a1.diagnostics["mode"] == "pour_warm"))
        o1 = asyncio.run(b.pour(r0.artifact, wl.texts[300:700], votes=a1.diagnostics["mode"] != "pour_warm"))
        assert _out(a1) == _out(f1)
        assert a1.certificate["cal_stream"] == o1.certificate["cal_stream"] and _cal_set(a1) == _cal_set(o1)
        a2 = asyncio.run(b.pour(a1.artifact, wl.texts[700:], votes="auto"))
        f2 = asyncio.run(b.pour(a1.artifact, wl.texts[700:], votes=a2.diagnostics["mode"] == "pour_warm"))
        assert _out(a2) == _out(f2)


def test_forced_pours_are_identical_across_modes_and_never_resolve_the_cross_fit():
    wl = toy_workload(900, seed=68)
    outs = {}
    for v in ("off", "canpay", "pour1", "full"):
        b = Bottler(_client(wl), SPEC, _cfg(seed=9, forecast_v2=v))
        r0 = asyncio.run(b.run(wl.texts[:300]))
        c = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=False))
        w = asyncio.run(b.pour(c.artifact, wl.texts[600:], votes=True))
        outs[v] = [_out(r) for r in (r0, c, w)]
        assert len(b._xfit_cache) == 0                                   # lazy: forced pours pay nothing
        for res in (r0, c, w):
            assert "resolved" not in res.artifact.forecast["xfit"]
    assert len({json.dumps(o) for o in outs.values()}) == 1


# ----------------------------------------------------------------------------------------------- mock lineage
def test_mock_lineage_where_the_ceiling_flips_warm_to_cold():
    """A lineage whose last measured warm gain is large (v0.7 says warm) but whose votes agree with the expert on few
    items: the can-pay ceiling shows that even the best warm pour cannot beat the cold arm, so it goes cold. With a
    high agreement the ceiling does not fire and the decision is v0.7's with the warm arm clipped at the cap."""
    wl = toy_workload(1200, seed=69)
    b_off = Bottler(_client(wl), SPEC, _cfg(seed=3, forecast_v2="off"))
    b_cp = Bottler(b_off.client, SPEC, _cfg(seed=3, forecast_v2="canpay"))
    r0 = asyncio.run(b_off.run(wl.texts[:300]))
    w = asyncio.run(b_off.pour(r0.artifact, wl.texts[300:600], votes=True))
    c = asyncio.run(b_off.pour(w.artifact, wl.texts[600:900], votes=False))      # cold: stale gain + carried a
    f = c.artifact.forecast
    f["gain"] = {"value": 0.5, "age": 1, "basis": "stale", "origin": "exact"}     # an optimistic stale gain
    nxt = wl.texts[900:]
    use_off, why_off = b_off._votes_pay(nxt, c.artifact, dense_next=False)
    assert use_off and why_off["basis"] == "measured"                            # v0.7: warm
    cold = why_off["cov_cold"]
    N, n = len(nxt), b_cp._planned_n(len(nxt))
    lab = 1 - n / N
    # agreement so low that cap - cov_cold < c_vote/c_exp: the votes cannot pay
    f["agree"] = {"a": max(0.0, (cold - 0.01) / lab), "n": 120, "n_cal": 120, "age": 1, "basis": "carried"}
    use_cp, why_cp = b_cp._votes_pay(nxt, c.artifact, dense_next=False)
    assert not use_cp and why_cp["basis"] == "can_pay_bound" and why_cp["can_pay"]["can_pay"] is False
    assert why_cp["cost_warm_best"] >= why_cp["cost_cold"]
    auto = asyncio.run(b_cp.pour(c.artifact, nxt, votes="auto"))
    forced = asyncio.run(b_cp.pour(c.artifact, nxt, votes=False))
    assert auto.diagnostics["mode"] == "pour" and _out(auto) == _out(forced)
    # high agreement: no flip; the warm arm is clipped at the cap and v0.7's rule decides
    f["agree"] = {"a": 1.0, "n": 120, "n_cal": 120, "age": 1, "basis": "carried"}
    use_hi, why_hi = b_cp._votes_pay(nxt, c.artifact, dense_next=False)
    assert why_hi["basis"] == "measured" and why_hi["cov_warm"] <= why_hi["can_pay"]["cap"] + 1e-15
    assert why_hi["warm_clipped_at_cap"] == (why_off["cov_warm"] > why_hi["can_pay"]["cap"])


# ----------------------------------------------------------------------------------------------- fallbacks
def test_old_pickles_without_the_v2_fields_fall_back_to_v07(tmp_path):
    wl = toy_workload(1000, seed=70)
    b_off = Bottler(_client(wl), SPEC, _cfg(seed=4, forecast_v2="off"))
    b_full = Bottler(b_off.client, SPEC, _cfg(seed=4))                            # default: full
    assert b_full.cfg.forecast_v2 == "full"
    r0 = asyncio.run(b_off.run(wl.texts[:300]))
    w = asyncio.run(b_off.pour(r0.artifact, wl.texts[300:600], votes=True))
    for art, nxt in ((r0.artifact, wl.texts[300:600]), (w.artifact, wl.texts[600:])):
        p = tmp_path / f"v{art.version}.pkl"
        art.save(p)
        d = pickle.loads(p.read_bytes())
        for k in ("agree", "xfit"):
            d["forecast"].pop(k)                                                  # a v0.7 bottle
        (tmp_path / "old.pkl").write_bytes(pickle.dumps(d))
        old = BottleArtifact.load(tmp_path / "old.pkl")
        u_f, w_f = b_full._votes_pay(nxt, old, dense_next=False)
        u_o, w_o = b_off._votes_pay(nxt, old, dense_next=False)
        assert u_f == u_o and w_f["basis"] == w_o["basis"]
        if w_o["basis"] == "measured":
            assert w_f["cov_cold"] == w_o["cov_cold"] and w_f["cov_warm"] == w_o["cov_warm"]
        # the reloaded v0.8 bottle decides exactly like the in-memory one
        back = BottleArtifact.load(p)
        (u1, w1), (u2, w2) = (b_full._votes_pay(nxt, back, dense_next=False),
                              b_full._votes_pay(nxt, art, dense_next=False))
        assert u1 == u2 and _rec(w1) == _rec(w2)


def test_evicted_or_colliding_rows_fall_back_instead_of_misassigning(monkeypatch):
    wl = toy_workload(1300, seed=71)
    b = Bottler(_client(wl), SPEC, _cfg(seed=5))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    small = BottleArtifact(SPEC, r0.artifact.texts, r0.artifact.labels, max_memory=500, version=r0.artifact.version,
                           src=r0.artifact.src, fingerprint=r0.artifact.fingerprint, forecast=r0.artifact.forecast)
    big = asyncio.run(b.pour(small, wl.texts[300:900], votes=False))             # 600 > max_memory: rows evicted
    assert len(big.artifact.texts) == 500
    use, why = b._votes_pay(wl.texts[900:], big.artifact, dense_next=False)
    fc = _v2(why)
    assert "not in the bottle's memory" in fc["xfit_error"]
    assert fc["arms_basis"]["cold"] == "realized"                               # the v0.7 (lagged) arm
    # a hash collision between a memory text and a batch text: fall back, never mis-assign
    c = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=False))
    blk = c.artifact.forecast["xfit"]
    victim = wl.texts[0]                                                          # pre-batch memory row
    target = blk["hashes"][0]
    real = H._text_hash
    monkeypatch.setattr(H, "_text_hash", lambda t: target if t == victim else real(t))
    use, why = b._votes_pay(wl.texts[600:900], c.artifact, dense_next=False)
    fc = _v2(why)
    assert "collision" in fc["xfit_error"]
    assert "resolved" not in blk and c.artifact not in b._xfit_cache          # nothing cached for a failed mapping


def test_fold_budget():
    """'auto': the largest K in [2, 5] with training rows K*M - R STRICTLY below 2 bottle fits (with <=,
    a run() bottle, R = M, would take K = 3, measured 2.15-2.45x one fit). Since R <= M this is K = 2."""
    b = Bottler(MockClient({}), SPEC, _cfg())
    assert b._xfit_folds(400, 400) == 2          # run(): 2*400 - 400 = 1 bottle fit (3 folds would be exactly 2)
    assert b._xfit_folds(400, 800) == 2          # pour-1 bottle: 2*800 - 400 = 1.5 fits
    assert b._xfit_folds(400, 5000) == 2
    assert b._xfit_folds(3000, 3000) == 2 and b._xfit_folds(3, 3) == 2
    for R, M in ((400, 400), (400, 800), (400, 1200), (400, 5000), (1200, 1600), (1, 1), (7, 7), (5000, 5000)):
        K = b._xfit_folds(R, M)
        assert K * M - R < 2 * M or K == 2
        assert (K + 1) * M - R >= 2 * M or K == 5
    b5 = Bottler(MockClient({}), SPEC, _cfg(forecast_folds=5))
    assert b5._xfit_folds(400, 5000) == 5 and b5._xfit_folds(3, 3) == 3
    with pytest.raises(ValueError):
        _cfg(forecast_folds=1)
    with pytest.raises(ValueError):
        _cfg(forecast_v2="bogus")


def test_run_cross_fit_with_five_folds_reproduces_the_v06_run_forecast():
    """Part (3) with forecast_folds=5 is exactly the cold arm of the v0.6 run-time cross-fit (same stream, folds,
    student and certificate), computed from the slim block instead of stored texts."""
    wl = toy_workload(700, seed=72)
    b = Bottler(_client(wl), SPEC, _cfg(seed=6, forecast_run=True, forecast_v2="off"))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    ref = b._resolve_pending(r0.artifact, r0.artifact.forecast["pending"])
    b5 = Bottler(b.client, SPEC, _cfg(seed=6, forecast_folds=5))
    got = b5._resolve_xfit(r0.artifact, r0.artifact.forecast["xfit"])
    assert got["folds"] == 5 and got["n_other_rows"] == 0
    assert got["cold"] == ref["cold"] and got["tau"] == ref["cold_tau"]


# ----------------------------------------------------------------------------------------------- edge cases
def test_a_decision_leaves_the_bottle_and_its_pickle_unchanged(tmp_path):
    """The cross-fit cache lives in the Bottler, never in the artifact: deciding (and pouring with votes='auto') leaves
    the input bottle's pickle bytes, its save() bytes and its forecast unchanged. A second Bottler deciding on the same
    bottle shares nothing with the first, and the cached (second) decision equals the first."""
    wl = toy_workload(1300, seed=73)
    b = Bottler(_client(wl), SPEC, _cfg(seed=11))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    c = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=False))
    w = asyncio.run(b.pour(c.artifact, wl.texts[600:900], votes=True))
    for art, nxt in ((r0.artifact, wl.texts[300:600]), (c.artifact, wl.texts[600:900]), (w.artifact, wl.texts[900:])):
        before, f_before = pickle.dumps(art), _rec(art.forecast)
        art.save(tmp_path / "before.pkl")
        d = Bottler(b.client, SPEC, _cfg(seed=11))
        u1, w1 = d._votes_pay(nxt, art, dense_next=False)
        assert isinstance(_v2(w1).get("xfit"), dict)                       # the cross-fit really ran
        assert art in d._xfit_cache and art not in b._xfit_cache          # per-Bottler cache, outside the bottle
        u2, w2 = d._votes_pay(nxt, art, dense_next=False)                 # cached
        assert u1 == u2 and _rec(w1) == _rec(w2)
        res = asyncio.run(d.pour(art, nxt, votes="auto"))
        assert res.diagnostics["mode"] == ("pour_warm" if u1 else "pour")
        art.save(tmp_path / "after.pkl")
        assert pickle.dumps(art) == before
        assert (tmp_path / "after.pkl").read_bytes() == (tmp_path / "before.pkl").read_bytes()
        assert _rec(art.forecast) == f_before and "resolved" not in art.forecast["xfit"]


def test_auto_pour_decision_record_is_reproducible_and_wall_time_is_separate():
    """diagnostics['votes_decision'] holds no wall time: an uncached decision, a cached one and a fresh Bottler's give
    byte-identical records. The decision's wall time is reported separately (diagnostics['wall_time_s']); forced pours
    have none."""
    wl = toy_workload(1000, seed=74)
    b = Bottler(_client(wl), SPEC, _cfg(seed=12))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    c = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=False))
    assert "wall_time_s" not in c.diagnostics
    for art, nxt in ((r0.artifact, wl.texts[300:600]), (c.artifact, wl.texts[600:])):
        recs = []
        for bot in (b, b, Bottler(b.client, SPEC, _cfg(seed=12))):        # uncached, cached, fresh Bottler
            res = asyncio.run(bot.pour(art, nxt, votes="auto"))
            rec = res.diagnostics["votes_decision"]
            assert isinstance(_v2(rec).get("xfit"), dict)
            assert not ({"seconds", "cached", "wall_time_s"} & _keys(rec))
            t = res.diagnostics["wall_time_s"]
            assert set(t) == {"votes_decision"} and t["votes_decision"] >= 0.0
            recs.append(_rec(rec))
        assert len(set(recs)) == 1
