"""Tests for the measured warm/cold forecast. No servers needed.

After each certified batch the exported bottle carries `forecast`, a plain dict used ONLY by pour(votes='auto'):
  * warm pour: the cold arm is the EXACT counterfactual (bottle alone, same calibration sample, raw expert answers);
  * cold pour: the realized cold coverage + the carried (decayed, 'stale') warm gain;
  * run(): a lazily resolved cross-fitted certified-output forecast (seed sub_seed(seed, 7)).
The forecast never changes the batch that produces it, and the calibration stream never depends on the choice.
"""

import asyncio
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
from bottlekit import BottleArtifact, BottleConfig, Bottler  # noqa: E402
from bottlekit.economics import measured_votes_pay, rescale_bottled  # noqa: E402
from bottlekit.labeling import sub_seed  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402


def _client(wl, acc=None):
    return MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc=acc or {"cheap": 0.85, "expert": 0.95})


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=7)
    base.update(kw)
    return BottleConfig(**base)


def _out(res):
    """Everything a caller reads from a result except diagnostics/ledger (byte-level comparison)."""
    return json.dumps({"answers": res.answers, "provenance": res.provenance, "certificate": res.certificate},
                      sort_keys=True, default=str)


def _cal_set(res):
    return {i for i, p in enumerate(res.provenance) if p.startswith("expert_calibration") and p != "expert_calibration_dup"}


def _emb(texts, d=8):
    out = []
    for t in texts:
        r = np.random.default_rng(abs(hash(t)) % (2 ** 32))
        v = r.normal(size=d)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, np.float32)


# ----------------------------------------------------------------------------------------------- exactness
CONFIGS = {
    "default": {},
    "fixed": {"calibration": "fixed", "n_cal": 120},
    "budget": {"expert_budget": 60},
    "hypergeometric": {"certificate": "hypergeometric"},
    "mean_cap": {"max_expected_eps": 0.03},
    "staged": {"design": "staged"},
    "fallback_off": {"fallback": False},
}


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("seed", [1, 2])
def test_warm_pour_cold_counterfactual_equals_the_actual_cold_pour(name, seed):
    wl = toy_workload(900, seed=20 + seed)
    client = _client(wl, acc={"cheap": 0.8, "expert": 0.93})
    b = Bottler(client, SPEC, _cfg(seed=seed, **CONFIGS[name]))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    cold = asyncio.run(b.pour(r0.artifact, wl.texts[300:900], votes=False))
    warm = asyncio.run(b.pour(r0.artifact, wl.texts[300:900], votes=True))
    assert "forecast_error" not in warm.diagnostics
    arm = warm.artifact.forecast["arms"]["cold"]
    if arm["basis"] == "unavailable":             # staged: that arm would stop at a look this batch did not label
        assert name == "staged"
        return
    assert arm["basis"] == "exact"
    assert arm["tau"] == cold.certificate["tau"]
    assert arm["bottled_count"] == sum(p == "bottle" for p in cold.provenance)
    assert arm["bottled"] == cold.coverage()
    assert arm["n_cal"] == cold.certificate["n_cal"]
    assert arm["budget_capped"] == bool(cold.certificate.get("budget_capped", False))
    w = warm.artifact.forecast["arms"]["warm"]
    assert w["bottled"] == warm.coverage() and w["basis"] == "realized"
    g = warm.artifact.forecast["gain"]
    assert g["age"] == 0 and abs(g["value"] - (warm.coverage() - cold.coverage())) < 1e-12


def test_counterfactual_exact_with_dense_bottle_and_unusable_expert_answers():
    wl = toy_workload(800, seed=31)
    E = _emb(wl.texts)

    class Flaky(MockClient):         # the expert answers "I am not sure." on ~7% of items (raw NONE, +fallback)
        def _answer(self, alias, text, salt, pos=None, k=None):
            if alias == "expert" and hash(text) % 14 == 0:
                return "I am not sure."
            return super()._answer(alias, text, salt, pos, k)

    client = Flaky({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.8, "expert": 0.95})
    b = Bottler(client, SPEC, _cfg(seed=3))
    r0 = asyncio.run(b.run(wl.texts[:300], emb=E[:300]))
    cold = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes=False, emb=E[300:]))
    warm = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes=True, emb=E[300:]))
    assert cold.certificate["n_expert_unusable"] > 0          # the raw NONE answers matter here
    arm = warm.artifact.forecast["arms"]["cold"]
    assert warm.artifact.forecast["dense"] is True
    assert arm["tau"] == cold.certificate["tau"]
    assert arm["bottled_count"] == sum(p == "bottle" for p in cold.provenance)


def test_counterfactual_exact_on_a_memo_restricted_pour():
    """memo=True with recurring items: the certified population is R with allowance m; the counterfactual uses the
    same R, the same m (keyword-only) and the same look sizes."""
    wl = toy_workload(900, seed=41)
    client = _client(wl, acc={"cheap": 0.8, "expert": 0.95})
    b = Bottler(client, SPEC, _cfg(seed=4, memo=True))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    batch = wl.texts[:150] + wl.texts[300:750]                     # a third recurs (expert-answered ones are memo)
    cold = asyncio.run(b.pour(r0.artifact, batch, votes=False))
    warm = asyncio.run(b.pour(r0.artifact, batch, votes=True))
    assert cold.certificate.get("n_memo", 0) > 0 and cold.certificate["allowance_m"] is not None
    f = warm.artifact.forecast
    arm = f["arms"]["cold"]
    assert f["N"] == cold.certificate["N_population"] and f["N_batch"] == len(batch)
    assert arm["tau"] == cold.certificate["tau"]
    assert arm["bottled_count"] == sum(p == "bottle" for p in cold.provenance)


def test_warm_and_cold_pours_draw_the_same_calibration_sample():
    """Validity watchout (b): the calibration key (seed, 1, version+1, |memory|) does not depend on the choice."""
    wl = toy_workload(700, seed=5)
    b = Bottler(_client(wl), SPEC, _cfg(seed=9))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    cold = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes=False))
    warm = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes=True))
    auto = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes="auto"))
    assert cold.certificate["cal_stream"] == warm.certificate["cal_stream"] == auto.certificate["cal_stream"]
    assert _cal_set(cold) == _cal_set(warm) == _cal_set(auto)


def test_auto_pour_is_exactly_the_forced_pour_it_chose():
    wl = toy_workload(700, seed=6)
    b = Bottler(_client(wl), SPEC, _cfg(seed=2))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    auto = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes="auto"))
    forced = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes=auto.diagnostics["mode"] == "pour_warm"))
    assert _out(auto) == _out(forced)


# ----------------------------------------------------------------------------------------------- report-only
def _chain(forecast: bool, seed=3, n=900, forecast_run=False):
    wl = toy_workload(n, seed=50 + seed)
    b = Bottler(_client(wl), SPEC, _cfg(seed=seed, forecast=forecast, forecast_run=forecast_run))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    c = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=False))
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=True))
    c2 = asyncio.run(b.pour(w.artifact, wl.texts[600:], votes=False))
    w2 = asyncio.run(b.pour(c.artifact, wl.texts[600:], votes=True))
    return [r0, c, w, c2, w2]


@pytest.mark.parametrize("forecast_run", [False, True])
@pytest.mark.parametrize("seed", [3, 4])
def test_outputs_byte_identical_with_forecast_on_and_off(seed, forecast_run):
    on, off = _chain(True, seed, forecast_run=forecast_run), _chain(False, seed)
    for a, b in zip(on, off):
        assert _out(a) == _out(b)
        assert a.artifact.texts == b.artifact.texts and a.artifact.labels == b.artifact.labels
        assert a.artifact.src == b.artifact.src and a.artifact.version == b.artifact.version
        assert b.artifact.forecast is None and a.artifact.forecast is not None
        assert a.ledger == b.ledger


def test_forecast_failure_never_breaks_or_changes_a_run(monkeypatch):
    ref = _chain(True, 5)

    async def boom(self, *a, **kw):
        raise RuntimeError("forecast exploded")
    monkeypatch.setattr(H.Bottler, "_forecast", boom)
    got = _chain(True, 5)
    for a, b in zip(ref, got):
        assert _out(a) == _out(b)
        assert "forecast exploded" in b.diagnostics["forecast_error"]


def test_decision_error_falls_back_to_the_prior_rule(monkeypatch):
    wl = toy_workload(600, seed=8)
    b = Bottler(_client(wl), SPEC, _cfg(seed=1))
    r0 = asyncio.run(b.run(wl.texts[:300]))

    def boom(self, *a, **kw):
        raise ValueError("bad forecast")
    monkeypatch.setattr(H.Bottler, "_measured_votes", boom)
    res = asyncio.run(b.pour(r0.artifact, wl.texts[300:], votes="auto"))
    why = res.diagnostics["votes_decision"]
    assert why["basis"] == "prior" and "bad forecast" in why["forecast_error"]


# ----------------------------------------------------------------------------------------------- persistence
def test_save_load_round_trip_old_pickle_and_carry(tmp_path):
    wl = toy_workload(700, seed=12)
    b = Bottler(_client(wl), SPEC, _cfg(seed=4, forecast_run=True))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:500], votes=True))
    for art in (r0.artifact, w.artifact):
        p = tmp_path / f"b{art.version}.pkl"
        art.save(p)
        back = BottleArtifact.load(p)
        assert back.forecast == H._plain(art.forecast)
        assert json.dumps(back.forecast, sort_keys=True)            # plain types only
    # a reloaded pending forecast decides exactly like the original
    p0 = tmp_path / "b0.pkl"
    d1 = asyncio.run(b.pour(BottleArtifact.load(p0), wl.texts[500:], votes="auto")).diagnostics["votes_decision"]
    d2 = asyncio.run(b.pour(r0.artifact, wl.texts[500:], votes="auto")).diagnostics["votes_decision"]
    assert d1 == d2 and d1["basis"] == "measured"
    # old pickle (no 'forecast' key) loads with forecast=None and auto uses the prior rule
    d = pickle.loads(p0.read_bytes())
    d.pop("forecast")
    (tmp_path / "old.pkl").write_bytes(pickle.dumps(d))
    old = BottleArtifact.load(tmp_path / "old.pkl")
    assert old.forecast is None
    res = asyncio.run(b.pour(old, wl.texts[500:], votes="auto"))
    assert res.diagnostics["votes_decision"]["basis"] == "prior"
    # carried by updated() and with_embeddings(); never read by predict()
    up = w.artifact.updated(["new text"], ["alpha"])
    assert up.forecast == w.artifact.forecast
    assert w.artifact.with_embeddings(None).forecast == w.artifact.forecast
    bare = BottleArtifact(SPEC, w.artifact.texts, w.artifact.labels)
    (p1, c1), (p2, c2) = bare.predict(wl.texts[:50]), w.artifact.predict(wl.texts[:50])
    assert p1 == p2 and np.array_equal(c1, c2)


def test_run_forecast_is_lazy_and_resolved_only_by_auto():
    wl = toy_workload(700, seed=13)
    b = Bottler(_client(wl), SPEC, _cfg(seed=6, forecast_run=True))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    f = r0.artifact.forecast
    assert f["source"] == "run" and f["status"] == "pending" and "texts" in f["pending"]
    asyncio.run(b.pour(r0.artifact, wl.texts[300:500], votes=False))
    asyncio.run(b.pour(r0.artifact, wl.texts[300:500], votes=True))
    assert f["status"] == "pending" and "resolved" not in f["pending"]   # forced pours never pay for the cross-fit
    res = asyncio.run(b.pour(r0.artifact, wl.texts[300:500], votes="auto"))
    assert f["status"] == "resolved" and "texts" not in f["pending"]
    r = f["pending"]["resolved"]
    assert 0 <= r["cold"] <= 1 and 0 <= r["warm"] <= 1
    assert res.diagnostics["votes_decision"]["arms_basis"] == {"cold": "crossfit", "warm": "crossfit"}


def test_resolution_uses_its_own_stream(monkeypatch):
    """Validity watchout (a): the run-time cross-fit uses sub_seed(seed, 7), never the student (2) or calibration
    (1, ...) streams, and the seed is the RUN's seed (stored in the forecast), whoever resolves it."""
    wl = toy_workload(600, seed=14)
    r0 = asyncio.run(Bottler(_client(wl), SPEC, _cfg(seed=21, forecast_run=True)).run(wl.texts[:300]))
    seeds = []
    real = H.cross_fit

    def spy(*a, **kw):
        seeds.append(kw.get("seed"))
        return real(*a, **kw)
    monkeypatch.setattr(H, "cross_fit", spy)
    b2 = Bottler(_client(wl), SPEC, _cfg(seed=99, forecast_run=True))   # another Bottler seed resolves it
    b2._votes_pay(wl.texts[300:], r0.artifact, dense_next=False)
    assert seeds == [sub_seed(21, 7)]
    assert sub_seed(21, 7) not in (sub_seed(21, 2), sub_seed(21, 1), sub_seed(21, 1, 1, 300))


def test_student_mode_run_has_no_warm_arm_and_uses_the_prior_rule():
    wl = toy_workload(900, seed=15)
    b = Bottler(_client(wl), SPEC, _cfg(seed=2, mode="student", n_build=200, forecast_run=True))
    r0 = asyncio.run(b.run(wl.texts[:600]))
    res = asyncio.run(b.pour(r0.artifact, wl.texts[600:], votes="auto"))
    why = res.diagnostics["votes_decision"]
    assert why["basis"] == "prior" and why["forecast"]["warm"] is None


def test_default_policy_without_run_time_forecast():
    """Default (forecast_run=False, the kill-criterion fallback): the first pour from a run() bottle and a pour after a
    cold first pour use the gain-prior rule; a pour after a WARM pour uses the measured (exact) forecast. A pending
    run-time forecast recorded with forecast_run=True is neither used nor resolved by a default Bottler.
    (v0.8: this is the v0.7 policy, kept as forecast_v2='off'; the v2 default is tested in test_forecast_v2.py)"""
    wl = toy_workload(1000, seed=18)
    b = Bottler(_client(wl), SPEC, _cfg(seed=5, forecast_v2="off"))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    f0 = r0.artifact.forecast
    assert f0["source"] == "run" and f0["status"] == "unavailable" and "pending" not in f0   # no texts duplicated
    assert f0["costs"]["vote_basis"] == "measured" and f0["costs"]["expert_basis"] == "measured"
    a1 = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes="auto"))
    why = a1.diagnostics["votes_decision"]
    assert why["basis"] == "prior" and "forecast_run" in why["forecast"]["basis_cold"]
    c = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=False))
    assert c.artifact.forecast["gain"] is None
    assert b._votes_pay(wl.texts[600:], c.artifact, dense_next=False)[1]["basis"] == "prior"
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=True))
    use, why = b._votes_pay(wl.texts[600:], w.artifact, dense_next=False)
    assert why["basis"] == "measured" and why["arms_basis"] == {"cold": "exact", "warm": "realized"}
    # a bottle carrying a pending run-time forecast (made with forecast_run=True) and its cold child
    rx = asyncio.run(Bottler(_client(wl), SPEC, _cfg(seed=5, forecast_run=True)).run(wl.texts[:300]))
    assert rx.artifact.forecast["status"] == "pending"
    assert b._votes_pay(wl.texts[300:600], rx.artifact, dense_next=False)[1]["basis"] == "prior"
    cx = asyncio.run(b.pour(rx.artifact, wl.texts[300:600], votes=False))
    assert cx.artifact.forecast["gain"]["pending"] is not None
    why = b._votes_pay(wl.texts[600:], cx.artifact, dense_next=False)[1]
    assert why["basis"] == "prior" and "forecast_run" in why["forecast"]["basis_warm"]
    assert rx.artifact.forecast["status"] == "pending" and "texts" in rx.artifact.forecast["pending"]   # untouched
    # the same bottles decide 'measured' (cross-fit) for a Bottler that opts in
    bx = Bottler(b.client, SPEC, _cfg(seed=5, forecast_run=True, forecast_v2="off"))
    assert bx._votes_pay(wl.texts[600:], cx.artifact, dense_next=False)[1]["basis"] == "measured"


# ----------------------------------------------------------------------------------------------- decision
def test_measured_votes_pay_units():
    use, info = measured_votes_pay(0.40, 0.60, 0.10, 1.0)          # warm 0.1+0.4=0.5 < cold 0.6-0.005
    assert use and abs(info["cost_cold"] - 0.6) < 1e-12 and abs(info["cost_warm"] - 0.5) < 1e-12
    assert not measured_votes_pay(0.40, 0.50, 0.10, 1.0)[0]        # tie -> cold (hysteresis)
    assert not measured_votes_pay(0.40, 0.504, 0.10, 1.0)[0]       # within hysteresis -> cold
    assert measured_votes_pay(0.40, 0.506, 0.10, 1.0)[0]
    assert not measured_votes_pay(0.58, 0.65, 0.17, 1.0)[0]        # strong bottle (agnews-like) -> cold
    assert measured_votes_pay(0.16, 0.28, 0.035, 1.0)[0]           # cheap votes, weak bottle (banking-like) -> warm
    # units cancel: scaling both costs leaves the choice unchanged
    assert measured_votes_pay(0.4, 0.6, 0.1, 1.0)[0] == measured_votes_pay(0.4, 0.6, 1e-6, 1e-5)[0]


def test_rescale_bottled():
    assert rescale_bottled(0.5, 400, 117, 400, 117) == 0.5
    assert rescale_bottled(0.5, 400, 117, 1200, 199) > 0.5          # a smaller labelled share bottles more
    assert rescale_bottled(0.9, 400, 117, 1200, 0) <= 1.0
    assert rescale_bottled(0.3, 10, 10, 400, 117) == 0.3            # degenerate source: unchanged


def test_auto_decision_follows_the_recorded_forecast():
    wl = toy_workload(900, seed=16)
    b = Bottler(_client(wl), SPEC, _cfg(seed=8, forecast_v2="off"))   # v0.7 policy (lagged exact cold arm)
    r0 = asyncio.run(b.run(wl.texts[:300]))
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=True))
    f = w.artifact.forecast
    nxt = wl.texts[600:]
    use, why = b._votes_pay(nxt, w.artifact, dense_next=False)
    assert why["basis"] == "measured" and why["forecast_source"] == "pour_warm"
    n_new = b._planned_n(len(nxt))
    cc = rescale_bottled(f["arms"]["cold"]["bottled"], f["N"], f["n_cal"], len(nxt), n_new)
    cw = rescale_bottled(f["arms"]["warm"]["bottled"], f["N"], f["n_cal"], len(nxt), n_new)
    assert abs(why["cov_cold"] - cc) < 1e-12 and abs(why["cov_warm"] - cw) < 1e-12
    assert use == (why["cost_warm"] < why["cost_cold"] - why["hysteresis"])
    # measured unit costs: realized cheap-vote USD per item and expert USD per call from the warm pour's ledger
    led = w.ledger
    assert abs(f["costs"]["vote_usd_per_item"] - led["cheap_vote"]["usd"] / f["N"]) < 1e-15
    calls = sum(v["calls"] for k, v in led.items() if k.startswith("expert"))
    assert abs(f["costs"]["expert_usd_per_call"] - sum(v["usd"] for k, v in led.items() if k.startswith("expert")) / calls) < 1e-15
    # a cold pour carries the gain (stale, decayed) and the vote cost
    c = asyncio.run(b.pour(w.artifact, nxt, votes=False))
    g = c.artifact.forecast["gain"]
    assert g["age"] == 1 and g["basis"] == "stale" and g["value"] == f["gain"]["value"]
    assert c.artifact.forecast["costs"]["vote_basis"] == "carried"
    arms = b._forecast_arms(c.artifact, c.artifact.forecast)
    assert abs(arms["warm"] - min(1.0, arms["cold"] + 0.8 * g["value"])) < 1e-12


def test_config_or_dense_mismatch_uses_the_prior_rule():
    wl = toy_workload(700, seed=17)
    b = Bottler(_client(wl), SPEC, _cfg(seed=3))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    w = asyncio.run(b.pour(r0.artifact, wl.texts[300:500], votes=True))
    other = Bottler(b.client, SPEC, _cfg(seed=3, eps=0.1))
    use, why = other._votes_pay(wl.texts[500:], w.artifact, dense_next=False)
    assert why["basis"] == "prior" and "configuration" in why["forecast"]["reason"]
    use, why = b._votes_pay(wl.texts[500:], w.artifact, dense_next=True)
    assert why["basis"] == "prior" and "dense" in why["forecast"]["reason"]
    off = Bottler(b.client, SPEC, _cfg(seed=3, forecast=False))
    assert off._votes_pay(wl.texts[500:], w.artifact)[1]["basis"] == "prior"


# ----------------------------------------------------------------------------------------------- replay (bench data)
def _bench():
    """The replay harness and stored model outputs of the paper's experiments (not part of this repository):
    $BOTTLEKIT_REPLAY_ROOT must hold bench/bench.py and its data. Without it these tests skip."""
    env = os.environ.get("BOTTLEKIT_REPLAY_ROOT")
    root = Path(env) if env else None
    if root is None or not (root / "bench" / "bench.py").exists():
        return None, None
    sys.path.insert(0, str(root / "bench"))
    import bench as B   # noqa: E402
    return B, root


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("task", ["agnews", "emotion", "esci", "banking77"])
def test_replay_counterfactual_equals_actual_cold_pour(task, dense):
    """>= 8 (task, seed) replay pairs on real stored LLM outputs (default + dense): exact tau and bottled count."""
    B, _ = _bench()
    if B is None:
        pytest.skip("replay data not available (set $BOTTLEKIT_REPLAY_ROOT)")
    import bottlekit as bk
    items, spec_d, votes, banks, emb = B.load_task(task)
    spec = bk.TaskSpec(name=task, instruction=spec_d["instruction"], labels=spec_d["labels"])
    texts = [it["text"] for it in items]
    exp_labels = [x or "__none__" for x in banks["oss-120b"]]
    client = B.ReplayClient(bk, spec, texts, votes, exp_labels, (0.05, 0.10, 0.425, 2.55))
    b = bk.Bottler(client, spec, bk.BottleConfig(cheap="cheap", expert="expert", seed=1003))
    A, P = slice(0, 400), slice(400, 800)
    kw = (lambda sl: {"emb": emb[sl]}) if dense else (lambda sl: {})
    r0 = asyncio.run(b.run(texts[A], **kw(A)))
    cold = asyncio.run(b.pour(r0.artifact, texts[P], votes=False, **kw(P)))
    warm = asyncio.run(b.pour(r0.artifact, texts[P], votes=True, **kw(P)))
    arm = warm.artifact.forecast["arms"]["cold"]
    assert arm["basis"] == "exact"
    assert arm["tau"] == cold.certificate["tau"]
    assert arm["bottled_count"] == sum(p == "bottle" for p in cold.provenance)


# ----------------------------------------------------------------------------------- edge cases
def test_staged_counterfactual_is_exercised():
    """The staged exactness case must not pass vacuously: at least one seed gives an exact cold arm, and every exact
    arm equals the actual cold pour."""
    n_exact = 0
    for seed in range(1, 9):
        wl = toy_workload(900, seed=40 + seed)
        client = _client(wl, acc={"cheap": 0.8, "expert": 0.93})
        b = Bottler(client, SPEC, _cfg(seed=seed, design="staged"))
        r0 = asyncio.run(b.run(wl.texts[:300]))
        cold = asyncio.run(b.pour(r0.artifact, wl.texts[300:900], votes=False))
        warm = asyncio.run(b.pour(r0.artifact, wl.texts[300:900], votes=True))
        arm = warm.artifact.forecast["arms"]["cold"]
        if arm["basis"] != "exact":
            continue
        n_exact += 1
        assert arm["tau"] == cold.certificate["tau"]
        assert arm["bottled_count"] == sum(p == "bottle" for p in cold.provenance)
    assert n_exact >= 1


def test_child_bottles_copy_the_forecast_dict():
    wl = toy_workload(600, seed=51)
    b = Bottler(_client(wl), SPEC, _cfg(seed=2))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    art = r0.artifact
    assert art.forecast is not None
    child = art.updated(wl.texts[300:310], [list(SPEC.labels)[0]] * 10)
    same = art.with_embeddings(None)
    assert child.forecast is not art.forecast and child.forecast == art.forecast
    assert same.forecast is not art.forecast and same.forecast == art.forecast
    child.forecast["mutated"] = True
    assert "mutated" not in art.forecast


def test_failed_forecast_leaves_next_auto_pour_on_the_prior_rule(monkeypatch):
    """A batch whose forecast raises must not let the next auto pour use an ancestor's arms as current."""
    wl = toy_workload(1200, seed=52)
    b = Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.93}), SPEC, _cfg(seed=3))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    p1 = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes=True))       # warm: records an exact forecast
    assert p1.artifact.forecast["artifact_version"] == p1.artifact.version

    async def boom(*a, **k):
        raise RuntimeError("forecast failed")
    monkeypatch.setattr(Bottler, "_forecast", boom)
    p2 = asyncio.run(b.pour(p1.artifact, wl.texts[600:900], votes=False))
    assert "forecast_error" in p2.diagnostics
    assert p2.artifact.forecast["artifact_version"] == p1.artifact.version != p2.artifact.version
    p3 = asyncio.run(b.pour(p2.artifact, wl.texts[900:], votes="auto"))
    why = p3.diagnostics["votes_decision"]
    assert why["basis"] == "prior"
    assert "stale forecast" in why["forecast"]["reason"]


def test_bottler_with_forecast_off_never_uses_an_inherited_forecast():
    wl = toy_workload(1200, seed=53)
    on = Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.93}), SPEC, _cfg(seed=4))
    off = Bottler(_client(wl, acc={"cheap": 0.8, "expert": 0.93}), SPEC, _cfg(seed=4, forecast=False))
    r0 = asyncio.run(on.run(wl.texts[:300]))
    p1 = asyncio.run(on.pour(r0.artifact, wl.texts[300:600], votes=True))
    p2 = asyncio.run(off.pour(p1.artifact, wl.texts[600:900], votes=False))     # writes no forecast
    p3 = asyncio.run(on.pour(p2.artifact, wl.texts[900:], votes="auto"))
    assert p3.diagnostics["votes_decision"]["basis"] == "prior"
