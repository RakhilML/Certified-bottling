"""Health report. Mock LLM only, no servers.

Report-only by construction: after tau, the answers, the escalation, the export and the forecast are fixed, each batch
gets diagnostics['health'] (agreement with the expert on the calibration sample with exact intervals, a valid lower
bound on the bottled-set precision, label-free drift statistics for pours) and the exported bottle gets one row of
`history` (its learning curve).
Covers: byte-identical outputs with health on/off (default, dense, fixed, staged, budget, memo, chunked, session);
history never changes a decision (votes='auto' included); CP conservative and the finite-population interval exact
under sampling WITHOUT replacement (exhaustive); the precision bound = 1 - u_delta/A from the batch's own law;
unbiased flags (staged -> False); memo scope; history through updated/with_embeddings/save/load, old pickles, cap;
the learning-curve rows of a run -> pour -> pour chain pinned by recomputation; drift statistics; non-fatal errors;
chunked aggregate; session report keys and config_hash.
"""

import asyncio
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta, hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.harness as H  # noqa: E402
import bottlekit.session as S  # noqa: E402
from bottlekit import (BottleArtifact, BottleConfig, Bottler, BottleSession, cp_interval, drift_stats,  # noqa: E402
                       finite_interval, format_learning_curve, learning_curve)
from bottlekit.certify import NONE  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

WL = toy_workload(2600, seed=91)
GOLD = {t[:400]: g for t, g in zip(WL.texts, WL.gold)}


def _client(acc=None, cls=MockClient):
    return cls(GOLD, acc=acc or {"cheap": 0.8, "expert": 0.95})


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=5)
    base.update(kw)
    return BottleConfig(**base)


def _emb(texts, d=8):
    out = []
    for t in texts:
        r = np.random.default_rng(abs(hash(t)) % (2 ** 32))
        v = r.normal(size=d)
        out.append(v / np.linalg.norm(v))
    return np.asarray(out, np.float32)


def _art_bytes(a):
    if a is None:
        return "none"
    return json.dumps({"texts": a.texts, "labels": a.labels, "src": list(a.src), "version": a.version,
                       "fingerprint": a.fingerprint, "forecast": H._plain(a.forecast),
                       "emb": None if a.emb is None else a.emb.tobytes().hex()}, sort_keys=True, default=str)


def _out(res):
    return json.dumps({"answers": res.answers, "provenance": res.provenance, "certificate": res.certificate,
                       "ledger": res.ledger}, sort_keys=True, default=str) + _art_bytes(res.artifact)


def _diag_wo_health(res):
    d = dict(res.diagnostics)
    d.pop("health", None)
    d.pop("wall_time_s", None)   # wall time (forecast-v2 decision timing) is never byte-identical
    return json.dumps(d, sort_keys=True, default=str)


def _chain(cfg, dense=False, client_cls=MockClient, sizes=(400, 400, 400)):
    """run -> warm pour -> cold pour -> cold pour (votes forced) and a votes='auto' pour; returns the results."""
    c = _client(cls=client_cls)
    b = Bottler(c, SPEC, cfg)
    T = WL.texts
    E = _emb(T) if dense else None
    a, o = 0, sizes[0]
    r0 = asyncio.run(b.run(T[a:o], emb=None if E is None else E[a:o]))
    out = [r0]
    art = r0.artifact
    for j, (n, votes) in enumerate(zip(sizes[1:] + (300,), (True, False, "auto"))):
        a, o = o, o + n
        r = asyncio.run(b.pour(art, T[a:o], votes=votes, emb=None if E is None else E[a:o]))
        out.append(r)
        art = r.artifact
    return out


# ----------------------------------------------------------------------------------------------- report-only
CONFIGS = {
    "default": {},
    "dense": {},
    "fixed": {"calibration": "fixed", "n_cal": 120},
    "staged": {"design": "staged"},
    "budget": {"expert_budget": 40},
    "student_mode": {"mode": "student"},
    "forecast_run": {"forecast_run": True},
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_health_on_off_byte_identical(name):
    on = _chain(_cfg(health=True, **CONFIGS[name]), dense=name == "dense")
    off = _chain(_cfg(health=False, **CONFIGS[name]), dense=name == "dense")
    for a, b in zip(on, off):
        assert _out(a) == _out(b)
        assert _diag_wo_health(a) == _diag_wo_health(b)       # only diagnostics['health'] differs
        assert "health" in a.diagnostics and "health" not in b.diagnostics
        assert "error" not in a.diagnostics["health"], a.diagnostics["health"]
        assert b.artifact.history == []                         # off: no row is ever appended
    assert len(on[-1].artifact.history) == len(on)


def test_health_on_off_identical_on_chunked_and_memo_pours():
    T = WL.texts
    for kw, votes in (({"chunking": "cold"}, False), ({"memo": True}, True)):
        res = {}
        for health in (True, False):
            c = _client()
            b = Bottler(c, SPEC, _cfg(health=health, **kw))
            r0 = asyncio.run(b.run(T[:300]))
            if kw.get("memo"):      # half of the pour recurs (memo answers it), half is novel
                texts = T[100:300] + T[300:700]
            else:                   # 2,000 >= 4 x 300: a chunked cold pour
                texts = T[300:2300]
            r1 = asyncio.run(b.pour(r0.artifact, texts, votes=votes))
            res[health] = r1
        assert _out(res[True]) == _out(res[False])
        assert _diag_wo_health(res[True]) == _diag_wo_health(res[False])
        h = res[True].diagnostics["health"]
        assert "error" not in h, h
        if kw.get("memo"):
            assert res[True].certificate.get("n_memo", 0) > 0
            assert h["scope"].startswith("population") and h["N"] == res[True].certificate["N_population"]
        else:
            assert res[True].certificate.get("chunked") and h["kind"] == "chunked"


def test_history_never_changes_a_decision():
    """A bottle with a (long, odd) history and the same bottle with history=[] pour identically, votes='auto'
    included (the decision record too): no decision reads the history."""
    out = _chain(_cfg())
    art = out[2].artifact                              # a bottle with a measured forecast and 3 history rows
    stripped = art.with_embeddings(None)
    stripped.history = []
    noisy = art.with_embeddings(None)
    noisy.history = noisy.history + [{"row": 99, "agree_bottle": 0.0, "conf_mean": 0.0, "precision_lower": 0.0}] * 50
    T = WL.texts[1200:1600]
    got = []
    for a in (art, stripped, noisy):
        c = _client()
        b = Bottler(c, SPEC, _cfg())
        r = asyncio.run(b.pour(a, T, votes="auto"))
        got.append((_out(r), _diag_wo_health(r),
                    json.dumps(r.diagnostics.get("votes_decision"), sort_keys=True, default=str)))
    assert got[0] == got[1] == got[2]
    assert H.Bottler._votes_pay.__code__.co_names.count("history") == 0
    assert H.Bottler._measured_votes.__code__.co_names.count("history") == 0
    assert H.Bottler.chunk_plan.__code__.co_names.count("history") == 0
    assert S._artifact_digest(art) == S._artifact_digest(stripped) == S._artifact_digest(noisy)


def test_health_does_not_touch_shared_random_state():
    """The global numpy and Python RNG states after a chain are the same with health on and off (the report uses
    only its own seeded generator), and its stream key is none of the pipeline's."""
    import random
    states = {}
    for health in (True, False):
        np.random.seed(123)
        random.seed(123)
        _chain(_cfg(health=health), sizes=(300, 300, 300))
        states[health] = (np.random.get_state()[1].copy(), random.getstate())
    assert (states[True][0] == states[False][0]).all() and states[True][1] == states[False][1]
    assert H.HEALTH_STREAM_KEY not in (1, 2, 7, 11, 13, H.PARTITION_STREAM_KEY, S.RESEED_TAG)


def test_health_errors_are_non_fatal(monkeypatch):
    ref = _chain(_cfg(health=False), sizes=(300, 300, 300))

    def boom(*a, **k):
        raise RuntimeError("drift boom")
    monkeypatch.setattr(H, "drift_stats", boom)
    got = _chain(_cfg(health=True), sizes=(300, 300, 300))
    for a, b in zip(got, ref):
        assert _out(a) == _out(b)
    assert "error" in got[1].diagnostics["health"]["drift"]

    def boom2(self, **k):
        raise RuntimeError("health boom")
    monkeypatch.setattr(H.Bottler, "_health", boom2)
    got = _chain(_cfg(health=True), sizes=(300, 300, 300))
    for a, b in zip(got, ref):
        assert _out(a) == _out(b)
        assert a.diagnostics["health"]["error"].startswith("RuntimeError")
        assert a.artifact.history == []


# ----------------------------------------------------------------------------------------------- intervals
@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (3, 17), (95, 117), (1, 1), (0, 1)])
def test_cp_interval_is_clopper_pearson(k, n):
    lo, hi = cp_interval(k, n, 0.95)
    assert lo == pytest.approx(0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1), abs=1e-12)
    assert hi == pytest.approx(1.0 if k == n else beta.ppf(0.975, k + 1, n - k), abs=1e-12)
    assert 0 <= lo <= k / n <= hi <= 1


@pytest.mark.parametrize("N,n", [(40, 15), (60, 30), (100, 10), (30, 29), (120, 40)])
def test_intervals_cover_under_sampling_without_replacement_exhaustive(N, n):
    """For EVERY population count M: P[interval covers M/N] >= 0.95 under the exact hypergeometric law of a uniform
    n-of-N sample. CP (binomial) is conservative; the finite-population interval is exact."""
    ks = np.arange(n + 1)
    cps = [cp_interval(int(k), n) for k in ks]
    fis = [finite_interval(int(k), n, N) for k in ks]
    widths_cp, widths_fi = [], []
    for M in range(N + 1):
        pm = hypergeom.pmf(ks, N, M, n)
        p = M / N
        cov_cp = sum(pm[k] for k in ks if cps[k][0] - 1e-12 <= p <= cps[k][1] + 1e-12)
        cov_fi = sum(pm[k] for k in ks if fis[k][0] - 1e-12 <= p <= fis[k][1] + 1e-12)
        assert cov_cp >= 0.95 - 1e-9, (M, cov_cp)
        assert cov_fi >= 0.95 - 1e-9, (M, cov_fi)
        widths_cp.append(sum(pm[k] * (cps[k][1] - cps[k][0]) for k in ks))
        widths_fi.append(sum(pm[k] * (fis[k][1] - fis[k][0]) for k in ks))
    assert np.mean(widths_fi) <= np.mean(widths_cp) + 1e-12      # the finite interval is the tighter one


def test_finite_interval_brute_force():
    N, n = 50, 12
    for k in range(n + 1):
        ok = [M for M in range(N + 1) if hypergeom.cdf(k, N, M, n) > 0.025 and
              (hypergeom.sf(k - 1, N, M, n) if k > 0 else 1.0) > 0.025]
        assert finite_interval(k, n, N) == (min(ok) / N, max(ok) / N)


# ----------------------------------------------------------------------------------------------- agreement + bound
def _cal_idx(res):
    return [i for i, p in enumerate(res.provenance) if p.startswith("expert_calibration") and
            not p.startswith("expert_calibration_dup")]


def test_learning_curve_rows_pinned_for_a_run_pour_pour_chain():
    """run -> warm pour -> cold pour: three rows whose values are recomputed independently from the outputs (the
    calibration answers are the expert's raw answers; the pre-escalation answers are recomputed from the bottles)."""
    c = _client()
    b = Bottler(c, SPEC, _cfg())
    T = WL.texts
    r0 = asyncio.run(b.run(T[:400]))
    r1 = asyncio.run(b.pour(r0.artifact, T[400:800], votes=True))
    r2 = asyncio.run(b.pour(r1.artifact, T[800:1200], votes=False))
    rows = learning_curve(r2.artifact)
    assert [r["row"] for r in rows] == [0, 1, 2]
    assert [r["kind"] for r in rows] == ["run", "pour_warm", "pour_cold"]
    assert [r["bottle_version"] for r in rows] == [None, 0, 1]
    assert [r["version_out"] for r in rows] == [0, 1, 2]
    assert [r["bottle_memory"] for r in rows] == [0, len(r0.artifact.texts), len(r1.artifact.texts)]
    assert [r["memory_out"] for r in rows] == [len(r0.artifact.texts), len(r1.artifact.texts), len(r2.artifact.texts)]
    assert [r["N"] for r in rows] == [400, 400, 400]
    for r, res in zip(rows, (r0, r1, r2)):
        assert r["bottled"] == round(res.coverage(), 6)
        assert r["n_cal"] == res.certificate["n_cal"] == len(_cal_idx(res))
        assert r["unbiased"] is True and r["delta"] == 0.1
    # pinned recomputation: the cold pour's bottle-alone agreement on its calibration sample
    cal = _cal_idx(r2)
    pred, _ = r1.artifact.predict(T[800:1200])
    k = sum(pred[i] == r2.answers[i] for i in cal)
    assert rows[2]["agree_bottle"] == rows[2]["agree_final"] == round(k / len(cal), 6)
    h2 = r2.diagnostics["health"]
    assert h2["agree"]["bottle"]["k"] == k and h2["agree"]["bottle"]["n"] == len(cal)
    lo, hi = cp_interval(k, len(cal), 0.95)                       # the reported interval is Clopper-Pearson
    assert rows[2]["agree_bottle_ci"] == [round(lo, 6), round(hi, 6)] and rows[2]["level"] == 0.95
    assert h2["agree"]["bottle"]["ci"] == list(cp_interval(k, len(cal), 0.95))
    assert h2["agree"]["bottle"]["ci_finite"] == list(finite_interval(k, len(cal), 400, 0.95))   # extra, exact
    # warm pour: the bottle alone is r0's bottle; the vote answers are this batch's votes
    cal1 = _cal_idx(r1)
    pred1, _ = r0.artifact.predict(T[400:800])
    assert r1.diagnostics["health"]["agree"]["bottle"]["k"] == sum(pred1[i] == r1.answers[i] for i in cal1)
    assert set(r1.diagnostics["health"]["agree"]) == {"final", "bottle", "vote"}
    # run(): the CROSS-FITTED student and the vote; never the exported bottle
    assert set(r0.diagnostics["health"]["agree"]) == {"final", "vote", "student"}
    assert rows[0]["agree_bottle"] is None and rows[0]["agree_student"] is not None
    # drift: pours only, conf next to the previous pour's
    assert rows[0]["conf_mean"] is None and rows[1]["conf_mean"] is not None
    assert r2.diagnostics["health"]["drift"]["conf_mean_prev"] == rows[1]["conf_mean"]
    txt = format_learning_curve(r2.artifact)
    assert txt.count("\n") == 3 and "pour_warm" in txt


def test_precision_bound_is_one_minus_u_delta_over_A():
    for res in _chain(_cfg()):
        h = res.diagnostics["health"]
        pr = h["precision"]
        A = sum(p == "bottle" for p in res.provenance)
        u = int(round(res.certificate["risk"]["eps_at_delta"]["0.1"] * len(res.answers)))
        assert pr["bottled"] == A and pr["u_delta"] == u
        assert pr["precision_lower"] == (max(0.0, 1 - u / A) if A else None)
        assert u <= int(np.floor(0.05 * len(res.answers)))    # a sub-event of the certified event: no new delta
        assert "guarantee" not in json.dumps(h["summary"]).lower() or "same event" in h["summary"]
        assert "health" not in res.certificate["guarantee"].lower()


def test_staged_and_fixed_designs_flags():
    T = WL.texts
    c = _client()
    hs = {}
    for name, kw in (("staged", {"design": "staged"}), ("fixed", {"calibration": "fixed", "n_cal": 150}),
                     ("single", {})):
        b = Bottler(c, SPEC, _cfg(**kw))
        r = asyncio.run(b.run(T[:1200]))
        hs[name] = r.diagnostics["health"]
    assert hs["staged"]["design"] == "staged" and len(H.default_design(1200)[0]) > 1
    assert hs["staged"]["unbiased"] is False and "indicative" in hs["staged"]["summary"]
    assert hs["fixed"]["unbiased"] is True and hs["fixed"]["design"] == "fixed"
    assert hs["single"]["unbiased"] is True and hs["single"]["design"] == "single"


def test_unusable_expert_answers_are_excluded_and_counted():
    class Flaky(MockClient):
        def _answer(self, alias, text, salt, pos=None, k=None):
            if alias == "expert" and hash(text) % 15 == 0:
                return "I am not sure."
            return super()._answer(alias, text, salt, pos, k)
    c = _client(cls=Flaky)
    b = Bottler(c, SPEC, _cfg())
    r = asyncio.run(b.run(WL.texts[:400]))
    h = r.diagnostics["health"]
    n_none = sum(1 for i in _cal_idx(r) if r.provenance[i].endswith("+fallback") or r.answers[i] == NONE)
    assert h["n_expert_none"] > 0 and h["n_usable"] + h["n_expert_none"] == h["n_cal"]
    assert h["n_expert_none"] == n_none
    assert h["agree"]["final"]["n"] == h["n_usable"]
    assert h["agree"]["final"]["ci_finite"] is None and h["agree"]["final"]["ci"] is not None
    assert "excluded" in h["unbiased_note"]


# ----------------------------------------------------------------------------------------------- history
def test_history_survives_updates_embeddings_save_load_and_is_capped(tmp_path):
    out = _chain(_cfg())
    art = out[-1].artifact
    assert len(art.history) == 4
    up = art.updated(["new text alpha"], ["alpha"])
    assert up.history == art.history and up.history is not art.history
    we = art.with_embeddings(_emb(art.texts))
    assert we.history == art.history
    p = tmp_path / "b.pkl"
    art.save(p)
    assert BottleArtifact.load(p).history == art.history
    d = pickle.loads(p.read_bytes())                     # a v0.8 pickle has no 'history' key
    del d["history"]
    p.write_bytes(pickle.dumps(d))
    old = BottleArtifact.load(p)
    assert old.history == [] and old.texts == art.texts
    big = BottleArtifact(SPEC, art.texts, art.labels, history=[{"row": i} for i in range(700)])
    assert len(big.history) == 700                       # the artifact is a container; the Bottler applies the cap
    assert _cfg().history_cap == H.HISTORY_CAP == 500
    r = asyncio.run(Bottler(_client(), SPEC, _cfg()).pour(big, WL.texts[1600:1900], votes=False))
    assert len(r.artifact.history) == 500 and [x["row"] for x in r.artifact.history] == list(range(201, 701))
    assert len(big.history) == 700                       # the parent bottle is untouched
    for r in art.history:
        assert set(r) == set(H.HISTORY_FIELDS)
        json.dumps(r)                                    # plain types only


def test_history_is_capped_in_a_long_lineage():
    out = _chain(_cfg(history_cap=3), sizes=(300, 300, 300))
    assert [r["row"] for r in out[-1].artifact.history] == [1, 2, 3]
    # one more batch: the row index keeps increasing after the cap dropped rows (never a repeated index)
    r = asyncio.run(Bottler(_client(), SPEC, _cfg(history_cap=3)).pour(out[-1].artifact, WL.texts[1500:1800],
                                                                        votes=False))
    assert [x["row"] for x in r.artifact.history] == [2, 3, 4]
    ref = _chain(_cfg(), sizes=(300, 300, 300))
    for a, b in zip(out, ref):                           # the cap changes no output
        assert _out(a) == _out(b)


# ----------------------------------------------------------------------------------------------- drift
def test_drift_stats_values():
    T = WL.texts[:300]
    art = BottleArtifact(SPEC, T, [GOLD[t[:400]] for t in T])
    new = WL.texts[300:500]
    pred, conf = art.predict(new)
    d = drift_stats(art, new, conf, pred, seed=1)
    assert abs(d["conf_mean"] - float(np.mean(conf))) < 1e-12
    assert abs(d["conf_p10"] - float(np.percentile(conf, 10))) < 1e-12
    from collections import Counter
    pm, mm = Counter(pred), Counter(art.labels)
    tv = 0.5 * sum(abs(pm[k] / len(pred) - mm[k] / len(art.labels)) for k in set(pm) | set(mm))
    assert abs(d["label_tv"] - tv) < 1e-12
    an, vocab = art.student.wv.build_analyzer(), art.student.wv.vocabulary_
    toks = [t for x in new for t in an(x)]
    assert abs(d["oov_mass"] - sum(t not in vocab for t in toks) / len(toks)) < 1e-12
    same = drift_stats(art, T, *art.predict(T)[::-1], seed=1)
    assert same["oov_mass"] < d["oov_mass"]                     # memory texts: only pruned n-grams are unseen
    assert "cos_mean" not in d                                   # TF-IDF bottle: no dense statistics


def test_drift_stats_dense_and_subsample():
    T = WL.texts[:400]
    E = _emb(WL.texts[:700])
    art = BottleArtifact(SPEC, T, [GOLD[t[:400]] for t in T], emb=E[:400])
    new, en = WL.texts[400:700], E[400:700]
    pred, conf = art.predict(new, emb=en)
    d = drift_stats(art, new, conf, pred, emb=en, seed=3)
    M = art.emb.astype(np.float32)
    M = M / np.linalg.norm(M, axis=1, keepdims=True)
    best = (en / np.linalg.norm(en, axis=1, keepdims=True)) @ M.T
    assert abs(d["cos_mean"] - float(best.max(1).mean())) < 1e-5
    self_d = drift_stats(art, T, *art.predict(T, emb=E[:400])[::-1], emb=E[:400], seed=3)
    assert self_d["cos_mean"] > 0.999                           # every memory item finds itself
    a = drift_stats(art, new, conf, pred, emb=en, seed=3, max_rows=100)
    b = drift_stats(art, new, conf, pred, emb=en, seed=3, max_rows=100)
    assert a == b and a["n_used"] == 100 and a["memory_rows_used"] == 100     # seeded, label-free subsample


# ----------------------------------------------------------------------------------------------- chunked
def test_chunked_pour_health_aggregate_and_tagged_rows():
    T = WL.texts
    c = _client()
    b = Bottler(c, SPEC, _cfg(chunking="cold"))
    r0 = asyncio.run(b.run(T[:300]))
    r = asyncio.run(b.pour(r0.artifact, T[300:2300], votes=False))
    assert r.certificate.get("chunked")
    J = len(r.certificate["plan"])
    h = r.diagnostics["health"]
    assert h["chunks"] == J and len(h["per_chunk"]) == J and h["unbiased"] is True
    sizes = r.certificate["plan"]
    rates = [pc["agree"]["final"]["rate"] for pc in h["per_chunk"]]
    assert abs(h["agree"]["final"]["rate"] - sum(n * x for n, x in zip(sizes, rates)) / 2000) < 1e-12
    lo, hi = h["agree"]["final"]["ci"]
    assert lo <= h["agree"]["final"]["rate"] <= hi
    A = sum(p == "bottle" for p in r.provenance)
    u = int(round(r.certificate["risk"]["eps_at_delta"]["0.1"] * 2000))
    assert h["precision"]["precision_lower"] == max(0.0, 1 - u / A)
    rows = r.artifact.history
    assert [x["chunk"] for x in rows[-J:]] == list(range(J)) and all(x["chunks"] == J for x in rows[-J:])
    assert rows[0]["chunk"] is None                               # the run() row is not part of the pour


# ----------------------------------------------------------------------------------------------- session
def test_session_report_health_keys_and_config_hash():
    assert S.config_hash(_cfg(health=True)) == S.config_hash(_cfg(health=False))
    assert S.config_hash(_cfg()) == S.config_hash(_cfg(health_level=0.9, history_cap=7, health_max_rows=None))
    assert S.config_hash(_cfg(seed=5)) != S.config_hash(_cfg(seed=6))
    c = _client()
    s = BottleSession(c, SPEC, _cfg(), min_batch=0, votes=False)
    T = WL.texts
    for a, o in ((0, 400), (400, 800), (800, 1100)):
        asyncio.run(s.submit(T[a:o]))
    rep = s.report()
    assert set(rep) == {"rows", "totals", "statement", "health"}
    hr = rep["health"]
    assert len(hr["batches"]) == 3 and len(hr["learning_curve"]) == 3
    assert hr["batches"][1]["agree_bottle"]["rate"] == s.batches[1]["diagnostics"]["health"]["agree"]["bottle"]["rate"]
    assert hr["learning_curve_table"].count("\n") == 3
    assert "health" not in json.dumps(rep["rows"]) and "health" not in rep["statement"].lower()


def test_session_resumes_with_health_toggled(tmp_path):
    c = _client()
    p = tmp_path / "s.json"
    s = BottleSession(c, SPEC, _cfg(health=True), state_path=p, min_batch=0, votes=False)
    asyncio.run(s.submit(WL.texts[:400]))
    s2 = BottleSession.resume(p, _client(), SPEC, _cfg(health=False))
    assert len(s2.artifact.history) == 1
    asyncio.run(s2.submit(WL.texts[400:800]))
    assert "health" not in s2.batches[-1]["diagnostics"] and len(s2.artifact.history) == 1


# ----------------------------------------------------------------------------------------------- settings (v0.9)
def test_health_settings_are_validated():
    for kw in ({"health_level": 0.0}, {"health_level": 1.0}, {"history_cap": 0}, {"health_max_rows": 0}):
        with pytest.raises(ValueError):
            _cfg(**kw)
    _cfg(health_max_rows=None, health_level=0.8, history_cap=1)


def test_health_level_changes_only_the_intervals():
    """health_level sets the level of every reported interval and nothing else (answers, certificate, rates)."""
    a = _chain(_cfg(health_level=0.95), sizes=(300, 300, 300))
    b = _chain(_cfg(health_level=0.80), sizes=(300, 300, 300))
    for x, y in zip(a, b):
        assert _out(x) == _out(y)
        hx, hy = x.diagnostics["health"], y.diagnostics["health"]
        assert hx["level"] == 0.95 and hy["level"] == 0.80
        for name, e in hx["agree"].items():
            f = hy["agree"][name]
            assert (e["k"], e["n"], e["rate"]) == (f["k"], f["n"], f["rate"])
            assert e["ci"] == list(cp_interval(e["k"], e["n"], 0.95)) and f["ci"] == list(cp_interval(f["k"], f["n"], 0.80))
            assert e["ci"][0] <= f["ci"][0] <= f["ci"][1] <= e["ci"][1]
    assert b[-1].artifact.history[-1]["level"] == 0.8 and "80%" in format_learning_curve(b[-1].artifact)


def test_budget_capped_precision_states_its_inversion_basis():
    out = _chain(_cfg(expert_budget=40), sizes=(300, 300, 300))
    seen = 0
    for res in out:
        pr = res.diagnostics["health"]["precision"]
        if pr.get("precision_lower") is None:
            continue
        seen += 1
        assert pr["basis"] == res.certificate["risk"]["basis"] == "budget split"     # v0.9.1: every budget batch
        assert "Bonferroni split" in pr["statement"] and "eps_achieved" in pr["statement"]
        assert "not a separate claim" in pr["statement"]
        N = res.certificate["N"]
        if res.certificate.get("budget_capped"):     # the same inversion at delta/2 as the batch's eps_achieved
            assert pr["u_delta"] == int(round(res.certificate["eps_achieved"] * N))
        else:                                       # the k-rule's quantile at delta/2: never above floor(eps N)
            assert pr["u_delta"] <= int(np.floor(res.certificate["eps"] * N))
    assert seen >= 1


def test_drift_stats_reads_everything_by_default_and_respects_the_config_cap():
    T = WL.texts[:300]
    art = BottleArtifact(SPEC, T, [GOLD[t[:400]] for t in T])
    new = WL.texts[300:800]
    pred, conf = art.predict(new)
    d = drift_stats(art, new, conf, pred)
    assert d["n_used"] == 500
    r0 = asyncio.run(Bottler(_client(), SPEC, _cfg(health_max_rows=50)).run(WL.texts[:300]))
    r1 = asyncio.run(Bottler(_client(), SPEC, _cfg(health_max_rows=50)).pour(r0.artifact, new, votes=False))
    assert r1.diagnostics["health"]["drift"]["n_used"] == 50
