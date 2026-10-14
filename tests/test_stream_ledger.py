"""Tests for the exact stream ledger. Mock LLM only, no servers.

(1) certify.StreamLaw == certify.batch_tail on the same designs (every u, <= 1e-12); order-invariant; its mean ==
    the sum of the closed-form expectations; to_plain/from_plain rebuilds identical numbers; union-bound law of a
    staged design == risk_profile's union tail; runtime and checkpoint size at 1,000 batches.
(2) validity: Monte Carlo of the all-bad worst-case workload through the ACTUAL k-rule (certify.fixed_sequence) on
    2,000 streams of 5-20 batches: empirical tail <= reported tail + 3se at every eps' (the workload ATTAINS the
    bound, so 2se fails by chance at ~2.3% of points), and the realized stream
    share exceeds eps_at_delta[d] in <= d + 2se of streams.
(3) BottleSession: outputs byte-identical with the ledger on and off; stream_risk == batch_tail of the certificates'
    designs; checkpoint/resume (mid-stream and after a crash) reproduces identical numbers; an old checkpoint without
    a ledger is rebuilt from the certificates; memo-restricted and budget-capped batches withdraw the claim (status
    'none'); chunked pours add one law per chunk; staged designs use the union-bound law; forced flushes are noted;
    Bottler.stream_risk on run()/pour() certificates matches.
"""

import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottlekit import BottleConfig, Bottler, BottleSession  # noqa: E402
from bottlekit.certify import (RISK_EPS_GRID, StreamLaw, batch_tail, design_law, expected_eps_closed_form,  # noqa: E402
                               fixed_sequence, k_used, risk_profile, single_look_size)
from bottlekit.harness import stream_design, stream_risk  # noqa: E402
from bottlekit.session import _norm  # noqa: E402
from test_bottlekit import SPEC  # noqa: E402
from test_session import GOLD, WL, CrashClient, _batch_view, _client, _subs  # noqa: E402


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=11)
    base.update(kw)
    return BottleConfig(**base)


def _design(N, eps=0.05, delta=0.1):
    n = single_look_size(N, eps, delta)
    return n, N, k_used(n, N, eps, delta, "rectified")


def _surv_of(law, upto):
    return np.array([law.tail(u) for u in range(upto)])


# ----------------------------------------------------------------------------------------------- (1) the law
@pytest.mark.parametrize("designs", [
    [(106, 355, 3)],
    [_design(355)] * 5,
    [_design(N) for N in (355, 400, 454, 380, 120, 60)],
    [(53, 100, 0), (2, 5, 0), (300, 1200, 9), (1, 10, -1), (40, 40, 3)],
])
def test_streamlaw_equals_batch_tail_mean_and_order(designs):
    bt = batch_tail(designs, eps=0.05, survival=True)
    s = np.array(bt["survival"])
    law = StreamLaw().add_designs(designs)
    got = _surv_of(law, len(s) + 3)
    ref = np.concatenate([s, np.zeros(3)])
    assert np.max(np.abs(got - ref)) <= 1e-12
    assert law.N == sum(N for _, N, _ in designs)
    cf = sum(expected_eps_closed_form(n, N, k) * N for n, N, k in designs)
    assert law.expected_closed_form == pytest.approx(cf, rel=1e-12)
    assert law.mean() == pytest.approx(cf, rel=1e-9, abs=1e-9)
    rev = StreamLaw().add_designs(designs[::-1])
    assert np.max(np.abs(_surv_of(rev, len(s) + 3) - got)) <= 1e-12
    eps = 0.05
    assert law.tail(math.floor(eps * law.N)) == pytest.approx(bt["delta_batch"], abs=1e-12)
    for d, v in law.eps_at_delta((0.5, 0.1, 0.05, 0.01)).items():
        assert v == pytest.approx(bt["eps_at_delta"][d], abs=1e-12)
    re = StreamLaw.from_plain(json.loads(json.dumps(law.to_plain())))
    assert re.summary(eps, n_unusable=3) == law.summary(eps, n_unusable=3)


def test_summary_adds_unusable_share_and_never_folds_it_into_the_tail():
    law = StreamLaw().add_designs([_design(400)] * 4)
    a, b = law.summary(0.05, n_unusable=0), law.summary(0.05, n_unusable=16)
    assert a["tail"] == b["tail"] and a["delta_stream"] == b["delta_stream"]
    for d in a["eps_at_delta"]:
        assert b["eps_at_delta"][d] == pytest.approx(a["eps_at_delta"][d] + 16 / 1600)
        assert b["eps_at_delta_U"][d] == a["eps_at_delta_U"][d]


def test_union_bound_law_matches_risk_profile_and_dominates_each_look():
    looks, N = ((100, 0), (300, 8)), 1200
    p, o = design_law(looks, N)
    assert o == 0.0 and abs(p.sum() - 1) < 1e-9
    G = 1 - np.cumsum(p)
    rp = risk_profile([100, 300], N, [0, 8], 0.05, 0.1, eps_grid=RISK_EPS_GRID)
    for e, v in rp["tail"].items():
        assert G[int(math.floor(float(e) * N))] == pytest.approx(v, abs=1e-12)
    for n, k in looks:          # the union law dominates each look's own exact law
        q, _ = design_law(((n, k),), N)
        Gq = 1 - np.cumsum(q)
        m = min(len(G), len(Gq))
        assert np.all(G[:m] >= Gq[:m] - 1e-12)
    pm, om = design_law(((10, -1), (20, -1)), 50)          # nothing certifiable: U = 0 surely
    assert list(pm) == [1.0] and om == 0.0


def test_runtime_and_size_at_1000_batches():
    rng = np.random.default_rng(0)
    sizes = rng.integers(355, 455, size=1000)
    for N in set(sizes.tolist()):        # warm the per-design cache (a real stream pays this once per distinct N)
        design_law(((single_look_size(N, 0.05, 0.1), _design(N)[2]),), N)
    law, worst = StreamLaw(), 0.0
    for N in sizes:
        n, N, k = _design(int(N))
        t0 = time.perf_counter()
        law.add(n, N, k)
        law.summary(0.05)
        worst = max(worst, time.perf_counter() - t0)
    assert worst < 0.05
    plain = json.dumps(law.to_plain())
    assert len(plain) < 100_000
    # a session's compact ledger entries for 1,000 batches
    entries = [{"b": i, "N": int(N), "F": 0, "t": "auto", "x": 0, "ok": True, "d": [[[[int(_design(int(N))[0]),
                                                                                    int(_design(int(N))[2])]], int(N)]]}
               for i, N in enumerate(sizes)]
    assert len(json.dumps(entries)) < 100_000
    s = law.summary(0.05)
    assert s["overflow"] < 1e-12 and s["tail"]["0.05"] < 1e-12            # 1,000 x ~400 items: far below eps
    assert s["eps_at_delta"]["0.1"] < s["expected_share_bound"] + 0.002


# ----------------------------------------------------------------------------------------------- (2) validity
def _allbad_U(rng, n, N, eps, delta):
    """One batch of the all-bad workload through the actual closed-form rule (certify.fixed_sequence)."""
    scores = rng.random(N)
    cal = rng.choice(N, size=n, replace=False)
    tau, _ = fixed_sequence(scores[cal], np.ones(n, bool), N, eps, delta, "rectified")
    unl = np.ones(N, bool)
    unl[cal] = False
    return int(((scores >= tau) & unl).sum())


def test_validity_allbad_worst_case_streams():
    rng = np.random.default_rng(2026)
    eps, delta = 0.05, 0.1
    grid = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
    deltas = (0.5, 0.1, 0.01)
    n_streams = 2000
    hits = {e: [] for e in grid}
    reps = {e: [] for e in grid}
    over = {d: [] for d in deltas}
    for _ in range(n_streams):
        B = int(rng.integers(5, 21))                       # horizon and sizes drawn BEFORE any outcome
        Ns = rng.integers(120, 601, size=B)
        law, W = StreamLaw(), 0
        for N in Ns:
            n, N, k = _design(int(N), eps, delta)
            law.add(n, N, k)
            W += _allbad_U(rng, n, N, eps, delta)
        for e in grid:
            reps[e].append(law.tail(math.floor(e * law.N)))
            hits[e].append(W > math.floor(e * law.N))
        ead = law.eps_at_delta(deltas)
        for d in deltas:
            over[d].append(W / law.N > ead[f"{d:g}"])
    for e in grid:
        h = np.asarray(hits[e], float)
        p = float(np.mean(reps[e]))
        se = math.sqrt(max(p * (1 - p), 1.0 / n_streams) / n_streams)
        # The all-bad workload ATTAINS the bound (P[U_b > u] = g_b(u) exactly), so the empirical tail sits ON the
        # reported one and a 2se band is exceeded by chance at ~2.3% of points; 3se per point here. The large
        # standalone simulation (2e5 streams) confirms the z-scores.
        assert h.mean() <= p + 3 * se, (e, h.mean(), p)
        # tight, not just valid
        assert h.mean() >= p - 4 * se - 1e-3, (e, h.mean(), p)
    for d in deltas:
        h = np.asarray(over[d], float)
        se = math.sqrt(d * (1 - d) / n_streams)
        assert h.mean() <= d + 2 * se, (d, h.mean())


# ----------------------------------------------------------------------------------------------- (3) session
def _run_session(cfg, sizes, start=0, min_batch=150, votes="auto", force=True, **kw):
    s = BottleSession(_client(), SPEC, cfg, min_batch=min_batch, votes=votes, **kw)
    for t in _subs(sizes, start):
        asyncio.run(s.submit(t))
    if force:
        asyncio.run(s.flush(force=True))
    return s


def _designs_of(s):
    out = []
    for rec in s.batches:
        sd = stream_design(rec["certificate"], rec["N"])
        assert sd["ok"], sd
        out += sd["designs"]
    return out


def _ref_law(designs):
    law = StreamLaw()
    for looks, N in designs:
        law.add_design(looks, N)
    return law


def test_session_ledger_on_off_byte_identical_and_equals_batch_tail():
    cfg = _cfg(seed=31)
    on = _run_session(cfg, [70] * 10)
    off = _run_session(cfg, [70] * 10, stream_ledger=False)
    assert len(on.batches) == len(off.batches) >= 4
    for a, b in zip(on.batches, off.batches):
        assert _batch_view(a) == _batch_view(b)
    ra, rb = on.report(), off.report()
    assert json.dumps(_norm(ra["rows"]), sort_keys=True) == json.dumps(_norm(rb["rows"]), sort_keys=True)
    ta, tb = dict(ra["totals"]), dict(rb["totals"])
    sa, sb = ta.pop("stream_risk"), tb.pop("stream_risk")
    assert ta == tb and sb["status"] == "off" and sa["status"] == "exact"
    assert rb["statement"] in ra["statement"] and "Stream (fixed horizon" in ra["statement"]
    assert "does not say that no batch fails" in ra["statement"]
    # equals batch_tail on the certificates' designs (all single looks here)
    singles = [(looks[0][0], N, looks[0][1]) for looks, N in _designs_of(on)]
    bt = batch_tail(singles, eps=0.05, deltas=(0.5, 0.1, 0.01, 0.001))
    assert sa["N_included"] == sum(r["N"] for r in on.batches) == bt["N"]
    assert sa["delta_stream"] == pytest.approx(bt["delta_batch"], abs=1e-12)
    for e, v in bt["tail"].items():
        assert sa["tail"][e] == pytest.approx(v, abs=1e-12)
    F = sum(int(r["certificate"].get("n_expert_unusable") or 0) for r in on.batches)
    for d, v in bt["eps_at_delta"].items():
        assert sa["eps_at_delta_U"][d] == pytest.approx(v, abs=1e-12)
        assert sa["eps_at_delta"][d] == pytest.approx(v + F / bt["N"], abs=1e-12)
    assert sa["expected_errors_bound"] == pytest.approx(ra["totals"]["expected_errors_bound"], rel=1e-9)
    assert sa["batches_included"] == len(on.batches) and sa["batches_excluded"] == 0
    assert sa["forced_flushes"] == 1 and any("force=True" in x for x in sa["notes"])
    # the run()/pour() helper on the same certificates gives the same numbers
    hr = Bottler(_client(), SPEC, cfg).stream_risk([r["certificate"] for r in on.batches])
    assert hr["status"] == "exact" and hr["tail"] == sa["tail"] and hr["eps_at_delta"] == sa["eps_at_delta"]


def test_session_resume_reproduces_stream_risk_and_old_checkpoints_rebuild(tmp_path):
    cfg = _cfg(seed=41)
    p = tmp_path / "s.json"
    s = BottleSession(_client(), SPEC, cfg, state_path=p, min_batch=150, votes="auto")
    subs = _subs([70] * 9)
    for t in subs[:5]:
        asyncio.run(s.submit(t))
    r = BottleSession.resume(p, _client(), SPEC, cfg)
    assert json.dumps(_norm(r.report()["totals"]["stream_risk"]), sort_keys=True) == \
        json.dumps(_norm(s.report()["totals"]["stream_risk"]), sort_keys=True)
    for t in subs[5:]:
        asyncio.run(s.submit(t))
    asyncio.run(s.flush(force=True))
    st = json.loads(p.read_text(encoding="utf-8"))
    assert len(st["stream"]["entries"]) == len(s.batches) and "pmf" not in json.dumps(st["stream"])
    r = BottleSession.resume(p, _client(), SPEC, cfg)
    sr_s, sr_r = s.report()["totals"]["stream_risk"], r.report()["totals"]["stream_risk"]
    assert json.dumps(_norm(sr_s), sort_keys=True) == json.dumps(_norm(sr_r), sort_keys=True)
    assert r._stream_law.summary(0.05) == s._stream_law.summary(0.05)
    # the lineage journal carries the entries too
    ev = [json.loads(x) for x in Path(str(p) + ".lineage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["stream"] for e in ev if e.get("ev") == "commit"] == st["stream"]["entries"]
    # an old checkpoint (written before the ledger): rebuilt from the certificates, same numbers, flagged
    del st["stream"]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(st), encoding="utf-8")
    o = BottleSession.resume(old, _client(), SPEC, cfg)
    so = o.report()["totals"]["stream_risk"]
    assert so["status"] == "exact"
    for k in ("tail", "eps_at_delta", "delta_stream", "N_included", "expected_errors_bound"):
        assert so[k] == sr_s[k]
    assert any("rebuilt" in x for x in so["notes"]) and any("no recorded flush trigger" in x for x in so["notes"])


@pytest.mark.parametrize("dense", [False])
def test_crash_mid_batch_resume_gives_the_same_stream_risk(tmp_path, dense):
    subs = _subs([60] * 11)
    cfg = _cfg(seed=21)
    u = BottleSession(_client(), SPEC, cfg, state_path=tmp_path / "u.json", min_batch=150, votes="auto")
    for t in subs:
        asyncio.run(u.submit(t))
    asyncio.run(u.flush(force=True))
    exp_per_batch = [sum(d["calls"] for k, d in r["ledger"].items() if k.startswith("expert")) for r in u.batches]
    cc = CrashClient(GOLD, acc={"cheap": 0.82, "expert": 0.95}, crash_at=sum(exp_per_batch[:2]) + 7)
    p = tmp_path / "c.json"
    c = BottleSession(cc, SPEC, cfg, state_path=p, min_batch=150, votes="auto")
    j0 = None
    for j, t in enumerate(subs):
        try:
            asyncio.run(c.submit(t))
        except RuntimeError:
            j0 = j
            break
    assert j0 is not None
    r = BottleSession.resume(p, _client(), SPEC, cfg)
    asyncio.run(r.flush())
    for t in subs[j0 + 1:]:
        asyncio.run(r.submit(t))
    asyncio.run(r.flush(force=True))
    a, b = u.report()["totals"]["stream_risk"], r.report()["totals"]["stream_risk"]
    assert json.dumps(_norm(a), sort_keys=True) == json.dumps(_norm(b), sort_keys=True)
    assert [e["t"] for e in r._stream] == [e["t"] for e in u._stream]     # the retry kept its reserved trigger


def test_memo_restricted_batch_withdraws_the_stream_claim():
    cfg = _cfg(seed=8, memo=True)
    s = BottleSession(_client(), SPEC, cfg, min_batch=0, votes="auto")
    texts = WL.texts
    for t in (texts[0:120], texts[120:220], texts[60:160] + texts[220:260], texts[260:330]):
        asyncio.run(s.submit(t))
    sr = s.report()["totals"]["stream_risk"]
    assert any(r["certificate"].get("n_memo") for r in s.batches)
    assert sr["status"] == "none" and sr["batches_excluded"] >= 1
    assert all("memo-restricted" in x["reason"] for x in sr["excluded"])
    assert "No stream-level tail is claimed" in s.report()["statement"]
    assert "tail" not in sr and "eps_at_delta" not in sr


def test_budget_capped_batch_withdraws_but_non_binding_budget_is_covered():
    s = _run_session(_cfg(seed=13, expert_budget=15), [70] * 7, min_batch=120)
    capped = [r["batch"] for r in s.batches if r["certificate"].get("budget_capped")]
    sr = s.report()["totals"]["stream_risk"]
    assert capped and sr["status"] == "none" and [x["batch"] for x in sr["excluded"]] == capped
    s2 = _run_session(_cfg(seed=13, expert_budget=10_000), [70] * 7, min_batch=120)
    assert not any(r["certificate"].get("budget_capped") for r in s2.batches)
    assert s2.report()["totals"]["stream_risk"]["status"] == "exact"


def test_chunked_pour_adds_one_law_per_chunk():
    cfg = _cfg(seed=5, chunking="auto", eps=0.1)
    s = BottleSession(_client(), SPEC, cfg, min_batch=0, votes=False)
    asyncio.run(s.submit(WL.texts[:200]))
    asyncio.run(s.submit(WL.texts[200:1400]))
    cert = s.batches[1]["certificate"]
    assert cert.get("chunked") and cert["plan_consistent"]
    sr = s.report()["totals"]["stream_risk"]
    assert sr["status"] == "exact"
    assert sr["designs"] == 1 + len(cert["chunk_designs"])
    singles = [(looks[0][0], N, looks[0][1]) for looks, N in _designs_of(s)]
    assert singles[1:] == [tuple(d) for d in cert["chunk_designs"]]
    bt = batch_tail(singles, eps=0.1, deltas=(0.5, 0.1, 0.01, 0.001))
    assert sr["delta_stream"] == pytest.approx(bt["delta_batch"], abs=1e-12)
    assert sr["tail"] == pytest.approx(bt["tail"], abs=1e-12)


def test_staged_design_uses_the_union_bound_law():
    s = _run_session(_cfg(seed=17, design="staged"), [130] * 4, min_batch=250)
    ds = _designs_of(s)
    assert any(len(looks) > 1 for looks, _ in ds)
    sr = s.report()["totals"]["stream_risk"]
    assert sr["status"] == "exact"
    ref = _ref_law(ds).summary(0.05, n_unusable=sr["n_expert_unusable"])
    assert sr["tail"] == ref["tail"] and sr["eps_at_delta"] == pytest.approx(ref["eps_at_delta"])
    # each batch's own law: the stream tail at one batch equals its union-bound risk profile
    one = _ref_law(ds[:1])
    rp = s.batches[0]["certificate"]["risk"]
    for e, v in rp["tail"].items():
        assert one.tail(math.floor(float(e) * one.N)) == pytest.approx(v, abs=1e-12)


def test_chunking_auto_after_an_output_without_answer_withdraws():
    s = BottleSession(_client(), SPEC, _cfg(seed=3, chunking="auto"), min_batch=0, votes=False)
    asyncio.run(s.submit(WL.texts[:150]))
    s._stream[0]["x"] = 1                      # pretend batch 0 left an output without a usable answer
    asyncio.run(s.submit(WL.texts[150:300]))
    sr = s.report()["totals"]["stream_risk"]
    assert sr["status"] == "none" and "chunking='auto'" in sr["excluded"][0]["reason"]
    s2 = BottleSession(_client(), SPEC, _cfg(seed=3, chunking="off"), min_batch=0, votes=False)  # chunking off: not affected
    asyncio.run(s2.submit(WL.texts[:150]))
    s2._stream[0]["x"] = 1
    asyncio.run(s2.submit(WL.texts[150:300]))
    assert s2.report()["totals"]["stream_risk"]["status"] == "exact"


def test_empty_session_and_helper_on_mixed_certificates():
    s = BottleSession(_client(), SPEC, _cfg(), min_batch=150)
    sr = s.report()["totals"]["stream_risk"]
    assert sr["status"] == "empty"
    b = Bottler(_client(), SPEC, _cfg(seed=2))
    r1 = asyncio.run(b.run(WL.texts[:300]))
    r2 = asyncio.run(b.pour(r1.artifact, WL.texts[300:700], votes=False))
    ok = stream_risk([r1.certificate, r2.certificate], 0.05, 0.1)
    assert ok["status"] == "exact" and ok["N_included"] == 700
    bad = dict(r2.certificate, budget_capped=True)
    assert stream_risk([r1.certificate, bad], 0.05, 0.1)["status"] == "none"


def test_cert_streams_reads_the_chunked_top_level_key_list():
    """_pour_chunked records cal_stream as a list of per-chunk keys; cert_streams must read it (a TypeError there
    would crash a session with chunking='auto' on its first chunked pour)."""
    from bottlekit.session import cert_streams
    chunked = {"chunked": True, "cal_stream": [[3, 1, 2, 400], [3, 1, 3, 800]],
               "chunks": [{"cal_stream": [3, 1, 2, 400]}, {"cal_stream": [3, 1, 3, 800]}]}
    assert cert_streams(chunked) == [(3, 1, 2, 400), (3, 1, 3, 800)]
    assert cert_streams({"cal_stream": [[3, 1, 2, 400], [3, 1, 3, 800]]}) == [(3, 1, 2, 400), (3, 1, 3, 800)]
    assert cert_streams({"cal_stream": [3, 1, 2, 400]}) == [(3, 1, 2, 400)]


def test_cold_gate_with_auto_votes_withdraws_the_stream_claim():
    """chunking='cold' with votes='auto': a pour that passed the cold gate has a
    design chosen by the measured forecast (label-dependent), so the stream claim is withdrawn; the report says that
    withdrawal is possible and that the probability is over the whole procedure."""
    import asyncio
    from bottlekit import BottleConfig, BottleSession
    from test_bottlekit import SPEC, MockClient, toy_workload
    wl = toy_workload(2400, seed=61)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.82, "expert": 0.95})
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=9, chunking="cold")
    s = BottleSession(client, SPEC, cfg, min_batch=0, votes="auto")
    asyncio.run(s.submit(wl.texts[:300]))                 # run(): memory 300
    asyncio.run(s.submit(wl.texts[300:2400]))             # 2,100 >= 4 x 300: passes the cold gate
    ch = s.batches[-1]["diagnostics"]["chunking"]
    assert ch.get("rule") == "cold" and ch.get("decision") in ("warm", "cold")
    sr = s.report()["totals"]["stream_risk"]
    assert sr["status"] == "none" and "label-dependent design" in sr["excluded"][0]["reason"]
    assert sr["withdrawal_possible"] and "chunking='cold' with votes='auto'" in sr["withdrawal_reasons"]
    assert "NOT conditional" in sr["probability_basis"]


def test_cold_gate_not_passed_is_not_affected_by_missing_answers():
    """chunking='cold': a pour that FAILED the cold gate keeps its stream claim after an earlier output
    without a usable answer (a larger memory would fail the gate too)."""
    s = BottleSession(_client(), SPEC, _cfg(seed=3, chunking="cold"), min_batch=0, votes=False)
    asyncio.run(s.submit(WL.texts[:150]))
    s._stream[0]["x"] = 1
    asyncio.run(s.submit(WL.texts[150:300]))            # 150 < 4 x 150: gate not passed
    assert "decision" not in s.batches[-1]["diagnostics"]["chunking"]
    assert s.report()["totals"]["stream_risk"]["status"] == "exact"

