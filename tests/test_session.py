"""Tests for BottleSession. Mock LLM only, no servers.

(a) min_batch=0 + fixed votes policy == the manual run/pour chain, bit for bit
(b) crash mid-batch t+1 (after checkpoint t) + resume() == the uninterrupted stream, bit for bit (forecast included)
(c) calibration keys never repeat; a forced reuse (swapped artifact, stale checkpoint) raises before any LLM call
(d) a version-0 artifact= reseeds deterministically with a warning, recorded in the state, never again on resume
(e) coalescing maps answers back to the right submissions in order; flush(force=True); no submission is split;
    batch composition is label-free
(f) report totals == summed ledgers; expected-error totals == sum of per-batch risk bounds; sum of delta
plus: resume refuses cfg-hash / fingerprint mismatch; atomic checkpoints with plain types only; min_certified_size.
"""

import asyncio
import io
import json
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bottlekit.session as S  # noqa: E402
from bottlekit import BottleArtifact, BottleConfig, Bottler, BottleSession, ModelSpec, TaskSpec  # noqa: E402
from bottlekit import min_certified_size  # noqa: E402
from bottlekit.certify import single_look_design, spend, tuned_looks  # noqa: E402
from bottlekit.labeling import sub_seed  # noqa: E402
from bottlekit.session import _norm, cert_streams  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

WL = toy_workload(1400, seed=77)
GOLD = {t[:400]: g for t, g in zip(WL.texts, WL.gold)}


def _client(acc=None):
    return MockClient(GOLD, acc=acc or {"cheap": 0.82, "expert": 0.95})


def _cfg(**kw):
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=11)
    base.update(kw)
    return BottleConfig(**base)


def _subs(sizes, start=0):
    out, o = [], start
    for n in sizes:
        out.append(WL.texts[o:o + n])
        o += n
    return out


def _emb(texts, d=8):
    rows = []
    for t in texts:
        r = np.random.default_rng(abs(hash(t)) % (2 ** 32))
        v = r.normal(size=d)
        rows.append(v / np.linalg.norm(v))
    return np.asarray(rows, np.float32)


def _j(o):
    return json.dumps(_norm(o), sort_keys=True)


def _mem(art):
    return json.dumps({"v": art.version, "texts": art.texts, "labels": art.labels, "src": list(art.src),
                       "fp": art.fingerprint, "emb": None if art.emb is None else art.emb.tobytes().hex(),
                       "eid": art.embedder_id, "forecast": _norm(art.forecast)}, sort_keys=True)


def _calls(client) -> int:
    return sum(d["calls"] for d in client.ledger.rows.values())


def _batch_view(rec):
    return json.dumps({k: rec[k] for k in ("answers", "provenance", "certificate", "ledger", "kind", "N",
                                           "submissions", "streams")}, sort_keys=True)


# ----------------------------------------------------------------------------------------------- sizing
def test_min_certified_size_is_label_free_and_matches_the_harness_design():
    for eps, expect in ((0.05, 355), (0.1, 190)):
        N = min_certified_size(eps, 0.1)
        assert N == expect

        def share(n_):
            looks, sp = single_look_design(n_, eps, 1.25)
            return tuned_looks(list(looks), spend(list(looks), sp), n_, eps, 0.1, "rectified")[-1] / n_
        assert share(N) <= 0.30
        start = int(4 * 1.25 ** 2 / eps)
        assert all(share(n_) > 0.30 for n_ in range(start, N))      # smallest N in the unclipped regime
    assert min_certified_size(0.05, 0.1, tune=False) == 347          # (1.25/0.3)^2/0.05 rounded up, untuned
    with pytest.raises(ValueError):
        min_certified_size(0.05, 0.1, tax=0.6)
    s = BottleSession(_client(), SPEC, _cfg())
    assert s.min_batch == 355 and s.min_batch_spec == "auto"


# ----------------------------------------------------------------------------------------------- (a)
@pytest.mark.parametrize("votes", [False, True, "auto"])
@pytest.mark.parametrize("dense", [False, True])
def test_a_min_batch0_equals_the_manual_run_pour_chain(votes, dense):
    subs = _subs([90, 70, 110, 80, 100])
    cfg = _cfg(seed=3)
    # manual chain
    c1 = _client()
    b = Bottler(c1, SPEC, cfg)
    manual, art = [], None
    for t in subs:
        e = _emb(t) if dense else None
        r = asyncio.run(b.run(t, emb=e)) if art is None else asyncio.run(b.pour(art, t, votes=votes, emb=e))
        manual.append(r)
        art = r.artifact
    # session
    c2 = _client()
    s = BottleSession(c2, SPEC, cfg, min_batch=0, votes=votes)
    hs = [asyncio.run(s.submit(t, emb=_emb(t) if dense else None)) for t in subs]
    assert all(h.done for h in hs) and len(s.batches) == len(subs)
    for h, r, rec in zip(hs, manual, s.batches):
        assert h.answers == r.answers and h.provenance == r.provenance
        assert _j(rec["certificate"]) == _j(r.certificate)
        assert _j(rec["ledger"]) == _j(r.ledger)
        assert _j(rec["answers"]) == _j(r.answers)
        assert h.certificate is rec["certificate"]
    assert _mem(s.artifact) == _mem(art)
    kinds = [rec["kind"] for rec in s.batches]
    assert kinds[0] == "run"
    if votes is True:
        assert set(kinds[1:]) == {"pour_warm"}
    if votes is False:
        assert set(kinds[1:]) == {"pour_cold"}


# ----------------------------------------------------------------------------------------------- (b)
class CrashClient(MockClient):
    """Raises on the crash_at-th expert call (once): an injected crash in the middle of a batch."""

    def __init__(self, *a, crash_at=None, **kw):
        super().__init__(*a, **kw)
        self.crash_at, self.n_exp = crash_at, 0

    async def chat(self, alias, messages, **kw):
        if alias == "expert":
            self.n_exp += 1
            if self.crash_at is not None and self.n_exp == self.crash_at:
                raise RuntimeError("injected crash")
        return await super().chat(alias, messages, **kw)


@pytest.mark.parametrize("dense", [False, True])
def test_b_crash_mid_batch_then_resume_equals_the_uninterrupted_stream(tmp_path, dense):
    subs = _subs([60] * 11)
    cfg = _cfg(seed=21)
    emb = (lambda t: _emb(t)) if dense else (lambda t: None)
    # uninterrupted
    cu = _client()
    u = BottleSession(cu, SPEC, cfg, state_path=tmp_path / "u.json", min_batch=150, votes="auto")
    for t in subs:
        asyncio.run(u.submit(t, emb=emb(t)))
    asyncio.run(u.flush(force=True))
    assert len(u.batches) == 4 and [r["N"] for r in u.batches] == [180, 180, 180, 120]
    exp_per_batch = [sum(d["calls"] for k, d in r["ledger"].items() if k.startswith("expert")) for r in u.batches]
    t_ck = 1                                                # crash in batch t_ck + 1 = 2, after checkpoint of batch 1
    crash_at = sum(exp_per_batch[:t_ck + 1]) + 7
    # crashing run
    cc = CrashClient(GOLD, acc={"cheap": 0.82, "expert": 0.95}, crash_at=crash_at)
    p = tmp_path / "c.json"
    c = BottleSession(cc, SPEC, cfg, state_path=p, min_batch=150, votes="auto")
    crashed_at_sub = None
    for j, t in enumerate(subs):
        try:
            asyncio.run(c.submit(t, emb=emb(t)))
        except RuntimeError as e:
            assert "injected crash" in str(e)
            crashed_at_sub = j
            break
    assert crashed_at_sub is not None and len(c.batches) == t_ck + 1
    del c
    # resume with a fresh client; finish the interrupted batch, then the rest of the stream
    cr = _client()
    r = BottleSession.resume(p, cr, SPEC, cfg)
    assert len(r.batches) == t_ck + 1 and r._inflight is not None and r._inflight["batch"] == t_ck + 1
    rows = asyncio.run(r.flush())
    assert len(rows) == 1 and rows[0]["batch"] == t_ck + 1
    for t in subs[crashed_at_sub + 1:]:
        asyncio.run(r.submit(t, emb=emb(t)))
    asyncio.run(r.flush(force=True))
    assert len(r.batches) == len(u.batches)
    for a, b in zip(u.batches, r.batches):
        assert _batch_view(a) == _batch_view(b)
    assert _mem(r.artifact) == _mem(u.artifact)              # bottle memory AND its forecast
    assert r.artifact.forecast is not None
    assert [h.answers for h in u.submissions] == [h.answers for h in r.submissions]
    assert _j(u.report()) == _j(r.report())
    # the resumed client continues the checkpointed cumulative ledger and paid only for the batches after the
    # checkpoint (the crashed attempt's calls were never recorded): no checkpointed batch was re-run
    assert _j(cr.ledger.rows) == _j(cu.ledger.rows)
    assert _calls(cr) == _calls(cu)
    # a resume after the stream finished re-runs nothing
    r2 = BottleSession.resume(p, _client(), SPEC, cfg)
    n0 = _calls(r2.client)
    assert asyncio.run(r2.flush(force=True)) == [] and _calls(r2.client) == n0 == _calls(cu)


def test_b_retry_after_a_failure_keeps_the_frozen_composition():
    """An exception mid-batch freezes that batch's composition: later submissions do not join it, and the retry uses
    the same calibration key (identical draw)."""
    subs = _subs([100, 100, 100])
    cfg = _cfg(seed=5)
    cc = CrashClient(GOLD, acc={"cheap": 0.82, "expert": 0.95}, crash_at=3)
    s = BottleSession(cc, SPEC, cfg, min_batch=150, votes=False)
    asyncio.run(s.submit(subs[0]))
    with pytest.raises(RuntimeError, match="injected crash"):
        asyncio.run(s.submit(subs[1]))
    assert s._inflight["ids"] == [0, 1]
    h3 = asyncio.run(s.submit(subs[2]))      # retries the frozen batch [0, 1]; 100 pending < 150 stays pending
    assert [r["submissions"] for r in s.batches] == [[0, 1]] and not h3.done
    assert s.batches[0]["streams"] == [[5, 1]]
    ref = BottleSession(_client(), SPEC, cfg, min_batch=150, votes=False)
    for t in subs[:2]:
        asyncio.run(ref.submit(t))
    a, b = ref.batches[0], s.batches[0]
    for k in ("answers", "provenance", "certificate", "kind", "N", "submissions", "streams"):
        assert _j(a[k]) == _j(b[k])
    for ph in a["ledger"]:                       # same calls and tokens (USD only up to cumulative float rounding)
        assert {k: v for k, v in a["ledger"][ph].items() if k != "usd"} == \
               {k: v for k, v in b["ledger"][ph].items() if k != "usd"}


# ----------------------------------------------------------------------------------------------- (c)
def test_c_calibration_keys_never_repeat_over_a_stream():
    cfg = _cfg(seed=8, memo=True)
    s = BottleSession(_client(), SPEC, cfg, min_batch=0, votes="auto")
    texts = WL.texts
    subs = [texts[0:120], texts[120:220], texts[60:160] + texts[220:260], texts[260:330], texts[0:50] * 2,
            texts[330:400], texts[400:470]]
    for t in subs:
        asyncio.run(s.submit(t))
    keys = [k for rec in s.batches for k in cert_streams(rec["certificate"])]
    assert len(keys) == len(set(keys)) and len({sub_seed(k[0], *k[1:]) for k in keys}) == len(keys)
    assert keys[0] == (8, 1) and all(len(k) == 4 for k in keys[1:])
    assert sorted(keys) == sorted(s.used_streams)
    for rec in s.batches:                       # the guard uses the keys the harness actually recorded
        assert [tuple(k) for k in rec["streams"]] == cert_streams(rec["certificate"])


def test_c_cert_streams_reads_chunked_certificates():
    """A chunked pour (if right-sized-pours merges) records one key per chunk under certificate['chunks']."""
    plain = {"cal_stream": [3, 1, 2, 400]}
    assert cert_streams(plain) == [(3, 1, 2, 400)]
    chunked = {"chunked": True, "chunks": [{"cal_stream": [3, 1, 2, 400]}, {"cal_stream": [3, 1, 3, 800]}]}
    assert cert_streams(chunked) == [(3, 1, 2, 400), (3, 1, 3, 800)]
    top_too = {"cal_stream": [3, 1, 2, 400], **chunked}
    assert cert_streams(top_too) == [(3, 1, 2, 400), (3, 1, 3, 800)]
    assert cert_streams({"no_calibration_needed": True}) == []
    assert S._expected_errors({"chunks": [{"risk": {"expected_eps_bound": 0.03}, "N": 400},
                                          {"risk": {"expected_eps_bound": 0.02}, "N": 600}]}) == pytest.approx(24.0)


def test_c_swapped_artifact_raises_before_any_llm_call():
    cfg = _cfg(seed=9)
    client = _client()
    s = BottleSession(client, SPEC, cfg, min_batch=0, votes=False)
    subs = _subs([80, 80, 80, 80])
    asyncio.run(s.submit(subs[0]))
    old = s.artifact                             # bottle after batch 0
    asyncio.run(s.submit(subs[1]))
    asyncio.run(s.submit(subs[2]))
    s.artifact = old                             # manual swap back to an artifact whose next stream is used
    n0 = _calls(client)
    with pytest.raises(ValueError, match="lineage guard"):
        asyncio.run(s.submit(subs[3]))
    assert _calls(client) == n0                  # nothing was called: no vote, no expert
    # a hand-copied bottle of the same lineage point is caught too (key, not object identity)
    s.artifact = BottleArtifact(SPEC, old.texts, old.labels, version=old.version, src=old.src,
                                fingerprint=old.fingerprint)
    with pytest.raises(ValueError, match="lineage guard"):
        asyncio.run(s.flush(force=True))
    assert _calls(client) == n0


def test_c_prior_streams_and_prediction_mismatch():
    cfg = _cfg(seed=4)
    s = BottleSession(_client(), SPEC, cfg, min_batch=0, votes=False, prior_streams=[[4, 1]])
    with pytest.raises(ValueError, match="outside this session"):
        asyncio.run(s.submit(_subs([60])[0]))
    # a harness that records another key than predicted is a bug: fail loudly
    s = BottleSession(_client(), SPEC, cfg, min_batch=0, votes=False)
    asyncio.run(s.submit(_subs([80])[0]))
    orig = s.bottler.pour

    async def bad_pour(*a, **kw):
        r = await orig(*a, **kw)
        r.certificate["cal_stream"] = [4, 1, 99, 99]
        return r
    s.bottler.pour = bad_pour
    with pytest.raises(RuntimeError, match="lineage bug"):
        asyncio.run(s.submit(_subs([60], start=80)[0]))


def test_c_stale_checkpoint_is_refused(tmp_path):
    cfg = _cfg(seed=12)
    p = tmp_path / "s.json"
    s = BottleSession(_client(), SPEC, cfg, state_path=p, min_batch=0, votes=False)
    subs = _subs([70, 70, 70, 70])
    asyncio.run(s.submit(subs[0]))
    asyncio.run(s.submit(subs[1]))
    backup = tmp_path / "backup.json"
    shutil.copy(p, backup)                       # checkpoint after batch 1
    asyncio.run(s.submit(subs[2]))
    asyncio.run(s.submit(subs[3]))
    # resuming the stale copy (elsewhere, or copied back over the live state) is refused before any call
    with pytest.raises(ValueError, match="stale checkpoint"):
        BottleSession.resume(backup, _client(), SPEC, cfg)
    shutil.copy(backup, p)
    with pytest.raises(ValueError, match="stale checkpoint"):
        BottleSession.resume(p, _client(), SPEC, cfg)
    # without its journal the stale state still cannot reuse a key the state itself records; a NEW session at the
    # same path is refused while the journal exists
    with pytest.raises(ValueError, match="exists"):
        BottleSession(_client(), SPEC, cfg, state_path=p)


# ----------------------------------------------------------------------------------------------- (d)
def _v0_bottle():
    b = Bottler(_client(), SPEC, _cfg(seed=11))
    r = asyncio.run(b.run(WL.texts[:200]))
    return BottleArtifact(SPEC, WL.texts[:200], r.answers)           # hand-built: version 0, no lineage


def test_d_version0_artifact_reseeds_deterministically_once(tmp_path):
    cfg = _cfg(seed=11)
    art = _v0_bottle()
    p = tmp_path / "v0.json"
    with pytest.warns(UserWarning, match="version 0"):
        s = BottleSession(_client(), SPEC, cfg, state_path=p, artifact=art, min_batch=0, votes=False)
    with pytest.warns(UserWarning, match="version 0"):
        s_twin = BottleSession(_client(), SPEC, cfg, artifact=art, min_batch=0, votes=False)
    assert s.seed == s_twin.seed != cfg.seed and (s.seed & 0xFFFFFFFF) != (cfg.seed & 0xFFFFFFFF)
    assert s.cfg.seed == s.seed and s.cfg_user.seed == cfg.seed
    st = json.loads(p.read_text(encoding="utf-8"))
    assert st["seed"]["effective"] == s.seed and st["seed"]["reseed"]["from"] == cfg.seed
    assert st["seed"]["reseed"]["to"] == s.seed
    subs = _subs([80, 80, 80], start=200)
    asyncio.run(s.submit(subs[0]))
    assert s.batches[0]["streams"][0] == [s.seed, 1, 1, len(art.texts)]
    assert s.batches[0]["kind"] == "pour_cold"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = BottleSession.resume(p, _client(), SPEC, cfg)
    assert not [x for x in w if "version 0" in str(x.message)]          # no second reseed
    assert r.seed == s.seed and r._reseed == st["seed"]["reseed"]
    for t in subs[1:]:
        asyncio.run(r.submit(t))
    for t in subs:
        asyncio.run(s_twin.submit(t))
    assert [x["answers"] for x in r.batches] == [x["answers"] for x in s_twin.batches]
    assert [x["streams"] for x in r.batches] == [x["streams"] for x in s_twin.batches]
    # opting out keeps the user's seed (the user takes responsibility for the lineage)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        s3 = BottleSession(_client(), SPEC, cfg, artifact=art, reseed_v0=False)
    assert s3.seed == cfg.seed and not [x for x in w if "version 0" in str(x.message)]
    # a version > 0 artifact is not reseeded
    b1 = art.updated(WL.texts[300:310], ["alpha"] * 10)
    assert BottleSession(_client(), SPEC, cfg, artifact=b1).seed == cfg.seed


# ----------------------------------------------------------------------------------------------- (e)
def test_e_coalescing_maps_answers_back_in_order_and_never_splits():
    cfg = _cfg(seed=6)
    sizes = [30, 50, 0, 70, 20, 90, 10]
    subs = _subs(sizes)
    s = BottleSession(_client(), SPEC, cfg, min_batch=100, votes=False)
    hs = []
    for t in subs:
        hs.append(asyncio.run(s.submit(t)))
    assert [r["submissions"] for r in s.batches] == [[0, 1, 3], [4, 5]]
    assert hs[2].done and hs[2].answers == [] and hs[2].batch is None          # empty submission
    assert not hs[6].done and hs[6].answers is None and s.pending_items == 10
    assert asyncio.run(s.flush()) == []                                        # 10 < min_batch: nothing
    rows = asyncio.run(s.flush(force=True))                                    # partial buffer certified
    assert len(rows) == 1 and rows[0]["N"] == 10 and hs[6].done and s.pending_items == 0
    # reference: the same coalesced batches through the manual chain
    b = Bottler(_client(), SPEC, cfg)
    groups = [[0, 1, 3], [4, 5], [6]]
    art, ref = None, []
    for g in groups:
        texts = [x for i in g for x in subs[i]]
        r = asyncio.run(b.run(texts)) if art is None else asyncio.run(b.pour(art, texts, votes=False))
        ref.append(r)
        art = r.artifact
    for g, r, rec in zip(groups, ref, s.batches):
        o = 0
        for i in g:
            h = hs[i]
            assert h.batch == rec["batch"] and h.offset == o                    # one batch, contiguous slice
            assert h.answers == r.answers[o:o + sizes[i]] and h.provenance == r.provenance[o:o + sizes[i]]
            assert h.batch_N == len(r.answers) and "no guarantee of its own" in h.guarantee_unit
            o += sizes[i]
        assert o == len(r.answers)
    # every handle is in exactly one batch (or empty)
    seen = [h.id for rec in s.batches for h in hs if h.batch == rec["batch"]]
    assert sorted(seen) == [0, 1, 3, 4, 5, 6]


def test_e_batch_composition_is_label_free():
    """Flush times and compositions depend only on submission sizes and min_batch: two streams with very different
    expert/cheap behaviour (hence labels, scores, certificates and forecasts) coalesce identically."""
    sizes = [40, 70, 25, 90, 60, 15, 80, 45]
    subs = _subs(sizes)
    comps = []
    for acc in ({"cheap": 0.95, "expert": 0.99}, {"cheap": 0.4, "expert": 0.7}):
        s = BottleSession(_client(acc), SPEC, _cfg(seed=2), min_batch=120, votes="auto")
        for t in subs:
            asyncio.run(s.submit(t))
        asyncio.run(s.flush(force=True))
        comps.append([r["submissions"] for r in s.batches])
    assert comps[0] == comps[1] == [[0, 1, 2], [3, 4], [5, 6, 7]]


def test_e_mixed_embeddings_are_refused():
    s = BottleSession(_client(), SPEC, _cfg(), min_batch=500)
    t = _subs([20, 20])
    asyncio.run(s.submit(t[0], emb=_emb(t[0])))
    with pytest.raises(ValueError, match="emb"):
        asyncio.run(s.submit(t[1]))


# ----------------------------------------------------------------------------------------------- (f)
@pytest.mark.parametrize("extra", [{}, {"expert_budget": 15}])
def test_f_report_totals_equal_summed_ledgers_and_risk_bounds(extra):
    cfg = _cfg(seed=13, **extra)
    s = BottleSession(_client(), SPEC, cfg, min_batch=120, votes="auto")
    for t in _subs([70] * 7):
        asyncio.run(s.submit(t))
    asyncio.run(s.flush(force=True))
    rep = s.report()
    tot, rows = rep["totals"], rep["rows"]
    led_usd = sum(d["usd"] for rec in s.batches for d in rec["ledger"].values())
    assert tot["usd"] == pytest.approx(led_usd, rel=1e-12)
    assert tot["usd"] == pytest.approx(s.client.ledger.total_usd(), rel=1e-12)
    for ph in {k for rec in s.batches for k in rec["ledger"]}:
        assert tot["usd_by_phase"][ph] == pytest.approx(sum(rec["ledger"].get(ph, {}).get("usd", 0.0)
                                                            for rec in s.batches), rel=1e-12)
    assert tot["expert_calls"] == sum(d["calls"] for rec in s.batches for k, d in rec["ledger"].items()
                                      if k.startswith("expert"))
    exp = []
    for rec in s.batches:
        risk = rec["certificate"]["risk"]
        exp.append(None if risk.get("expected_eps_bound") is None else risk["expected_eps_bound"] * rec["N"])
    known = [e for e in exp if e is not None]
    assert tot["expected_errors_bound"] == pytest.approx(sum(known), rel=1e-12)
    for r, e in zip(rows, exp):
        assert (r["expected_errors_bound"] is None) if e is None else r["expected_errors_bound"] == pytest.approx(e)
    assert tot["batches_without_expected_bound"] == sum(e is None for e in exp)
    assert tot["expected_failing_batches_bound"] == pytest.approx(0.1 * len(s.batches))
    assert tot["items"] == 490 and tot["bottled_items"] == sum(r["bottled_items"] for r in rows)
    ex_unit = tot["expert_usd_per_call"]
    assert tot["cost_frac"] == pytest.approx(tot["usd"] / (ex_unit * 490))
    st = rep["statement"]
    assert "No statement is made about the probability that no batch fails" in st
    assert "Certified unit = each flushed batch" in st
    if extra:
        assert any(rec["certificate"].get("budget_capped") for rec in s.batches)
        assert tot["batches_without_expected_bound"] >= 1


# ----------------------------------------------------------------------------------------------- resume / checkpoints
def test_resume_refuses_cfg_or_fingerprint_mismatch(tmp_path):
    cfg = _cfg(seed=1)
    p = tmp_path / "s.json"
    s = BottleSession(_client(), SPEC, cfg, state_path=p, min_batch=0, votes=False)
    asyncio.run(s.submit(_subs([60])[0]))
    with pytest.raises(ValueError, match="cfg hash"):
        BottleSession.resume(p, _client(), SPEC, _cfg(seed=1, eps=0.1))
    with pytest.raises(ValueError, match="cfg hash"):
        BottleSession.resume(p, _client(), SPEC, _cfg(seed=2))
    spec2 = TaskSpec("toy", "Classify the item differently.", SPEC.labels)
    with pytest.raises(ValueError, match="fingerprint"):
        BottleSession.resume(p, _client(), spec2, cfg)
    c = _client()
    c.models["expert"] = ModelSpec("expert", ["another-expert"], price_in=2.0, price_out=8.0)
    with pytest.raises(ValueError, match="fingerprint"):
        BottleSession.resume(p, c, SPEC, cfg)
    BottleSession.resume(p, _client(), SPEC, cfg)                  # the matching one resumes


class _NoGlobals(pickle.Unpickler):
    def find_class(self, module, name):
        raise AssertionError(f"checkpoint bottle pickles a class: {module}.{name}")


def test_checkpoints_are_atomic_and_plain(tmp_path, monkeypatch):
    cfg = _cfg(seed=3)
    p = tmp_path / "s.json"
    calls = []
    real_replace = S.os.replace

    def spy(src, dst):
        calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)
    monkeypatch.setattr(S.os, "replace", spy)
    s = BottleSession(_client(), SPEC, cfg, state_path=p, min_batch=100, votes=False)
    t = _subs([60, 60, 60])
    asyncio.run(s.submit(t[0], emb=None))
    asyncio.run(s.submit(t[1]))
    assert calls and all(src.parent == dst.parent == p.parent and dst == p and src != p for src, dst in calls)
    st = json.loads(p.read_text(encoding="utf-8"))                  # plain JSON
    assert st["format"] == "bottlekit.session" and st["artifact"] is not None
    blob = __import__("base64").b64decode(st["artifact"]["b64"])
    d = _NoGlobals(io.BytesIO(blob)).load()                          # artifact.save format: plain types only
    assert set(d) >= {"texts", "labels", "version", "src"} and b"sklearn" not in blob
    # a crash while writing leaves the previous checkpoint intact and no temp file behind
    before = p.read_bytes()

    def boom(fd):
        raise OSError("disk full")
    monkeypatch.setattr(S.os, "fsync", boom)
    with pytest.raises(OSError):
        asyncio.run(s.submit(t[2]))
    assert p.read_bytes() == before
    assert not [x for x in tmp_path.iterdir() if ".tmp-" in x.name or "bottle-tmp" in x.name]


def test_session_requires_export_and_valid_args():
    with pytest.raises(ValueError, match="export"):
        BottleSession(_client(), SPEC, _cfg(export=False))
    with pytest.raises(ValueError, match="votes"):
        BottleSession(_client(), SPEC, _cfg(), votes="sometimes")
    with pytest.raises(ValueError, match="min_batch"):
        BottleSession(_client(), SPEC, _cfg(), min_batch=-1)


def test_submit_error_after_checkpoint_carries_the_queued_handle():
    """An error raised while certifying, i.e. after the submission is durable, carries the handle
    as exc.submission; flush() then retries the frozen batch and fills that same handle (no resubmission)."""
    subs = _subs([100, 100])
    cc = CrashClient(GOLD, acc={"cheap": 0.82, "expert": 0.95}, crash_at=3)
    s = BottleSession(cc, SPEC, _cfg(seed=6), min_batch=150, votes=False)
    asyncio.run(s.submit(subs[0]))
    with pytest.raises(RuntimeError, match="injected crash") as ei:
        asyncio.run(s.submit(subs[1]))
    h = ei.value.submission
    assert h.id == 1 and not h.done and s._inflight["ids"] == [0, 1]
    asyncio.run(s.flush())
    assert h.done and len(h.answers) == 100 and len(s.submissions) == 2
