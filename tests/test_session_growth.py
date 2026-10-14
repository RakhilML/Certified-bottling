"""Tests for growing BottleSession flushes. Mock LLM only.

batch_growth=g makes the flush target T = min(max_batch, max(min_batch, ceil(g * M_plan))), with the PLANNED memory
M_plan = |initial artifact| + the sizes of committed batches. Checked here:
  - g = 0 is the v0.8 session (targets = min_batch, outputs identical to a session built without the new arguments,
    a checkpoint without growth fields resumes as g = 0)
  - the schedule is a function of submission sizes only: identical under different expert banks, unusable expert
    answers (smaller actual memory), votes policies; equal to a pure-Python simulation of the rule
  - crash + resume reproduces the schedule; M_plan is rebuilt from batch sizes, never from the bottle
  - the max_batch cap (and 'auto' = ceil((design_c/0.1)^2/eps)); argument validation
  - lineage guard / checkpoints unchanged; stream ledger 'exact' under growth (votes=False); forced vs target
  - min_batch = 0 ignores growth; g > 1 can pass the 'cold' chunk gate and the v0.8 withdrawal rules still apply
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottlekit import BottleArtifact, BottleConfig, Bottler, BottleSession  # noqa: E402
from bottlekit.certify import StreamLaw  # noqa: E402
from bottlekit.session import _norm, cert_streams, growth_target, max_batch_auto  # noqa: E402
from test_bottlekit import SPEC, MockClient  # noqa: E402
from test_session import GOLD, WL, CrashClient, _batch_view, _calls, _cfg, _client, _j, _mem, _subs  # noqa: E402


def simulate(sizes, min_batch, g, max_batch, m_init, force_end=True):
    """Pure-Python reference of the flush rule: [(submission ids, target, trigger)]."""
    out, pend, M = [], [], m_init
    for i, n in enumerate(sizes):
        if n:
            pend.append((i, n))
        T = growth_target(min_batch, g, max_batch, M)
        tot = sum(x for _, x in pend)
        if tot > 0 and tot >= T:
            out.append(([j for j, _ in pend], T, "auto"))
            M += tot
            pend = []
    if force_end and pend:
        T = growth_target(min_batch, g, max_batch, M)
        tot = sum(x for _, x in pend)
        out.append(([j for j, _ in pend], T, "auto" if tot >= T else "forced"))
    return out


def _stream(client, cfg, sizes, start=0, subs=None, **kw):
    s = BottleSession(client, SPEC, cfg, **kw)
    for t in (subs if subs is not None else _subs(sizes, start=start)):
        asyncio.run(s.submit(t))
    asyncio.run(s.flush(force=True))
    return s


def _schedule(s):
    return [(r["submissions"], r["flush_target"], e.get("t")) for r, e in zip(s.batches, s._stream)]


class UnusableExpert(MockClient):
    """The expert answers about `frac` of items with an unparseable reply (another expert bank; F > 0)."""

    def __init__(self, *a, frac=0.12, **kw):
        super().__init__(*a, **kw)
        self.frac = frac

    def _answer(self, alias, text, salt, pos=None, k=None):
        if alias == "expert" and int(hashlib.md5(text.encode()).hexdigest()[:6], 16) / 16 ** 6 < self.frac:
            return "I am not sure."
        return super()._answer(alias, text, salt, pos=pos, k=k)


# ----------------------------------------------------------------------------------------------- helpers
def test_growth_target_and_cap_values():
    assert max_batch_auto(0.05, 1.25) == 3125 and max_batch_auto(0.1, 1.25) == 1563
    assert max_batch_auto(0.02, 1.25) == 7813
    assert max_batch_auto(0.05, design_c=1.0) == 2000
    assert max_batch_auto(0.05, 1.25, share=0.2) == 782             # max_batch_share: a named session argument
    assert growth_target(355, 0.0, 3125, 10_000) == 355            # g = 0: v0.8
    assert growth_target(0, 0.5, 3125, 10_000) == 0                 # min_batch = 0 ignores growth
    assert growth_target(355, 0.5, 3125, 400) == 355
    assert growth_target(355, 0.5, 3125, 1200) == 600
    assert growth_target(355, 0.5, 3125, 10_000) == 3125            # capped
    assert growth_target(355, 0.5, None, 10_000) == 5000            # no cap
    assert growth_target(300, 0.1, None, 3000) == 300               # 0.1 * 3000 is 300.00000000000006: not 301
    assert growth_target(100, 0.1, None, 3010) == 301
    assert growth_target(500, 0.5, 400, 10_000) == 500              # min_batch always wins over the cap
    s = BottleSession(_client(), SPEC, _cfg())
    assert s.batch_growth == 0.0 and s.max_batch == 3125 and s.flush_target() == s.min_batch == 355
    s = BottleSession(_client(), SPEC, _cfg(eps=0.1), batch_growth=0.5)
    assert s.max_batch == 1563 and s.min_batch == 190


def test_argument_validation():
    for g in (-0.1, float("nan"), float("inf"), True, "0.5", None):
        with pytest.raises(ValueError, match="batch_growth"):
            BottleSession(_client(), SPEC, _cfg(), batch_growth=g)
    with pytest.raises(ValueError, match="max_batch"):
        BottleSession(_client(), SPEC, _cfg(), min_batch=200, batch_growth=0.5, max_batch=150)
    for mb in ("big", 1.5, True):
        with pytest.raises(ValueError, match="max_batch"):
            BottleSession(_client(), SPEC, _cfg(), min_batch=100, batch_growth=0.5, max_batch=mb)
    s = BottleSession(_client(), SPEC, _cfg(), min_batch=5000, batch_growth=0.5)   # 'auto' never below min_batch
    assert s.max_batch == 5000
    s = BottleSession(_client(), SPEC, _cfg(), min_batch=100, batch_growth=2, max_batch=None)
    assert s.max_batch is None and s.batch_growth == 2.0
    for sh in (0, 1, -0.1, 1.5, True, "0.1", None):
        with pytest.raises(ValueError, match="max_batch_share"):
            BottleSession(_client(), SPEC, _cfg(), batch_growth=0.5, max_batch_share=sh)


def test_max_batch_share_sets_the_auto_cap_and_is_persisted(tmp_path):
    s = BottleSession(_client(), SPEC, _cfg(), batch_growth=0.5, max_batch_share=0.2)
    assert s.max_batch == max_batch_auto(0.05, 1.25, 0.2) == 782 and s.max_batch_share == 0.2
    assert BottleSession(_client(), SPEC, _cfg(), batch_growth=0.5, max_batch=900, max_batch_share=0.2).max_batch == 900
    st = tmp_path / "s.json"
    s = BottleSession(_client(), SPEC, _cfg(), state_path=st, batch_growth=0.5, max_batch_share=0.2)
    asyncio.run(s.submit(["a b c"]))
    r = BottleSession.resume(st, _client(), SPEC, _cfg())
    assert r.max_batch == 782 and r.max_batch_share == 0.2 and r.report()["totals"]["max_batch_share"] == 0.2


# ----------------------------------------------------------------------------------------------- g = 0
@pytest.mark.parametrize("votes", [False, "auto"])
def test_g0_is_the_v08_session(tmp_path, votes):
    sizes = [70, 40, 90, 60, 30, 110, 50, 80, 20]
    cfg = _cfg(seed=31)
    a = _stream(_client(), cfg, sizes, min_batch=120, votes=votes, state_path=tmp_path / "a.json")
    b = _stream(_client(), cfg, sizes, min_batch=120, votes=votes, state_path=tmp_path / "b.json",
                batch_growth=0.0, max_batch=None)
    assert [x["submissions"] for x in a.batches] == [[0, 1, 2], [3, 4, 5], [6, 7], [8]]
    for x, y in zip(a.batches, b.batches):
        assert _batch_view(x) == _batch_view(y)
    assert all(x["flush_target"] == 120 for x in a.batches)
    assert [e["t"] for e in a._stream] == ["auto", "auto", "auto", "forced"]
    assert _mem(a.artifact) == _mem(b.artifact)
    ra, rb = a.report(), b.report()
    assert ra["statement"] == rb["statement"] and _j(ra["totals"]["stream_risk"]) == _j(rb["totals"]["stream_risk"])
    assert [r["items_waited"] for r in ra["rows"]] == [130, 140, 80, 0]
    # schedule equals the v0.8 rule (n >= min_batch)
    assert [(c, t, tr) for c, t, tr in simulate(sizes, 120, 0.0, None, 0)] == _schedule(a)


def test_pre_iter7_checkpoint_resumes_as_g0(tmp_path):
    """A checkpoint without the 'growth' field (written by v0.8) resumes with batch_growth = 0."""
    p = tmp_path / "s.json"
    cfg = _cfg(seed=17)
    s = BottleSession(_client(), SPEC, cfg, state_path=p, min_batch=100, batch_growth=0.5, max_batch=None)
    for t in _subs([60, 60]):
        asyncio.run(s.submit(t))
    st = json.loads(p.read_text(encoding="utf-8"))
    assert st["growth"]["batch_growth"] == 0.5 and st["growth"]["m_init"] == 0
    del st["growth"]
    p.write_text(json.dumps(st), encoding="utf-8")
    r = BottleSession.resume(p, _client(), SPEC, cfg)
    assert r.batch_growth == 0.0 and r.max_batch is None and r.flush_target() == 100


# ----------------------------------------------------------------------------------------------- label-free schedule
def test_schedule_is_invariant_to_expert_bank_unusable_answers_duplicates_and_votes():
    """Same submission sizes -> same compositions and targets, whatever the labels, unusable expert answers, votes
    policy or texts. The repeated-texts stream dedupes the bottle (actual memory far below the planned memory), and
    the schedule still follows the planned memory."""
    sizes = [40] * 30
    cfg = _cfg(seed=2)
    kw = dict(min_batch=100, batch_growth=0.5, max_batch=300)
    ref = simulate(sizes, 100, 0.5, 300, 0)
    assert [len(c) * 40 for c, _, _ in ref] == [120, 120, 120, 200, 280, 320, 40]
    dup = _subs([40] * 15) * 2                                 # the same 15 submissions twice
    scheds, mems = [], []
    for client, votes, subs in ((_client({"cheap": 0.95, "expert": 0.99}), False, None),
                                (_client({"cheap": 0.4, "expert": 0.7}), "auto", None),
                                (UnusableExpert(GOLD, acc={"cheap": 0.82, "expert": 0.95}, frac=0.12), False, None),
                                (_client(), True, None),
                                (_client(), False, dup)):
        s = _stream(client, cfg, sizes, subs=subs, votes=votes, **kw)
        scheds.append(_schedule(s))
        mems.append((s.planned_memory, len(s.artifact.texts)))
        assert s.planned_memory == sum(sizes)
    assert all(x == scheds[0] for x in scheds)
    assert scheds[0] == [(c, t, tr) for c, t, tr in ref]
    assert mems[4][1] <= 600 < mems[4][0] == 1200              # deduped bottle; the schedule did not follow it


def test_schedule_never_reads_the_bottle_memory():
    """Shrinking the bottle's actual memory between batches (as dropped labels would) does not move the schedule."""
    sizes = [50] * 16
    cfg = _cfg(seed=4)
    a = BottleSession(_client(), SPEC, cfg, min_batch=100, batch_growth=0.5, max_batch=None, votes=False)
    for t in _subs(sizes[:4]):
        asyncio.run(a.submit(t))
    assert len(a.batches) == 2 and a.planned_memory == 200
    T_before = a.flush_target()
    art = a.artifact
    a.artifact = BottleArtifact(SPEC, art.texts[:20], art.labels[:20], version=art.version, src=art.src[:20],
                                fingerprint=art.fingerprint)          # an artificially tiny bottle
    assert a.flush_target() == T_before == 100 and a.planned_memory == 200


def test_initial_artifact_sets_m_init_and_growth_schedule():
    cfg = _cfg(seed=19)
    b = Bottler(_client(), SPEC, cfg)
    r0 = asyncio.run(b.run(WL.texts[:300]))
    m0 = len(r0.artifact.texts)
    s = _stream(_client(), cfg, [50] * 20, start=300, artifact=r0.artifact, min_batch=100, batch_growth=0.5,
                max_batch=None, votes=False)
    assert s._m_init == m0 and s.planned_memory == m0 + 1000
    assert _schedule(s) == [(c, t, tr) for c, t, tr in simulate([50] * 20, 100, 0.5, None, m0)]
    assert [x["kind"] for x in s.batches] == ["pour_cold"] * len(s.batches)
    sizes = [x["N"] for x in s.batches]
    assert sizes[0] >= 150 and sizes[1] > sizes[0] and len(sizes) < 10     # grows instead of 10 x 100


# ----------------------------------------------------------------------------------------------- resume
def test_crash_and_resume_reproduce_the_schedule(tmp_path):
    sizes = [40] * 25
    cfg = _cfg(seed=21)
    kw = dict(min_batch=100, batch_growth=0.5, max_batch=400, votes="auto")
    u = BottleSession(_client(), SPEC, cfg, state_path=tmp_path / "u.json", **kw)
    for t in _subs(sizes):
        asyncio.run(u.submit(t))
    asyncio.run(u.flush(force=True))
    assert len(u.batches) >= 4
    exp_per_batch = [sum(d["calls"] for k, d in r["ledger"].items() if k.startswith("expert")) for r in u.batches]
    crash_at = sum(exp_per_batch[:3]) + 5                       # inside batch 3 (a grown batch)
    p = tmp_path / "c.json"
    c = BottleSession(CrashClient(GOLD, acc={"cheap": 0.82, "expert": 0.95}, crash_at=crash_at), SPEC, cfg,
                      state_path=p, **kw)
    stop = None
    for j, t in enumerate(_subs(sizes)):
        try:
            asyncio.run(c.submit(t))
        except RuntimeError as e:
            assert "injected crash" in str(e)
            stop = j
            break
    assert stop is not None and len(c.batches) == 3
    target_frozen = c._inflight["flush_target"]
    del c
    r = BottleSession.resume(p, _client(), SPEC, cfg)
    assert (r.batch_growth, r.max_batch, r._m_init) == (0.5, 400, 0)
    assert r.planned_memory == sum(x["N"] for x in u.batches[:3])
    asyncio.run(r.flush())
    assert r.batches[3]["flush_target"] == target_frozen == u.batches[3]["flush_target"]
    for t in _subs(sizes)[stop + 1:]:
        asyncio.run(r.submit(t))
    asyncio.run(r.flush(force=True))
    assert _schedule(r) == _schedule(u)
    for a, b in zip(u.batches, r.batches):
        assert _batch_view(a) == _batch_view(b) and a["flush_target"] == b["flush_target"]
    assert _mem(r.artifact) == _mem(u.artifact)
    assert _j(u.report()) == _j(r.report())


def test_resume_rebuilds_planned_memory_from_batch_sizes_not_the_bottle(tmp_path):
    p = tmp_path / "s.json"
    cfg = _cfg(seed=23)
    s = BottleSession(_client(), SPEC, cfg, state_path=p, min_batch=100, batch_growth=1.0, max_batch=None,
                      votes=False)
    for t in _subs([60] * 3) * 2:                             # repeated texts: the bottle dedupes them
        asyncio.run(s.submit(t))
    assert [x["N"] for x in s.batches] == [120, 120] and s.pending_items == 120
    assert len(s.artifact.texts) == 180 < s.planned_memory == 240
    r = BottleSession.resume(p, _client(), SPEC, cfg)
    assert r.planned_memory == s.planned_memory == 240
    assert r.flush_target() == s.flush_target() == 240            # not 180 (the actual memory)


def test_max_batch_cap_holds():
    sizes = [30] * 40
    s = _stream(_client(), _cfg(seed=8), sizes, min_batch=60, batch_growth=1.0, max_batch=150, votes=False)
    for rec, e in zip(s.batches, s._stream):
        assert rec["flush_target"] <= 150
        if e["t"] == "auto":
            assert rec["N"] < 150 + 30                       # flushed as soon as the buffer reached the target
    assert max(x["N"] for x in s.batches) == 150
    rep = s.report()
    assert rep["totals"]["max_batch"] == 150 and rep["totals"]["batch_growth"] == 1.0
    fs = rep["totals"]["flush_schedule"]
    assert fs["batch_sizes"] == [x["N"] for x in s.batches] and fs["flush_targets"] == [x["flush_target"]
                                                                                          for x in s.batches]
    assert fs["max_items_waited"] == max(r["items_waited"] for r in rep["rows"]) <= 150 - 1


def test_min_batch0_ignores_growth():
    subs = _subs([90, 70, 110, 80])
    cfg = _cfg(seed=3)
    a = BottleSession(_client(), SPEC, cfg, min_batch=0, votes=False)
    b = BottleSession(_client(), SPEC, cfg, min_batch=0, votes=False, batch_growth=0.9, max_batch=None)
    for t in subs:
        asyncio.run(a.submit(t))
        asyncio.run(b.submit(t))
    assert [x["submissions"] for x in b.batches] == [[0], [1], [2], [3]]
    for x, y in zip(a.batches, b.batches):
        assert _batch_view(x) == _batch_view(y) and y["flush_target"] == 0


# ----------------------------------------------------------------------------------------------- lineage / ledger
def test_lineage_keys_checkpoint_and_journal_under_growth(tmp_path):
    cfg = _cfg(seed=9)
    p = tmp_path / "s.json"
    s = BottleSession(_client(), SPEC, cfg, state_path=p, min_batch=80, batch_growth=0.5, max_batch=None,
                      votes=False)
    for t in _subs([40] * 12):
        asyncio.run(s.submit(t))
    assert len(s.batches) >= 3
    keys = [k for rec in s.batches for k in cert_streams(rec["certificate"])]
    assert len(keys) == len(set(keys)) and sorted(keys) == sorted(s.used_streams)
    for rec in s.batches:
        assert [tuple(k) for k in rec["streams"]] == cert_streams(rec["certificate"])
    st = json.loads(p.read_text(encoding="utf-8"))
    assert st["growth"] == {"v": 1, "batch_growth": 0.5, "max_batch": {"spec": None, "value": None, "share": 0.1},
                            "m_init": 0}
    assert [b["flush_target"] for b in st["batches"]] == [b["flush_target"] for b in s.batches]
    ev = [json.loads(x) for x in Path(str(p) + ".lineage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["ev"] for e in ev] == ["reserve", "commit"] * len(s.batches)


def test_swapped_artifact_under_growth_raises_before_any_call():
    cfg = _cfg(seed=10)
    s = BottleSession(_client(), SPEC, cfg, min_batch=60, batch_growth=0.5, max_batch=None, votes=False)
    for t in _subs([60, 60]):
        asyncio.run(s.submit(t))
    s2 = BottleSession(_client(), SPEC, cfg, min_batch=60, batch_growth=0.5, max_batch=None, votes=False)
    asyncio.run(s2.submit(_subs([60])[0]))
    first = s2.artifact                                   # the bottle after batch 0 (same lineage point)
    s.artifact = BottleArtifact(SPEC, first.texts, first.labels, version=first.version, src=first.src,
                                fingerprint=first.fingerprint)
    n0 = _calls(s.client)
    with pytest.raises(ValueError, match="lineage guard"):
        asyncio.run(s.submit(_subs([200], start=120)[0]))
    assert _calls(s.client) == n0


@pytest.mark.parametrize("g", [0.5, 1.0])
def test_stream_ledger_stays_exact_under_growth(g):
    sizes = [40] * 20
    s = _stream(_client(), _cfg(seed=12), sizes, min_batch=80, batch_growth=g, max_batch=None, votes=False)
    sr = s.report()["totals"]["stream_risk"]
    assert sr["status"] == "exact"
    assert all(e["ok"] for e in s._stream) and len(s._stream) == len(s.batches)
    law = StreamLaw()                       # the ledger's law is the convolution of the batches' label-free designs
    for e in s._stream:
        for looks, Nj in e["d"]:
            law.add_design(looks, Nj)
    assert law.N == s._stream_law.N == sum(x["N"] for x in s.batches)
    assert law.expected_closed_form == pytest.approx(s._stream_law.expected_closed_form, rel=1e-12)
    # the last batch is 'forced' only if it is below ITS target (not below min_batch)
    last, tgt = s.batches[-1], s.batches[-1]["flush_target"]
    assert (s._stream[-1]["t"] == "forced") == (last["N"] < tgt)
    assert sr["forced_flushes"] == sum(e["t"] == "forced" for e in s._stream)


def test_forced_flush_is_relative_to_the_growth_target():
    """A buffer above min_batch but below the grown target is not flushed automatically; flush(force=True) certifies
    it as 'forced', and the ledger note names the flush target."""
    cfg = _cfg(seed=14)
    s = BottleSession(_client(), SPEC, cfg, min_batch=60, batch_growth=1.0, max_batch=None, votes=False)
    for t in _subs([100, 100]):
        asyncio.run(s.submit(t))
    assert [x["N"] for x in s.batches] == [100, 100] and s.flush_target() == 200
    asyncio.run(s.submit(_subs([150], start=200)[0]))
    assert len(s.batches) == 2 and s.pending_items == 150           # 150 >= min_batch but < target 200
    asyncio.run(s.flush(force=True))
    assert s._stream[-1]["t"] == "forced" and s.batches[-1]["flush_target"] == 200
    sr = s.stream_risk()
    assert sr["status"] == "exact" and sr["forced_flushes"] == 1
    assert any("flush target" in x and "force=True" in x for x in sr["notes"])


def test_growth_above_one_can_pass_the_cold_gate_and_v08_rules_apply():
    """g > 1 can make a batch >= 4 x memory: the pour is chunked under chunking='cold' (votes=False: the ledger stays
    exact) and, under votes='auto', the v0.8 withdrawal for a gate-passing pour still fires."""
    cfg = _cfg(seed=15)
    b = Bottler(_client(), SPEC, cfg)
    r0 = asyncio.run(b.run(WL.texts[:200]))
    m0 = len(r0.artifact.texts)
    for votes, status in ((False, "exact"), ("auto", "none")):
        s = BottleSession(_client(), SPEC, cfg, artifact=r0.artifact, min_batch=100, batch_growth=5.0,
                          max_batch=None, votes=votes)
        assert s.flush_target() == 5 * m0
        for t in _subs([100] * 10, start=200):
            asyncio.run(s.submit(t))
        asyncio.run(s.flush(force=True))
        assert s.batches[0]["N"] >= 4 * m0 and s.batches[0]["submissions"] == list(range(len(
            s.batches[0]["submissions"])))
        ch = (s.batches[0]["diagnostics"] or {}).get("chunking") or {}
        assert ch.get("rule") == "cold" and ch.get("decision") in ("cold", "warm")
        if ch.get("decision") == "cold":
            assert s.batches[0]["certificate"].get("chunked")
        assert s.stream_risk()["status"] == status


def test_flush_rows_and_report_carry_schedule_fields():
    s = BottleSession(_client(), SPEC, _cfg(seed=16), min_batch=100, batch_growth=0.5, max_batch=None, votes=False)
    for t in _subs([60] * 8):
        asyncio.run(s.submit(t))
    rep = s.report()
    assert len(rep["rows"]) == len(s.batches) >= 2
    for r, rec in zip(rep["rows"], s.batches):
        assert r["flush_target"] == rec["flush_target"] and r["items_waited"] == rec["N"] - rec["subs"][0][2]
    assert s.pending_items > 0
    rr = asyncio.run(s.flush(force=True))
    assert len(rr) == 1 and {"flush_target", "items_waited"} <= set(rr[0])
    tot = s.report()["totals"]
    assert tot["flush_schedule"]["planned_memory"] == s.planned_memory == 480
    assert tot["flush_schedule"]["next_flush_target"] == s.flush_target() == 240
    assert tot["batch_growth"] == 0.5 and tot["max_batch"] is None
