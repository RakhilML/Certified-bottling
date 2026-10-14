"""Expert budget: Bonferroni split of delta between the k-rule and the budget inversion.

Up to v0.9 a batch with BottleConfig(expert_budget=B) made the k-rule's claim at delta when the budget did not bind and
the budget threshold's test-inversion claim at delta when it did. Which one is made depends on the calibration labels,
and the switch was not covered: on a trap workload (N = 400) the per-batch failure rate is 0.146 (exact, below)
against delta = 0.1. v0.9.1 runs the k-rule at delta/2 and the inversion at delta/2; both failure events are defined
on every draw, so P[the claim made fails] <= delta/2 + delta/2 = delta whatever the labels select.

Covers:
  (a) expert_budget=None: byte-identical outputs to the code without the split (loaded by importlib from
      $BOTTLEKIT_PRE_BUDGETFIX_SNAPSHOT when set) on a fixed chain (run, warm, cold, auto, memo and chunked pours, fixed and
      staged calibration, a session report), and pinned digests of the same outputs (tests/data/budgetfix_pins.json,
      produced by the unpatched code);
  (b) EXACT rule-level failure probability (tests/budget_exact.py: hypergeometric recursion, tied to the package's
      fixed_sequence / certified_eps by exhaustive enumeration): v0.9 fails on the trap (0.146 > 0.1), the split
      holds there and on every layout of an exact worst-case search, each part within delta/2;
  (c) the real harness on the trap (cold pours, 200 draws);
  (d) certificate text and fields, risk profile, health precision, session statement and stream ledger under a
      budget; every path (sequential single look, staged, custom looks, fixed calibration, hypergeometric, memo pours,
      trivial memo populations, refused chunk designs).
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
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
from bottlekit import BottleConfig, Bottler, BottleSession  # noqa: E402
from bottlekit.certify import k_used, single_look_design, tune_size, tuned_looks  # noqa: E402
from budget_exact import brute_force, design, exact_fail, layout, m_upper_table  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
PINS = DATA / "budgetfix_pins.json"
EPS, DELTA = 0.05, 0.1
WL = toy_workload(3200, seed=17)
GOLD = {t[:400]: g for t, g in zip(WL.texts, WL.gold)}


# ----------------------------------------------------------------------------------------------- helpers
def _client(acc=None, cls=MockClient):
    return cls(GOLD, acc=acc or {"cheap": 0.8, "expert": 0.95})


def _strip(o):
    """Drop wall-clock fields (never byte-identical) anywhere in a nested structure."""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k not in ("wall_time_s", "seconds")}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    return o


def _canon(res) -> str:
    a = res.artifact
    art = None if a is None else {"v": a.version, "texts": a.texts, "labels": a.labels, "src": list(a.src),
                                  "fp": a.fingerprint, "forecast": H._plain(a.forecast), "history": a.history}
    return json.dumps(_strip({"answers": res.answers, "provenance": res.provenance, "certificate": res.certificate,
                              "ledger": res.ledger, "diagnostics": res.diagnostics, "artifact": art}),
                      sort_keys=True, default=str)


def _chains(pkg) -> dict:
    """Fixed chains with expert_budget=None through package `pkg` (a module with Bottler, BottleConfig,
    BottleSession). Returns {name: [canonical output strings]}."""
    T = WL.texts
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=7)
    out = {}
    # default: run -> warm -> cold -> auto pour
    b = pkg.Bottler(_client(), SPEC, pkg.BottleConfig(**base))
    r = asyncio.run(b.run(T[:400]))
    rs = [r]
    for lo, hi, votes in ((400, 800, True), (800, 1200, False), (1200, 1500, "auto")):
        r = asyncio.run(b.pour(r.artifact, T[lo:hi], votes=votes))
        rs.append(r)
    out["default"] = [_canon(x) for x in rs]
    # fixed calibration, staged design, custom looks + hypergeometric: run -> cold pour
    for name, kw in (("fixed", {"calibration": "fixed", "n_cal": 120}), ("staged", {"design": "staged"}),
                     ("custom_hyper", {"looks": [60, 150], "spending": [1.0, 3.0], "certificate": "hypergeometric"}),
                     ("maxexp", {"max_expected_eps": 0.03})):
        b = pkg.Bottler(_client(), SPEC, pkg.BottleConfig(**base, **kw))
        r0 = asyncio.run(b.run(T[:400]))
        r1 = asyncio.run(b.pour(r0.artifact, T[400:800], votes=False))
        out[name] = [_canon(r0), _canon(r1)]
    # memo pour (half recurring) and a chunked cold pour (2,000 >= 4 x 300)
    b = pkg.Bottler(_client(), SPEC, pkg.BottleConfig(**base, memo=True))
    r0 = asyncio.run(b.run(T[:300]))
    r1 = asyncio.run(b.pour(r0.artifact, T[100:300] + T[300:700], votes=True))
    out["memo"] = [_canon(r0), _canon(r1)]
    b = pkg.Bottler(_client(), SPEC, pkg.BottleConfig(**base, chunking="cold"))
    r0 = asyncio.run(b.run(T[:300]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r1 = asyncio.run(b.pour(r0.artifact, T[300:2300], votes=False))
    assert r1.certificate.get("chunked")
    out["chunked"] = [_canon(r0), _canon(r1)]

    # a session: 6 submissions of 150, min_batch 300, then the report (stream ledger, health, statement)
    async def go():
        s = pkg.BottleSession(_client(), SPEC, pkg.BottleConfig(**base), min_batch=300, votes="auto")
        for i in range(6):
            await s.submit(T[2300 + 150 * i:2300 + 150 * (i + 1)])
        await s.flush(force=True)
        return s
    s = asyncio.run(go())
    rep = s.report()
    out["session"] = [json.dumps(_strip({"rows": rep["rows"], "totals": rep["totals"], "statement": rep["statement"],
                                         "health": rep["health"]}), sort_keys=True, default=str)]
    return out


def _snapshot_dir() -> Path | None:
    """The package without the split: $BOTTLEKIT_PRE_BUDGETFIX_SNAPSHOT, else a bottlekit/ directory in a parent folder
    when it is not the package under test and does not carry the split."""
    env = os.environ.get("BOTTLEKIT_PRE_BUDGETFIX_SNAPSHOT")
    if env:
        p = Path(env)
        return p if (p / "__init__.py").exists() else None
    here = Path(H.__file__).resolve().parent
    for c in Path(__file__).resolve().parents:
        d = c / "bottlekit"
        if (d / "__init__.py").exists() and d.resolve() != here:
            src = (d / "harness.py").read_text(encoding="utf-8")
            return d if "budget_split" not in src else None
    return None


def _load_snapshot():
    base = _snapshot_dir()
    if base is None:
        return None
    name = "bottlekit_pre_budgetfix"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, base / "__init__.py", submodule_search_locations=[str(base)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _digests(chains: dict) -> dict:
    return {k: [hashlib.sha256(s.encode("utf-8")).hexdigest() for s in v] for k, v in chains.items()}


def write_pins():
    """Run once with the UNPATCHED package available (from tests/):
    python -c "import test_budget_split as t; t.write_pins()" """
    snap = _load_snapshot()
    assert snap is not None, "the unpatched v0.9 package is needed to produce the pins"
    PINS.write_text(json.dumps({"produced_by": "bottlekit 0.9 (before the budget split)",
                                "digests": _digests(_chains(snap))}, indent=1), encoding="utf-8", newline="\n")


# ----------------------------------------------------------------------------------------------- (a) byte-identical
@pytest.fixture(scope="module")
def chains_new():
    return _chains(sys.modules["bottlekit"])


def test_no_budget_byte_identical_to_pre_split_code(chains_new):
    snap = _load_snapshot()
    if snap is None:
        pytest.skip("the code without the budget split is not present (set $BOTTLEKIT_PRE_BUDGETFIX_SNAPSHOT)")
    old = _chains(snap)
    assert old.keys() == chains_new.keys()
    for k in old:
        assert len(old[k]) == len(chains_new[k])
        for a, b in zip(old[k], chains_new[k]):
            assert a == b, f"chain {k!r} differs with expert_budget=None"


def test_no_budget_matches_pinned_digests(chains_new):
    pins = json.loads(PINS.read_text(encoding="utf-8"))["digests"]
    assert _digests(chains_new) == pins


# ----------------------------------------------------------------------------------------------- (b) exact rule level
def test_exact_recursion_equals_enumeration_through_the_package():
    """budget_exact.exact_fail (recursion) == brute_force (every calibration subset through fixed_sequence and
    certified_eps) on random small workloads, budgets and levels: ties the exact model to the package code."""
    rng = np.random.default_rng(11)
    for _ in range(25):
        N, n = int(rng.integers(9, 13)), int(rng.integers(3, 6))
        eps = float(rng.choice([0.1, 0.2]))
        kl, il = float(rng.choice([0.25, 0.5])), float(rng.choice([0.1, 0.25, 0.5]))
        budget = int(rng.integers(1, N - 2))
        bad = rng.random(N) < rng.random()
        e = exact_fail(bad, n, k_used(n, N, eps, kl, "rectified"), budget, m_upper_table(N, n, il),
                       int(math.floor(eps * N)))
        b = brute_force(bad, n, eps, kl, il, budget)
        assert e["fail"] == pytest.approx(b["fail"], abs=1e-12)
        assert e["p_binding"] == pytest.approx(b["p_binding"], abs=1e-12)


TRAP = layout(400, 25, 215, 40)          # 25 bad on top, 215 good, 40 bad, 120 good; budget 115


def test_exact_trap_old_rule_fails_and_the_split_holds():
    N, budget = 400, 115
    old = design(N, EPS, DELTA, budget_set=False)            # the v0.9 rule: n 117, k 4 at delta, inversion at delta
    assert (old["n"], old["k"]) == (117, 4)
    r_old = exact_fail(TRAP, old["n"], old["k"], budget, m_upper_table(N, old["n"], DELTA), old["m"])
    assert r_old["fail"] == pytest.approx(0.14629, abs=1e-5)  # Monte Carlo: 0.1467 (se 0.0018), harness 0.144
    assert r_old["fail"] > DELTA
    # each part alone holds at delta: it is the switch that fails
    assert r_old["p_k_rule_fails"] <= DELTA and r_old["p_inversion_fails"] <= DELTA
    new = design(N, EPS, DELTA, budget_set=True)             # v0.9.1: n tuned at delta/2 (116), k 3, inversion delta/2
    assert (new["n"], new["k"], new["k_level"], new["inv_level"]) == (116, 3, 0.05, 0.05)
    r = exact_fail(TRAP, new["n"], new["k"], budget, m_upper_table(N, new["n"], DELTA / 2), new["m"])
    assert r["fail"] == pytest.approx(0.05313, abs=1e-5) and r["fail"] <= DELTA
    assert r["p_k_rule_fails"] <= DELTA / 2 and r["p_inversion_fails"] <= DELTA / 2
    assert r["fail"] <= r["p_k_rule_fails"] + r["p_inversion_fails"] + 1e-12     # the union bound, exactly


# the worst layouts of the exact search (a grid of 840 layouts + coordinate search, for
# each rule; v0.9's worst 0.1686 at (25, 59, 80, 240), the split's worst 0.0865 at (24, 64, 80, 240)) and a spread
SEARCH_LAYOUTS = [  # (top, gap, mid, budget)
    (25, 215, 40, 115), (25, 59, 80, 240), (24, 64, 80, 240), (24, 215, 40, 115), (24, 155, 40, 171),
    (24, 235, 20, 114), (22, 200, 60, 100), (27, 180, 30, 140), (24, 100, 80, 160), (30, 150, 120, 90),
]


@pytest.mark.parametrize("top,gap,mid,budget", SEARCH_LAYOUTS)
def test_exact_split_holds_on_search_layouts(top, gap, mid, budget):
    N = 400
    new = design(N, EPS, DELTA, budget_set=True)
    r = exact_fail(layout(N, top, gap, mid), new["n"], new["k"], budget, m_upper_table(N, new["n"], DELTA / 2),
                   new["m"])
    assert r["fail"] <= DELTA
    assert r["p_k_rule_fails"] <= DELTA / 2 + 1e-12 and r["p_inversion_fails"] <= DELTA / 2 + 1e-12
    if (top, gap, mid, budget) == (24, 64, 80, 240):       # the split's worst layout found: 0.0865
        assert r["fail"] == pytest.approx(0.08650, abs=1e-5)
    if (top, gap, mid, budget) == (25, 59, 80, 240):       # v0.9's worst layout found: 0.1686 under v0.9
        old = design(N, EPS, DELTA, budget_set=False)
        r_old = exact_fail(layout(N, top, gap, mid), old["n"], old["k"], budget, m_upper_table(N, old["n"], DELTA),
                           old["m"])
        assert r_old["fail"] == pytest.approx(0.16857, abs=1e-5)


def test_exact_split_other_size():
    """N = 600, eps = 0.02 (m = 12; v0.9: n 222, k 3; split: n 217, k 2), the worst layouts of an exact search for
    each rule: v0.9 reaches 0.1735 on its worst layout, the split 0.0692 there and
    0.0881 on its own worst layout."""
    N, eps = 600, 0.02
    old, new = design(N, eps, DELTA, False), design(N, eps, DELTA, True)
    assert (old["n"], old["k"], new["n"], new["k"]) == (222, 3, 217, 2)
    MUo, MUn = m_upper_table(N, old["n"], DELTA), m_upper_table(N, new["n"], DELTA / 2)
    lay_old, b_old = layout(N, 16, 88, 96), 360          # v0.9's worst found
    lay_new, b_new = layout(N, 15, 228, 108), 258        # the split's worst found
    r = exact_fail(lay_old, old["n"], old["k"], b_old, MUo, old["m"])
    assert r["fail"] == pytest.approx(0.17354, abs=1e-5) and r["fail"] > DELTA
    for lay, b in ((lay_old, b_old), (lay_new, b_new)):
        r = exact_fail(lay, new["n"], new["k"], b, MUn, new["m"])
        assert r["fail"] <= DELTA and r["p_k_rule_fails"] <= DELTA / 2 + 1e-12
        assert r["p_inversion_fails"] <= DELTA / 2 + 1e-12
    assert r["fail"] == pytest.approx(0.08806, abs=1e-5)


# ----------------------------------------------------------------------------------------------- (c) harness, trap
class TrapClient(MockClient):
    """The expert answers from `table` when set (fixed per batch by the trap hook before the calibration draw)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.table = {}

    def _answer(self, alias, text, salt, pos=None, k=None):
        if alias == "expert" and text in self.table:
            return self.table[text]
        return super()._answer(alias, text, salt, pos, k)


def _trap_hook(top, gap, mid):
    orig = H.Bottler._certify_and_escalate

    async def trap(self, texts, final, score, diag, **kw):
        order = np.argsort(-np.asarray(score, float), kind="stable")
        bad = set(order[:top].tolist()) | set(order[top + gap:top + gap + mid].tolist())
        for i, t in enumerate(texts):
            self.client.table[H.expert_key(self.spec, t)] = (next(x for x in SPEC.names if x != final[i])
                                                             if i in bad else final[i])
        return await orig(self, texts, final, score, diag, **kw)
    return orig, trap


def test_harness_trap_cold_pours_hold_delta():
    """200 cold pours of fresh 400-item batches (export, forecast off: the certificate path is unchanged by them),
    the trap adapted to the split's k (top = m+1+k(delta/2) = 24; the exact worst layout of the search, budget 240):
    the realized violation rate stays near the exact rule-level rate (<= delta). The 1,000-draw before/after runs are
    in NOTES.md."""
    top, gap, mid, budget = 24, 64, 80, 240
    wl0 = toy_workload(400, seed=3)
    r0 = asyncio.run(Bottler(MockClient({t[:400]: g for t, g in zip(wl0.texts, wl0.gold)}), SPEC,
                             BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=1)).run(wl0.texts))
    art = r0.artifact
    orig, trap = _trap_hook(top, gap, mid)
    H.Bottler._certify_and_escalate = trap
    viol = capped = 0
    D = 200
    try:
        for d in range(D):
            wl = toy_workload(400, seed=10_000 + d)
            wl.texts[:] = [f"{t} b{d}" for t in wl.texts]          # fresh items per draw
            cl = TrapClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.85, "expert": 1.0})
            cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=100 + d, expert_budget=budget,
                               export=False, forecast=False, health=False)
            res = asyncio.run(Bottler(cl, SPEC, cfg).pour(art, wl.texts, votes=False))
            ex = [cl.table[H.expert_key(SPEC, t)] for t in wl.texts]
            share = float(np.mean([a != e for a, e in zip(res.answers, ex)]))
            viol += share > res.certificate["eps_certified"] + 1e-12
            capped += bool(res.certificate.get("budget_capped"))
    finally:
        H.Bottler._certify_and_escalate = orig
    assert 0 < capped < D
    assert viol / D <= 0.13, viol         # measured: 15/200 = 0.075 (exact rule-level rate of this layout: 0.0865)


# ----------------------------------------------------------------------------------------------- (d) fields and text
def _run(N=600, **kw):
    wl = toy_workload(N, seed=1)
    cfg = BottleConfig(**{"cheap": "cheap", "expert": "expert", "mode": "fused", "k": 20, "seed": 2, **kw})
    return asyncio.run(Bottler(MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)},
                                          acc={"cheap": 0.7, "expert": 0.95}), SPEC, cfg).run(wl.texts))


def test_binding_budget_certificate_fields_and_text():
    res = _run(eps=0.01, expert_budget=40)
    c = res.certificate
    assert c["budget_capped"] is True and c["delta"] == DELTA and c["expert_budget"] == 40
    assert c["delta_split"] == {"k_rule": 0.05, "budget_inversion": 0.05}
    g = c["guarantee"]
    assert g.startswith(f"P[share of final answers != expert > {c['eps_achieved']:.4f}] <= 0.1 (budget-capped: the "
                        "budget inversion's claim; expert_budget=40: k-rule at 0.05 + budget inversion at 0.05, a "
                        "Bonferroni split of delta = 0.1")
    assert "no expected-error bound under an expert budget" in g
    assert sum(p == "expert_escalation" for p in res.provenance) <= 40
    # the k-rule part: tuned and applied at delta/2 (the unbudgeted tau is the k-rule's at 0.05)
    n0 = single_look_design(600, 0.01)[0][0]
    n = tuned_looks([n0], [1.0], 600, 0.01, 0.05, "rectified")[0]
    assert c["n_cal"] == n and c["risk"]["delta_split"]["k_rule"] == 0.05
    r = c["risk"]
    assert r["basis"] == "budget split" and r["budget_binding"] and r["looks"] is None
    assert r["eps_at_delta"]["0.1"] == c["eps_achieved"]
    pr = res.diagnostics["health"]["precision"]
    assert pr["basis"] == "budget split" and pr["u_delta"] == int(round(c["eps_achieved"] * 600))


def test_nonbinding_budget_certificate_fields_and_text():
    res = _run(expert_budget=10_000)
    c, r = res.certificate, res.certificate["risk"]
    assert c["budget_capped"] is False and c["delta"] == DELTA and "eps_achieved" not in c
    assert c["guarantee"].startswith("P[share of final answers != expert > 0.05] <= 0.1 (the budget did not bind, so "
                                     "this claim comes from the k-rule; the probability is over the whole procedure, "
                                     "not conditional on that branch; expert_budget=10000: k-rule at 0.05 + budget "
                                     "inversion at 0.05")
    (look,) = r["looks"]
    n0 = single_look_design(600, EPS)[0][0]
    assert look["n_cal"] == tuned_looks([n0], [1.0], 600, EPS, 0.05, "rectified")[0] == c["n_cal"]
    assert look["alpha"] == 0.05 and look["k"] == k_used(look["n_cal"], 600, EPS, 0.05, "rectified")
    assert r["expected_eps_bound"] is None and r["tail"] is None and r["delta_spent"] is None
    pr = res.diagnostics["health"]["precision"]
    assert pr["basis"] == "budget split" and pr["u_delta"] <= int(math.floor(EPS * 600))
    assert pr["u_delta"] == int(round(r["eps_at_delta"]["0.1"] * 600))
    assert "Bonferroni split" in pr["statement"]


def test_unusable_expert_answers_keep_the_split_in_the_text():
    from test_bottlekit_v3 import NoneExpertClient
    wl = toy_workload(600, seed=1)
    cl = NoneExpertClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, wl.texts[::25],
                          acc={"cheap": 0.7, "expert": 0.95})
    res = asyncio.run(Bottler(cl, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=2,
                                                     eps=0.01, expert_budget=40)).run(wl.texts))
    c = res.certificate
    assert c["n_expert_unusable"] > 0
    assert "expert-routed items without a usable expert answer" in c["guarantee"]
    assert "k-rule at 0.05 + budget inversion at 0.05" in c["guarantee"] and " <= 0.1 " in c["guarantee"]


@pytest.mark.parametrize("kw", [{"calibration": "fixed", "n_cal": 150}, {"design": "staged"},
                                {"looks": [80, 200], "spending": [1.0, 2.0]}, {"certificate": "hypergeometric"},
                                {"max_expected_eps": 0.03}])
def test_every_calibration_path_splits(kw):
    """Fixed, staged, custom looks, hypergeometric, mean cap: the k-rule's pre-specified looks share delta/2 and the
    budget inversion runs at delta/2 (checked against the package's own inversion functions)."""
    res = _run(eps=0.01, expert_budget=40, **kw)
    c = res.certificate
    assert c["delta"] == DELTA and c["delta_split"] == {"k_rule": 0.05, "budget_inversion": 0.05}
    assert c["budget_capped"]
    seen = {}
    o_seq, o_fix = H.sequential_certified_eps, H.certified_eps

    def spy_seq(*a, **k):
        seen["level"] = a[7]
        return o_seq(*a, **k)

    def spy_fix(*a, **k):
        seen["level"] = a[4]
        return o_fix(*a, **k)
    H.sequential_certified_eps, H.certified_eps = spy_seq, spy_fix
    try:
        again = _run(eps=0.01, expert_budget=40, **kw)
    finally:
        H.sequential_certified_eps, H.certified_eps = o_seq, o_fix
    assert seen["level"] == 0.05 and again.certificate == c
    if kw.get("calibration") == "fixed":
        assert c["n_cal"] == tune_size(150, 600, 0.01, 0.05, "rectified")
    else:
        # the k-rule part's plan (recorded with the split risk when not binding) sums to delta/2
        res_nb = _run(eps=0.05, expert_budget=10_000, **kw)
        looks = res_nb.certificate["risk"]["looks"]
        assert sum(l["alpha"] for l in looks) == pytest.approx(0.05)
        meth = kw.get("certificate", "rectified")
        cap = kw.get("max_expected_eps")
        shares = [l["alpha"] / 0.05 for l in looks]
        assert [l["k"] for l in looks] == [k_used(l["n_cal"], 600, 0.05, l["alpha"], meth,
                                                  max_expected_eps=None if cap is None else cap * w)
                                           for l, w in zip(looks, shares)]


def test_budget_split_in_memo_pours_and_not_in_trivial_populations():
    T = WL.texts
    base = dict(cheap="cheap", expert="expert", mode="fused", k=20, seed=7, memo=True)
    r0 = asyncio.run(Bottler(_client(), SPEC, BottleConfig(**base)).run(T[:300]))
    # half recurring: the population R is certified with the split; eps_achieved (if binding) is over the whole batch
    seen = {}
    o_seq = H.sequential_certified_eps

    def spy(*a, **k):
        seen.update(level=a[7], denom=k.get("denom"))
        return o_seq(*a, **k)
    H.sequential_certified_eps = spy
    try:
        r1 = asyncio.run(Bottler(_client(), SPEC, BottleConfig(**base, eps=0.01, expert_budget=10)).pour(
            r0.artifact, T[100:300] + T[300:500], votes=False))
    finally:
        H.sequential_certified_eps = o_seq
    c = r1.certificate
    assert c["n_memo"] > 0 and c["delta"] == DELTA and c["delta_split"] == {"k_rule": 0.05, "budget_inversion": 0.05}
    assert c["budget_capped"] and seen == {"level": 0.05, "denom": c["N"]}
    assert c["risk"]["basis"] == "budget split" and c["risk"]["population_N"] == c["N_population"]
    assert "certified population" in c["risk"]["scope"]
    # a batch of recurring expert-answered items plus 5 novel ones, allowance 'batch': m >= N_R, nothing to
    # calibrate and the budget never applies, so no split (the claim cannot fail)
    rec = [t for t, p in zip(T[:300], r0.provenance) if H.src_of(p) == "expert"][:150]
    with pytest.warns(UserWarning, match="memo_allowance='batch'"):
        r2 = asyncio.run(Bottler(_client(), SPEC, BottleConfig(**base, memo_allowance="batch", expert_budget=10)).pour(
            r0.artifact, rec + T[600:605], votes=False))
    c2 = r2.certificate
    assert c2.get("no_calibration_needed") and "delta_split" not in c2 and not c2.get("budget_capped")


def test_chunk_design_with_budget_is_refused():
    b = Bottler(_client(), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", expert_budget=50))
    assert "expert_budget set" in b._chunk_ineligible()
    assert b.chunk_plan(5000, 300)["chunks"] == 1
    cd = H._ChunkDesign(n=100, N=400, k=3, alpha=0.2, delta_batch=0.09, j=0, J=2, token=H._CHUNK_TOKEN)
    with pytest.raises(ValueError, match="without an expert budget"):
        asyncio.run(b._certify_and_escalate(WL.texts[:400], ["alpha"] * 400, np.linspace(1, 0, 400), {},
                                            chunk_design=cd))


def test_session_statement_and_stream_ledger_under_a_budget():
    """Budget batches: no expectation claim for ANY of them (binding or not: label-free exclusion), the statement
    says so; the stream ledger adds the k-rule's design at delta/2 for non-binding batches and withdraws on a
    binding one ('expert_budget is set' is named as a possible withdrawal)."""
    T = WL.texts

    async def go(budget):
        s = BottleSession(_client(), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=7,
                                                        expert_budget=budget), min_batch=300, votes=False)
        for i in range(3):
            await s.submit(T[300 * i:300 * (i + 1)])
        await s.flush(force=True)
        return s
    s = asyncio.run(go(10_000))                        # never binds
    rep = s.report()
    assert all(not r_["certificate"].get("budget_capped") for r_ in s.batches)
    assert rep["totals"]["batches_without_expected_bound"] == len(s.batches)
    assert "no expected-error bound is claimed for the 3 batch(es) under the expert budget" in rep["statement"]
    sr = rep["totals"]["stream_risk"]
    assert sr["status"] == "exact" and "expert_budget is set" in sr["withdrawal_reasons"]
    for e, rec in zip(s._stream, s.batches):
        (look,) = rec["certificate"]["risk"]["looks"]
        assert e["d"] == [[[[look["n_cal"], look["k"]]], rec["N"]]]
        assert look["k"] == k_used(look["n_cal"], rec["N"], EPS, 0.05, "rectified") and look["alpha"] == 0.05
    s2 = asyncio.run(go(5))                            # binds: the stream claim is withdrawn
    assert any(r_["certificate"].get("budget_capped") for r_ in s2.batches)
    assert s2.report()["totals"]["stream_risk"]["status"] == "none"


def test_resume_refuses_pre_091_budget_batches(tmp_path):
    """A budget session whose checkpoint holds a batch certified by the old rule (no delta_split) is
    refused on resume; the same checkpoint with the 0.9.1 field resumes."""
    import json
    import asyncio
    from bottlekit import BottleConfig, BottleSession
    from test_bottlekit import SPEC, MockClient, toy_workload
    wl = toy_workload(600, seed=71)
    mk = lambda: MockClient({x[:400]: g for x, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.8, "expert": 0.95})  # noqa: E731
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", k=20, seed=5, expert_budget=150)
    p = tmp_path / "stream.json"
    s = BottleSession(mk(), SPEC, cfg, state_path=p, min_batch=0, votes=False)
    asyncio.run(s.submit(wl.texts[:300]))
    st = json.loads(p.read_text(encoding="utf-8"))
    assert all("delta_split" in b["certificate"] for b in st["batches"])
    BottleSession.resume(p, mk(), SPEC, cfg)                     # 0.9.1 checkpoint: fine
    for b in st["batches"]:
        b["certificate"].pop("delta_split", None)
    p.write_text(json.dumps(st), encoding="utf-8")
    with pytest.raises(ValueError, match="<= 0.9"):
        BottleSession.resume(p, mk(), SPEC, cfg)

