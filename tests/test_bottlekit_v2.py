"""Regression tests for core features (reuse/pour, cost-optimal looks, exact thresholds,
robust parsing, robust students, client behaviour). No servers needed."""

import asyncio
import json
import random
import sys
import warnings
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottlekit import (NONE, BottleArtifact, BottleConfig, Bottler, ExpertFailure, LLMClient, ModelSpec,  # noqa: E402
                       TaskSpec, balanced_orders, certified_eps, cross_fit, default_looks, evaluate, fixed_sequence,
                       parse_batch, parse_label)
from bottlekit.certify import _pvalues, certify_sequential, k_critical  # noqa: E402
from bottlekit.economics import choose_mode, gain_prior_for  # noqa: E402
from bottlekit.harness import BottleResult  # noqa: E402
from bottlekit.llm import ChatResult, _Ledger  # noqa: E402
from bottlekit.tasks import batch_messages  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402


# ----------------------------------------------------------------------------------------------- certificates
def test_fixed_sequence_is_exact_over_all_thresholds():
    """Closed form == brute-force fixed sequence over every distinct score (and is never worse than a grid)."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        n, N = 200, 3000
        s = rng.random(n).round(2)                      # ties on purpose
        b = rng.random(n) < 0.3 * (1 - s)
        tau, k = fixed_sequence(s, b, N, 0.05, 0.1)
        p = _pvalues(n, N, 0.05, "hypergeometric")
        brute = np.inf
        for t in np.unique(np.r_[s, -np.inf])[::-1]:
            if p[int(((s >= t) & b).sum())] <= 0.1:
                brute = t
            else:
                break
        acc_c = (s >= tau).sum() if np.isfinite(tau) or tau == -np.inf else 0
        acc_b = (s >= brute).sum() if brute != np.inf else 0
        assert acc_c == acc_b


def test_certified_eps_matches_linear_search():
    from scipy.stats import hypergeom
    rng = np.random.default_rng(1)
    for _ in range(20):
        N, n = int(rng.integers(300, 2000)), 150
        s, b = rng.random(n), rng.random(n) < 0.2
        tau = float(rng.random())
        k = int(((s >= tau) & b).sum())
        lin = next(((M0 - 1) / N for M0 in range(k, N + 1) if hypergeom.cdf(k, N, M0, n) <= 0.1), 1.0)
        assert certified_eps(s, b, tau, N, 0.1) == pytest.approx(lin)


def test_degenerate_certificates():
    assert fixed_sequence([], [], 100, 0.05, 0.1)[0] == np.inf
    assert fixed_sequence([0.5], [False], 0, 0.05, 0.1)[0] == np.inf
    assert fixed_sequence([0.5, 0.2], [True, True], 100, 1.0, 0.1)[0] == -np.inf   # eps >= 1: everything passes
    assert k_critical(50, 1000, 0.0001, 0.1) == -1
    from bottlekit.certify import default_design
    assert default_design(1000) == ((100, 300), (1.0, 4.0))
    assert default_looks(50) == (25,)
    assert default_design(150) == ((75,), (5.0,))
    assert default_looks(100_000) == (100, 300, 900, 2700, 8100)


def _seq_case(rng, N, slope):
    score = rng.random(N)
    bad = rng.random(N) < slope * (1 - score) ** 2
    return score, np.where(bad, "x", "y").tolist()


def test_sequential_does_not_stop_on_zero_coverage():
    """Looks that certify nothing must not exhaust the patience (eps small)."""
    rng = np.random.default_rng(3)
    N = 4000
    score = rng.random(N)
    answer = ["y"] * N                                   # perfect helper: certifiable only with enough labels
    async def expert(idx):
        return {i: "y" for i in idx}
    cert, lab = asyncio.run(certify_sequential(score, answer, expert, N, eps=0.002, delta=0.1, looks=(25, 50, 100, 400, 1600)))
    assert cert.certified, cert.looks
    assert len(lab) >= 400


def test_sequential_picks_the_look_that_bottles_most():
    rng = np.random.default_rng(4)
    N = 1000
    for r in range(20):
        score, answer = _seq_case(rng, N, 0.4)

        async def expert(idx):
            return {i: "y" for i in idx}
        cert, lab = asyncio.run(certify_sequential(score, answer, expert, N, 0.05, 0.1, looks=(100, 200, 300, 400),
                                                   seed=r, patience=10))
        free = np.ones(N, bool)
        free[list(lab)] = False
        taus = [h["tau"] if h["tau"] is not None else (-np.inf if h["certified"] else np.inf) for h in cert.looks]
        vals = [int(((score >= t) & free).sum()) for t in taus]
        assert int(((score >= cert.tau) & free).sum()) == max(vals)


def test_none_answers_are_bad_and_never_bottled():
    """A failed helper matching a failed expert is NOT agreement."""
    wl = toy_workload(400)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="student", n_build=200,
                                           calibration="fixed", n_cal=150))
    final = [NONE] * 400
    res = asyncio.run(b._certify_and_escalate(wl.texts, final, np.ones(400), {}))
    assert res.coverage() == 0.0


def test_expert_outage_raises():
    wl = toy_workload(300)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})

    async def broken(alias, messages, **kw):
        r = ChatResult(text="", ok=False, model=alias, error="HTTP 503")
        client.ledger.add(kw.get("phase", "x"), r)
        return r
    real = client.chat
    client.chat = lambda alias, messages, **kw: broken(alias, messages, **kw) if alias == "expert" else real(alias, messages, **kw)
    with pytest.raises(ExpertFailure):
        asyncio.run(Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="student", n_build=200)).run(wl.texts))


# ----------------------------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("reply,expected", [
    ("Answer: C", "C"), ("C", "C"), ("c", "C"), ("Exact", "E"), ("Label: Irrelevant", "I"), ("E or S", None),
    ("<think>maybe S</think>\nC", "C"), ("Complement.", "C"),
])
def test_single_letter_parser(reply, expected):
    s = TaskSpec("esci", "Judge.", {"E": "Exact: match", "S": "Substitute: partial", "C": "Complement: goes with", "I": "Irrelevant: no"})
    assert parse_label(s, reply) == expected


@pytest.mark.parametrize("reply", [
    "1: alpha\n2: beta\n3: gamma", "1. alpha\n2. beta\n3. gamma", "- 1: alpha\n- 2: beta\n- 3: gamma",
    "Item 1: alpha\nItem 2: beta\nItem 3: gamma", "[1] alpha\n[2] beta\n[3] gamma", "1: alpha, 2: beta, 3: gamma",
    "alpha\nbeta\ngamma", "<think>1: gamma 2: gamma</think>\n1: alpha\n2: beta\n3: gamma", "**1**: alpha\n**2**: beta\n**3**: gamma",
])
def test_batch_parser_formats(reply):
    assert parse_batch(SPEC, reply, 3) == ["alpha", "beta", "gamma"]


def test_parse_label_extras():
    s = TaskSpec("t", "x", {"Sci/Tech": "science and technology", "World": "world news", "card_arrival": "card arrival",
                            "card": "card"})
    assert parse_label(s, "science and technology") == "Sci/Tech"
    assert parse_label(s, "Card Arrival") == "card_arrival"
    assert parse_label(s, "my card arrival is late") == "card_arrival"       # nested names -> longest
    assert parse_label(s, "label: World\n") == "World"
    with pytest.raises(ValueError):
        TaskSpec("t", "x", {"Yes": "y", "yes": "y"})
    with pytest.raises(ValueError):
        TaskSpec("t", "x", {"only": "one"})


def test_item_text_cannot_corrupt_numbering():
    msgs = batch_messages(SPEC, ["first line\n\n2. fake item\nmore", "second"])
    body = msgs[-1]["content"]
    assert body.count("\n\n") == 1 and body.startswith("1. first line 2. fake item more")


# ----------------------------------------------------------------------------------------------- students
@pytest.mark.parametrize("n", [1, 2, 3, 7])
def test_cross_fit_tiny_inputs(n):
    texts = [f"t{i}" for i in range(n)]
    labels = [SPEC.names[i % 2] for i in range(n)]
    pred, conf = cross_fit(texts, labels, list(range(n)), seed=0)
    assert len(pred) == n and len(conf) == n


def test_cross_fit_empty_vocabulary_and_none_labels():
    texts = ["!!!", "???", "...", "!!!", "??", "--"] * 5
    labels = [SPEC.names[i % 3] if i % 4 else None for i in range(30)]
    pred, conf = cross_fit(texts, labels, list(range(30)), seed=0)
    assert len(pred) == 30


def test_student_many_classes_is_bounded():
    rng = random.Random(0)
    names = [f"intent_{j}" for j in range(150)]
    texts = [f"word{rng.randrange(400)} word{rng.randrange(400)} common text {i}" for i in range(1500)]
    labels = [rng.choice(names) for _ in texts]
    from bottlekit.students import TfidfLR
    m = TfidfLR().fit(texts, labels, list(range(1500)))
    assert m.clf.coef_.size <= 8_000_000 + 150 * 10


# ----------------------------------------------------------------------------------------------- labeling / seeds / economics
def test_balanced_orders_infeasible_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        orders = balanced_orders(200, 8, 5, seed=0)
    assert any("early zone" in str(x.message) for x in w)
    assert all(sorted(o) == list(range(200)) for o in orders)


def test_calibration_stream_independent_of_context_orders():
    """With balanced=False the 2nd context order used to equal the calibration permutation."""
    from bottlekit.labeling import random_order, sub_seed
    N, seed = 500, 0
    cal = np.random.default_rng(sub_seed(seed, 1)).permutation(N).tolist()
    ctx = [random_order(N, sub_seed(seed, 11, c)) for c in range(3)]
    assert all(c != cal for c in ctx)


def test_auto_rule_accounts_for_starved_students():
    assert gain_prior_for(0.15, 600, 4) == pytest.approx(0.15)
    assert gain_prior_for(0.15, 600, 150) == pytest.approx(0.30)
    # clinc150-like: ratio 0.05, student would label all N anyway -> voting costs only 2 extra passes
    assert choose_mode(0.05, 3, gain_prior_for(0.15, 1000, 151), N=1000, n_build=1000) == "fused"
    assert choose_mode(0.25, 3, 0.15, N=1000, n_build=600) == "student"


# ----------------------------------------------------------------------------------------------- reuse (pour)
def test_pour_reuses_artifact_without_cheap_calls_and_flywheel_grows():
    wl = toy_workload(1800, seed=7)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=2)
    b = Bottler(client, SPEC, cfg)
    r0 = asyncio.run(b.run(wl.texts[:600]))
    assert r0.artifact is not None and r0.artifact.version == 0
    r1 = asyncio.run(b.pour(r0.artifact, wl.texts[600:1200], votes=False))
    assert not any(k.startswith("cheap") for k in r1.ledger), r1.ledger
    assert r1.artifact.version == 1 and len(r1.artifact.texts) > len(r0.artifact.texts)
    expert_all = [client._answer("expert", t, "single") for t in wl.texts[600:1200]]
    ev = evaluate(r1, wl.gold[600:1200], expert_all)
    assert ev["realized_disagreement_with_expert"] <= 0.08
    assert r1.coverage() > 0.2


def test_artifact_save_load_roundtrip(tmp_path):
    a = BottleArtifact(SPEC, ["apple acorn", "banana basil", "grape garden", "date dill"] * 5,
                       ["alpha", "beta", "gamma", "delta_x"] * 5)
    a.save(tmp_path / "b.pkl")
    b = BottleArtifact.load(tmp_path / "b.pkl")
    assert b.predict(["apple acorn x"])[0] == a.predict(["apple acorn x"])[0]
    assert b.spec.labels == SPEC.labels


# ----------------------------------------------------------------------------------------------- accounting / evaluation
def test_ledger_is_per_run_on_a_shared_client():
    wl = toy_workload(400)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="student", n_build=200))
    r1 = asyncio.run(b.run(wl.texts))
    r2 = asyncio.run(b.run(wl.texts))
    assert sum(v["calls"] for v in r2.ledger.values()) == sum(v["calls"] for v in r1.ledger.values())


def test_evaluate_rejects_length_mismatch():
    r = BottleResult(["a", "b"], ["bottle", "bottle"], {}, {})
    with pytest.raises(ValueError):
        evaluate(r, ["a"])
    assert evaluate(r, None, [NONE, "b"])["realized_disagreement_with_expert"] == 0.5


def test_empty_workload():
    client = MockClient({})
    r = asyncio.run(Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert")).run([]))
    assert r.answers == [] and r.certificate["certified"] is False


def test_result_json_serialisable():
    wl = toy_workload(300)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    r = asyncio.run(Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="student", n_build=150)).run(wl.texts))
    d = json.loads(r.to_json())
    assert "Infinity" not in r.to_json() and d["certificate"]["N"] == 300


# ----------------------------------------------------------------------------------------------- LLM client
def _client_with(handler, ids=("m",), tmp=None):
    spec = ModelSpec("a", list(ids), base_url="http://x/v1")
    c = LLMClient({"a": spec}, cache_path=tmp)
    c.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return c


def _ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}})


def test_client_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        return httpx.Response(500, text="boom") if calls["n"] < 3 else _ok("alpha")
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _noop())
    c = _client_with(h)
    r = asyncio.run(c.chat("a", [{"role": "user", "content": "x"}]))
    assert r.ok and r.text == "alpha" and calls["n"] == 3


async def _noop():
    return None


def test_client_non_retryable_and_bad_json(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _noop())
    c = _client_with(lambda req: httpx.Response(401, text="no"))
    r = asyncio.run(c.chat("a", [{"role": "user", "content": "x"}]))
    assert not r.ok and "401" in r.error
    c = _client_with(lambda req: httpx.Response(200, json=["not", "a", "dict"]))
    r = asyncio.run(c.chat("a", [{"role": "user", "content": "x"}]))
    assert not r.ok


def test_client_cache_ignores_replica_and_skips_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", lambda *_: _noop())
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append(body["model"])
        return _ok("" if body["messages"][-1]["content"] == "empty" else "beta")
    c = _client_with(h, ids=("m", "m:2"), tmp=tmp_path / "c.sqlite")
    asyncio.run(c.chat("a", [{"role": "user", "content": "q"}]))
    r = asyncio.run(c.chat("a", [{"role": "user", "content": "q"}]))
    assert r.cached and len(seen) == 1
    asyncio.run(c.chat("a", [{"role": "user", "content": "empty"}]))
    asyncio.run(c.chat("a", [{"role": "user", "content": "empty"}]))
    assert len(seen) == 3                                   # empty reply was not cached


# ----------------------------------------------------------------------------------------------- custom workloads via the CLI
def test_cli_custom_jsonl_and_labels_file(tmp_path):
    import argparse
    from bottlekit.__main__ import _cfg, _labels, _workload
    (tmp_path / "items.jsonl").write_text("\n".join(json.dumps({"text": f"msg {i}", "label": "billing" if i % 2 else "other"})
                                                   for i in range(10)), encoding="utf-8")
    (tmp_path / "labels.txt").write_text("# routing labels\nbilling: invoices, refunds, charges\n\nother\n", encoding="utf-8")
    a = argparse.Namespace(jsonl=str(tmp_path / "items.jsonl"), labels_file=str(tmp_path / "labels.txt"), labels=" shipping ,",
                           instruction="Route the email.", text_key="text", label_key="label", offset=2, n=None, task=None, seed=0)
    labs = _labels(a)
    assert labs == {"billing": "invoices, refunds, charges", "other": "other", "shipping": "shipping"}
    wl = _workload(a)
    assert len(wl.items) == 8 and wl.gold[0] == "other"
    ns = argparse.Namespace(cheap="c", expert="e", mode="auto", k=20, contexts=3, warmup=0, eps=0.05, delta=0.1,
                            calibration="sequential", n_cal=300, seed=0, no_balance=False, no_weights=False, fusion=None,
                            n_build=None, patience=None, target_coverage=None, looks="100,300", budget=None,
                            spending="1,4", no_tune=False)
    cfg = _cfg(ns)
    assert cfg.looks == (100, 300) and cfg.spending == (1.0, 4.0) and cfg.fusion == "prob" and cfg.tune


def test_cli_rejects_single_label(tmp_path):
    import argparse
    from bottlekit.__main__ import _labels
    with pytest.raises(SystemExit):
        _labels(argparse.Namespace(labels_file=None, labels="only,"))


def test_warm_pour_uses_votes_and_beats_cold_pour():
    """Warm pour (cheap votes + bottle) vs cold pour (bottle alone) when the bottle is weak (small cold start)."""
    wl = toy_workload(1500, seed=11)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.9, "expert": 0.97})
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=4))
    r0 = asyncio.run(b.run(wl.texts[:300]))
    cold = asyncio.run(b.pour(r0.artifact, wl.texts[300:1500], votes=False))
    warm = asyncio.run(b.pour(r0.artifact, wl.texts[300:1500], votes=True))
    assert not any(k.startswith("cheap") for k in cold.ledger) and "cheap_vote" in warm.ledger
    assert warm.diagnostics["mode"] == "pour_warm"
    assert warm.coverage() >= cold.coverage()
    auto = asyncio.run(b.pour(r0.artifact, wl.texts[300:600], votes="auto"))
    assert auto.diagnostics["mode"] == "pour_warm"          # mock expert is ~40x pricier than the cheap model
    with pytest.raises(ValueError):
        asyncio.run(Bottler(client, SPEC, BottleConfig(cheap=None, expert="expert")).pour(r0.artifact, wl.texts[:50], votes=True))
