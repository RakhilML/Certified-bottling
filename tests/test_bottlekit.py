"""Unit + end-to-end tests for bottlekit with a MOCK LLM (no servers needed).

The mock labeler simulates the measured phenomena: a cheap model with a batch cold start
(first quarter of each batch less accurate) and an expert with higher accuracy. Answers are deterministic per
(item, call) so votes across contexts differ while repeated identical calls agree.
"""

import asyncio
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bottlekit import (BottleConfig, Bottler, ModelSpec, TaskSpec, balanced_orders, certify, cross_fit,  # noqa: E402
                       evaluate, fixed_sequence, from_records, parse_batch, parse_label)
from bottlekit.certify import certify_sequential  # noqa: E402
from bottlekit.economics import choose_mode  # noqa: E402
from bottlekit.llm import ChatResult, _Ledger  # noqa: E402

SPEC = TaskSpec("toy", "Classify the item.", {"alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta_x": "delta x"})


# ----------------------------------------------------------------------------------------------- mock client
class MockClient:
    """Implements the LLMClient surface used by bottlekit (models, ledger, chat, map)."""

    def __init__(self, gold_by_text, acc=None, cold=0.15):
        self.gold = gold_by_text
        self.acc = acc or {"cheap": 0.85, "expert": 0.95}
        self.cold = cold
        self.models = {"cheap": ModelSpec("cheap", ["cheap"], price_in=0.05, price_out=0.1),
                       "expert": ModelSpec("expert", ["expert"], price_in=2.0, price_out=8.0)}
        self.ledger = _Ledger()

    def _answer(self, alias, text, salt, pos=None, k=None):
        g = self.gold[text[:400]]
        h = int(hashlib.md5(f"{alias}|{text}|{salt}".encode()).hexdigest()[:8], 16)
        rng = random.Random(h)
        acc = self.acc[alias]
        if pos is not None and k and pos < max(1, k // 4):
            acc -= self.cold
        if rng.random() < acc:
            return g
        return rng.choice([n for n in SPEC.names if n != g])

    async def chat(self, alias, messages, max_tokens=256, temperature=0.0, phase="default", seed=None):
        user = messages[-1]["content"]
        if user[:3].strip().startswith("1."):  # batch prompt
            parts = [p.split(". ", 1)[1] for p in user.split("\n\n")]
            salt = hashlib.md5(user.encode()).hexdigest()
            k = len(parts)
            text = "\n".join(f"{i + 1}: {self._answer(alias, t, salt, pos=i, k=k)}" for i, t in enumerate(parts))
            pt, ct = 40 * k, 6 * k
        else:
            text = self._answer(alias, user, "single")
            pt, ct = 60, 3
        r = ChatResult(text=text, ok=True, model=alias, prompt_tokens=pt, completion_tokens=ct, usd=self.models[alias].usd(pt, ct))
        self.ledger.add(phase, r)
        return r

    async def map(self, coros, limit=64):
        return await asyncio.gather(*coros)


def toy_workload(n=900, seed=0):
    rng = random.Random(seed)
    words = {"alpha": ["apple", "acorn", "amber"], "beta": ["banana", "basil", "beach"],
             "gamma": ["grape", "garlic", "garden"], "delta_x": ["date", "dill", "desert"]}
    recs = []
    for i in range(n):
        lab = rng.choice(SPEC.names)
        toks = rng.sample(words[lab], 2) + [rng.choice(sum(words.values(), [])) for _ in range(2)] + [f"id{i}"]
        rng.shuffle(toks)
        recs.append({"text": " ".join(toks), "label": lab})
    return from_records(SPEC, recs)


# ----------------------------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("reply,expected", [
    ("alpha", "alpha"), ("  Beta. ", "beta"), ("**gamma**", "gamma"), ("delta x", "delta_x"), ("delta_x", "delta_x"),
    ("The answer is beta", "beta"), ("", None), ("unknown", None), ("alpha or beta", None),
])
def test_parse_label(reply, expected):
    assert parse_label(SPEC, reply) == expected


def test_parse_batch_formats():
    reply = "1: alpha\n**2**: beta\n3) gamma\n4 - delta x\nnoise line\n5: zzz"
    assert parse_batch(SPEC, reply, 6) == ["alpha", "beta", "gamma", "delta_x", None, None]


def test_single_letter_labels():
    s = TaskSpec("esci", "Judge.", {"E": "Exact: match", "S": "Substitute: partial", "C": "Complement: goes with", "I": "Irrelevant: no"})
    assert parse_label(s, "Substitute") == "S"
    assert parse_label(s, "E") == "E"
    assert parse_label(s, "irrelevant") == "I"


# ----------------------------------------------------------------------------------------------- batching
def test_balanced_orders_cold_start_zone():
    n, k, C = 400, 20, 3
    orders = balanced_orders(n, k, C, seed=1)
    early = np.zeros(n, int)
    for o in orders:
        assert sorted(o) == list(range(n))
        for j, i in enumerate(o):
            early[i] += (j % k) < k // 4
    assert early.max() <= 1, "an item sits in the cold-start zone of more than one context"


# ----------------------------------------------------------------------------------------------- certificates
@pytest.mark.parametrize("method", ["binomial", "hypergeometric"])
def test_certificate_validity(method):
    """Simulated workloads: realized disagreement of the final output must exceed eps in at most ~delta of runs."""
    rng = np.random.default_rng(0)
    N, n_cal, eps, delta, R = 2000, 300, 0.05, 0.1, 300
    viol = 0
    for r in range(R):
        score = rng.random(N)
        bad = rng.random(N) < (0.45 * (1 - score)) ** 1.5  # low scores are more often wrong
        cal = rng.choice(N, n_cal, replace=False)
        tau, _ = fixed_sequence(score[cal], bad[cal], N, eps, delta, method)
        acc = score >= tau
        acc[cal] = False
        viol += (acc & bad).mean() > eps
    assert viol / R <= delta + 0.04


def test_sequential_certificate_validity_and_savings():
    rng = np.random.default_rng(1)
    N, eps, delta, R = 2000, 0.05, 0.1, 120
    viol, used = 0, []
    for r in range(R):
        score = rng.random(N)
        bad = rng.random(N) < (0.3 * (1 - score)) ** 2
        answer = np.where(bad, "x", "y").tolist()
        truth = ["y"] * N

        async def expert(idx):
            return {i: truth[i] for i in idx}
        cert, lab = asyncio.run(certify_sequential(score, answer, expert, N, eps, delta, looks=(100, 200, 300, 400), seed=r))
        acc = score >= cert.tau
        acc[list(lab)] = False
        viol += (acc & bad).mean() > eps
        used.append(cert.n_cal)
    assert viol / R <= delta + 0.04
    assert np.mean(used) < 400, "sequential certificate should usually stop before the last look"


# ----------------------------------------------------------------------------------------------- cross-fitting
def test_cross_fit_is_out_of_fold():
    """Random labels: in-sample a student memorises them; out-of-fold it must be at chance."""
    rng = random.Random(0)
    texts = [f"tok{i} filler{i % 7}" for i in range(600)]
    labels = [rng.choice(SPEC.names) for _ in texts]
    pred, conf = cross_fit(texts, labels, list(range(600)), seed=0)
    oof_acc = np.mean([p == l for p, l in zip(pred, labels)])
    assert oof_acc < 0.40, f"OOF accuracy {oof_acc:.2f} suggests leakage"


# ----------------------------------------------------------------------------------------------- economics
def test_auto_mode_rule():
    assert choose_mode(1 / 300) == "fused"     # frontier-priced expert
    assert choose_mode(0.25) == "student"      # open-model prices


# ----------------------------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("mode", ["fused", "student"])
def test_end_to_end_mock(mode):
    wl = toy_workload(900)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    cfg = BottleConfig(cheap="cheap", expert="expert", mode=mode, k=20, n_build=400, calibration="sequential",
                       looks=(100, 200, 300), seed=3)
    res = asyncio.run(Bottler(client, SPEC, cfg).run(wl.texts))
    assert len(res.answers) == len(wl.items) == len(res.provenance)
    assert set(res.provenance) <= {"bottle", "expert_calibration", "expert_escalation"}
    expert_all = [client._answer("expert", t, "single") for t in wl.texts]
    ev = evaluate(res, wl.gold, expert_all)
    assert ev["realized_disagreement_with_expert"] <= 0.08  # eps=0.05 + finite-sample slack for one run
    assert res.coverage() > 0.2, f"coverage {res.coverage():.2f} unexpectedly low on an easy toy task"
    assert any(k.startswith("expert") for k in res.ledger)
    if mode == "fused":
        assert res.diagnostics.get("cold_start_detected") is True, "label-free cold-start detection failed"


def test_budget_capped_certificate_is_valid_and_honest():
    """Under a tight expert budget the harness escalates at most `budget` items and reports eps_achieved, which must
    upper-bound the realized disagreement in >= 1 - delta of runs."""
    from bottlekit.certify import certified_eps
    rng = np.random.default_rng(5)
    N, n_cal, delta, R = 2000, 300, 0.1, 300
    cover = 0
    for r in range(R):
        score = rng.random(N)
        bad = rng.random(N) < 0.35 * (1 - score)
        cal = rng.choice(N, n_cal, replace=False)
        tau_b = float(np.sort(score)[400])          # budget threshold, independent of calibration labels
        eps_hat = certified_eps(score[cal], bad[cal], tau_b, N, delta)
        acc = score >= tau_b
        acc[cal] = False
        cover += (acc & bad).mean() <= eps_hat
    assert cover / R >= 1 - delta - 0.04


def test_end_to_end_budget_mode():
    wl = toy_workload(600)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc={"cheap": 0.7, "expert": 0.95})
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="student", k=20, n_build=300, calibration="fixed",
                       n_cal=150, eps=0.01, expert_budget=50, seed=1)
    res = asyncio.run(Bottler(client, SPEC, cfg).run(wl.texts))
    n_esc = sum(p == "expert_escalation" for p in res.provenance)
    assert n_esc <= 50
    if res.certificate.get("budget_capped"):
        assert res.certificate["eps_achieved"] > 0.01


def test_auto_mode_end_to_end_picks_fused_for_expensive_expert():
    wl = toy_workload(300)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="auto"))
    mode, info = b.plan(wl.texts)
    assert mode == "fused", info
