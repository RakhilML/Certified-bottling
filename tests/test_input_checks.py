"""Input checks and failure messages: unknown model aliases, embeddings checked before any call, configuration
checks, load_workload errors, and the messages of expert and cheap-model failures.

No LLM: the MockClient of tests/test_bottlekit.py.
"""
import asyncio
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bottlekit import BottleArtifact, BottleConfig, Bottler, BottleSession, ExpertFailure, load_workload  # noqa: E402
from bottlekit.llm import ChatResult  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402


def _client(wl, **kw):
    return MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, **kw)


# ---------------------------------------------------------------- unknown model alias
@pytest.mark.parametrize("cheap,expert,mode", [("typo", "expert", "auto"), ("typo", "expert", "fused"),
                                               ("cheap", "typo", "fused"), ("cheap", "typo", "student")])
def test_unknown_alias_fails_before_any_call(cheap, expert, mode):
    from bottlekit import UnknownModelError
    wl = toy_workload(100)
    c = _client(wl)
    with pytest.raises(UnknownModelError, match="typo.*not in the models file.*known aliases") as ei:
        asyncio.run(Bottler(c, SPEC, BottleConfig(cheap=cheap, expert=expert, mode=mode)).run(wl.texts))
    assert isinstance(ei.value, KeyError) and isinstance(ei.value, ValueError)   # backward compatible
    assert c.ledger.rows == {}, "no LLM call before the error"


def test_unknown_alias_in_pour_but_cold_pour_needs_no_cheap():
    wl = toy_workload(300)
    c = _client(wl)
    art = BottleArtifact(SPEC, wl.texts[:200], wl.gold[:200])
    with pytest.raises(KeyError, match="nope"):
        asyncio.run(Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="nope")).pour(art, wl.texts[200:]))
    # a forced cold pour never needs the cheap alias (unchanged behaviour)
    res = asyncio.run(Bottler(c, SPEC, BottleConfig(cheap="not-there", expert="expert")).pour(art, wl.texts[200:],
                                                                                             votes=False))
    assert len(res.answers) == 100


def test_llm_client_unknown_alias_message():
    from bottlekit import LLMClient, ModelSpec
    cl = LLMClient({"a": ModelSpec("a", ["a"])})
    with pytest.raises(KeyError, match="'b' is not in the models file"):
        asyncio.run(cl.chat("b", [{"role": "user", "content": "x"}]))
    asyncio.run(cl.close())


# ---------------------------------------------------------------- embeddings validated up front
@pytest.mark.parametrize("bad", ["nan", "big"])
def test_bad_embeddings_fail_before_any_call(bad):
    wl = toy_workload(300)
    c = _client(wl)
    E = np.ones((300, 8))
    if bad == "nan":
        E[3, 2] = np.nan
    else:
        E *= 1e5                       # beyond float16: the export used to fail after every expert call
    with pytest.raises(ValueError, match="NaN|float16"):
        asyncio.run(Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused")).run(wl.texts, emb=E))
    assert c.ledger.rows == {}


def test_large_embeddings_allowed_without_export():
    wl = toy_workload(300)
    c = _client(wl)
    res = asyncio.run(Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", export=False))
                      .run(wl.texts, emb=np.full((300, 4), 1e5)))
    assert res.artifact is None and len(res.answers) == 300


def test_valid_embeddings_unchanged():
    wl = toy_workload(300)
    E = np.random.default_rng(0).normal(size=(300, 8)).astype(np.float32)
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused")
    r1 = asyncio.run(Bottler(_client(wl), SPEC, cfg).run(wl.texts, emb=E))
    r2 = asyncio.run(Bottler(_client(wl), SPEC, cfg).run(wl.texts, emb=E.tolist()))
    assert r1.answers == r2.answers and r1.provenance == r2.provenance


def test_session_rejects_bad_embeddings_before_queueing():
    wl = toy_workload(50)
    s = BottleSession(_client(wl), SPEC, BottleConfig(cheap="cheap", expert="expert"), min_batch=1000)
    with pytest.raises(ValueError, match="NaN"):
        asyncio.run(s.submit(wl.texts, emb=np.full((50, 4), np.nan)))
    assert s.pending_items == 0


# ---------------------------------------------------------------- configuration checks
@pytest.mark.parametrize("kw", [{"eps": 1.5}, {"eps": 1.0}, {"expert_budget": -5},
                                {"calibration": "fixed", "n_cal": 0}])
def test_invalid_config_rejected(kw):
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", **kw)


def test_valid_edge_configs_still_accepted():
    BottleConfig(cheap="c", expert="e", expert_budget=0)
    BottleConfig(cheap="c", expert="e", n_cal=0)            # unused with calibration='sequential'
    BottleConfig(cheap="c", expert="e", eps=0.999)


# ---------------------------------------------------------------- load_workload errors
def test_unknown_task_message():
    with pytest.raises(ValueError, match="unknown task 'trek'.*registry tasks"):
        load_workload("trek", n=10)


# ---------------------------------------------------------------- failure messages
class _BrokenExpert(MockClient):
    async def chat(self, alias, messages, max_tokens=256, temperature=0.0, phase="default", seed=None):
        if alias == "expert":
            r = ChatResult(text="", ok=False, model=alias, error="ConnectError: connection refused")
            self.ledger.add(phase, r)
            return r
        return await super().chat(alias, messages, max_tokens, temperature, phase, seed)


class _BrokenCheap(MockClient):
    async def chat(self, alias, messages, max_tokens=256, temperature=0.0, phase="default", seed=None):
        if alias == "cheap":
            r = ChatResult(text="", ok=False, model=alias, error="ConnectError: connection refused")
            self.ledger.add(phase, r)
            return r
        return await super().chat(alias, messages, max_tokens, temperature, phase, seed)


def test_expert_failure_names_the_request_error():
    wl = toy_workload(300)
    c = _BrokenExpert({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    with pytest.raises(ExpertFailure, match="Last unusable reply: request error: ConnectError"):
        asyncio.run(Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused")).run(wl.texts))


@pytest.mark.parametrize("mode", ["fused", "student"])
def test_cheap_outage_warns_and_outputs_unchanged(mode):
    wl = toy_workload(200)
    c = _BrokenCheap({t[:400]: g for t, g in zip(wl.texts, wl.gold)})
    with pytest.warns(UserWarning, match="cheap model 'cheap' gave no usable"):
        res = asyncio.run(Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert", mode=mode, n_build=150))
                          .run(wl.texts))
    assert res.coverage() == 0.0


def test_healthy_cheap_model_does_not_warn():
    wl = toy_workload(300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(Bottler(_client(wl), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused"))
                    .run(wl.texts))
