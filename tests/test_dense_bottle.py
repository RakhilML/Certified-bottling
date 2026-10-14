"""Dense-aware bottles (v0.3): embeddings carried in the BottleArtifact and used by pour() (cold and warm).
No servers and no network: embedders are mocks, the /v1/embeddings client is tested with httpx.MockTransport only."""

import asyncio
import hashlib
import json
import pickle
import random
import sys
import warnings
from pathlib import Path

import httpx
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bottlekit import (NONE, BottleArtifact, BottleConfig, Bottler, EmbeddingError, LLMClient, ModelSpec,  # noqa: E402
                       TaskSpec, TfidfLR, evaluate)
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

DIM = 16


def hashed_embed(texts, dim=DIM):
    """A fixed, label-free mock encoder: signed hashed bag of words, L2-normalised (a function of the text only)."""
    out = np.zeros((len(texts), dim), np.float32)
    for r, t in enumerate(texts):
        for w in str(t).lower().split():
            h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
            out[r, h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def signal_in_dense_only(n, seed=0, noise=0.3):
    """Texts carry NO label information; the dense rows do (one-hot of the label + noise)."""
    rng = np.random.default_rng(seed)
    labels = [SPEC.names[i] for i in rng.integers(0, 4, n)]
    texts = [f"tok{rng.integers(10 ** 9)} filler{rng.integers(10 ** 9)}" for _ in range(n)]
    D = rng.normal(0, noise, (n, 8)).astype(np.float32)
    for i, l in enumerate(labels):
        D[i, SPEC.names.index(l)] += 1.0
    return texts, labels, D


# ----------------------------------------------------------------------------------------------- students
def test_tfidflr_predict_with_dense_override():
    texts, labels, D = signal_in_dense_only(400)
    m = TfidfLR(dense=D[:300]).fit(texts[:300], labels[:300], list(range(300)))
    new_t, new_l, new_D = texts[300:], labels[300:], D[300:]
    pred, conf, _ = m.predict(new_t, list(range(100)), dense=new_D)
    assert np.mean([p == l for p, l in zip(pred, new_l)]) > 0.9
    # the override is exactly the indexed path when it carries the same rows
    a = m.predict(texts[:50], list(range(50)))
    b = m.predict(texts[:50], list(range(50)), dense=D[:50])
    assert a[0] == b[0] and np.allclose(a[1], b[1])
    # chunked prediction keeps rows aligned
    q = m.predict(new_t, list(range(100)), query=new_l, dense=new_D)[2]
    assert q.shape == (100,) and np.all((q >= 0) & (q <= 1))


def test_dense_override_shape_errors():
    texts, labels, D = signal_in_dense_only(60)
    m = TfidfLR(dense=D).fit(texts, labels, list(range(60)))
    with pytest.raises(ValueError):
        m.predict(texts[:10], list(range(10)), dense=D[:9])           # row-count mismatch
    with pytest.raises(ValueError):
        m.predict(texts[:10], list(range(10)), dense=D[:10, :5])      # width mismatch
    plain = TfidfLR().fit(texts, labels, list(range(60)))
    with pytest.raises(ValueError):
        plain.predict(texts[:10], list(range(10)), dense=D[:10])      # fitted without dense


# ----------------------------------------------------------------------------------------------- artifact
def test_artifact_emb_aligned_after_filter_dedupe_and_cap():
    # row i of emb encodes i, so alignment is checkable exactly (small ints are exact in float16)
    texts = ["a x", "b x", "a x", "c x", "d x", "b x", "e x", "f x"]
    labels = ["alpha", "beta", "gamma", NONE, "delta_x", "alpha", None, "beta"]
    emb = np.stack([np.arange(8.0), -np.arange(8.0)], 1)
    art = BottleArtifact(SPEC, texts, labels, emb=emb, embedder_id="mock")
    newest = {}
    for i, (t, l) in enumerate(zip(texts, labels)):
        if l is not None and l != NONE:
            newest[t] = i
    assert art.texts == ["a x", "b x", "d x", "f x"]                  # first-occurrence order, as before
    assert art.labels == [labels[newest[t]] for t in art.texts]       # newest label wins
    assert art.emb[:, 0].tolist() == [newest[t] for t in art.texts]   # ...and its embedding row
    assert art.emb.dtype == np.float16 and art.emb_dim == 2 and art.embedder_id == "mock"
    capped = BottleArtifact(SPEC, texts, labels, max_memory=2, emb=emb)
    assert capped.texts == ["d x", "f x"] and capped.emb[:, 0].tolist() == [4.0, 7.0]
    # the TF-IDF part of the memory is exactly what a TF-IDF artifact keeps
    plain = BottleArtifact(SPEC, texts, labels)
    assert plain.texts == art.texts and plain.labels == art.labels and not plain.dense
    # updated(): concatenation stays aligned (re-labelled duplicate takes the new row)
    up = art.updated(["b x", "g x"], ["gamma", "alpha"], emb=np.array([[10.0, -10.0], [11.0, -11.0]]))
    assert up.version == 1 and up.dense
    assert dict(zip(up.texts, up.emb[:, 0].tolist())) == {"a x": 2.0, "b x": 10.0, "d x": 4.0, "f x": 7.0, "g x": 11.0}
    assert dict(zip(up.texts, up.labels))["b x"] == "gamma"
    with pytest.raises(ValueError):
        BottleArtifact(SPEC, texts, labels, emb=emb[:7])                # row-count mismatch


def test_dense_artifact_predicts_with_emb_and_falls_back_without():
    texts, labels, D = signal_in_dense_only(500, seed=1)
    art = BottleArtifact(SPEC, texts[:400], labels[:400], emb=D[:400], embedder_id="mock")
    plain = BottleArtifact(SPEC, texts[:400], labels[:400])
    new_t, new_l, new_D = texts[400:], labels[400:], D[400:]
    pd, cd = art.predict(new_t, emb=new_D)
    assert np.mean([p == l for p, l in zip(pd, new_l)]) > 0.9
    pf, cf = art.predict(new_t)                                        # no new embeddings: TF-IDF fallback
    pp, cp = plain.predict(new_t)
    assert pf == pp and np.allclose(cf, cp)                            # == the TF-IDF artifact, exactly
    assert art.dense_status(new_D) == "dense" and art.dense_status(None) == "dense_missing"
    assert plain.dense_status(new_D) == "dense_missing" and plain.dense_status(None) == "off"
    p3, c3, q3 = art.predict(new_t, query=new_l, emb=new_D)
    assert p3 == pd and np.allclose(c3, cd) and q3.mean() > 0.7
    with pytest.raises(ValueError):
        art.predict(new_t, emb=new_D[:50])
    # a plain artifact ignores embeddings (rows still checked)
    assert plain.predict(new_t, emb=new_D)[0] == pp
    empty = BottleArtifact(SPEC, [], [], emb=np.zeros((0, 8)))
    assert empty.predict(["x"], emb=np.zeros((1, 8)))[0] == [None]


def test_updated_without_emb_drops_dense():
    texts, labels, D = signal_in_dense_only(300, seed=2)
    art = BottleArtifact(SPEC, texts[:200], labels[:200], emb=D[:200], embedder_id="mock")
    up = art.updated(texts[200:], labels[200:])                       # new batch without embeddings
    assert not up.dense and up.emb is None and up.embedder_id is None and len(up.texts) == 300
    up2 = art.updated(texts[200:], labels[200:], emb=D[200:])
    assert up2.dense and up2.emb.shape == (300, 8) and up2.embedder_id == "mock"
    plain = BottleArtifact(SPEC, texts[:200], labels[:200])
    assert not plain.updated(texts[200:], labels[200:], emb=D[200:]).dense   # memory has no embeddings


def test_save_load_roundtrip_float16_and_old_pickles(tmp_path):
    texts, labels, D = signal_in_dense_only(300, seed=3)
    art = BottleArtifact(SPEC, texts[:250], labels[:250], emb=D[:250], embedder_id="mock-v1", version=4)
    art.save(tmp_path / "d.pkl")
    raw = pickle.loads((tmp_path / "d.pkl").read_bytes())
    assert isinstance(raw["emb"], bytes) and tuple(raw["emb_shape"]) == (250, 8) and raw["embedder_id"] == "mock-v1"
    b = BottleArtifact.load(tmp_path / "d.pkl")
    assert b.dense and b.emb.dtype == np.float16 and np.array_equal(b.emb, art.emb)
    assert b.embedder_id == "mock-v1" and b.version == 4 and b.texts == art.texts and b.labels == art.labels
    p1, c1 = art.predict(texts[250:], emb=D[250:])
    p2, c2 = b.predict(texts[250:], emb=D[250:])
    assert p1 == p2 and np.array_equal(c1, c2)                         # exact: fitted on the stored float16 values
    # a v0.2 pickle (no embedding keys) still loads as a TF-IDF bottle
    old = {"spec": {"name": SPEC.name, "instruction": SPEC.instruction, "labels": SPEC.labels, "max_chars": SPEC.max_chars},
           "texts": texts[:50], "labels": labels[:50], "C": 4.0, "max_memory": 200_000, "version": 2}
    (tmp_path / "old.pkl").write_bytes(pickle.dumps(old))
    o = BottleArtifact.load(tmp_path / "old.pkl")
    assert not o.dense and o.version == 2 and len(o.texts) == 50
    plain = BottleArtifact(SPEC, texts[:250], labels[:250])
    plain.save(tmp_path / "p.pkl")
    assert "emb" not in pickle.loads((tmp_path / "p.pkl").read_bytes())


def test_embedder_id_and_width_mismatch_detected():
    texts, labels, D = signal_in_dense_only(120, seed=4)
    art = BottleArtifact(SPEC, texts[:100], labels[:100], emb=D[:100], embedder_id="enc-a")
    with pytest.raises(ValueError, match="width"):
        art.check_emb(np.zeros((20, 5)), 20)
    with pytest.raises(ValueError, match="mismatch"):
        art.check_emb(D[100:], 20, embedder_id="enc-b")
    assert art.check_emb(D[100:], 20, embedder_id="enc-a").shape == (20, 8)
    assert art.check_emb(D[100:], 20).shape == (20, 8)                 # unknown id: width decides
    with pytest.raises(ValueError):
        art.updated(texts[100:], labels[100:], emb=D[100:], embedder_id="enc-b")


# ----------------------------------------------------------------------------------------------- Bottler run/pour
def _toy(n=1500, seed=7, acc=None):
    wl = toy_workload(n, seed=seed)
    client = MockClient({t[:400]: g for t, g in zip(wl.texts, wl.gold)}, acc=acc)
    return wl, client


def test_run_exports_dense_artifact_and_pours_use_emb():
    wl, client = _toy()
    E = hashed_embed(wl.texts)
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=2, embedder_id="hash16"))
    r0 = asyncio.run(b.run(wl.texts[:500], emb=E[:500]))
    assert r0.artifact.dense and r0.artifact.embedder_id == "hash16" and r0.artifact.emb.shape[0] == len(r0.artifact.texts)
    assert r0.diagnostics["dense"] == "dense" and r0.diagnostics["next_bottle_dense"]
    cold = asyncio.run(b.pour(r0.artifact, wl.texts[500:1000], votes=False, emb=E[500:1000]))
    assert cold.diagnostics["dense"] == "dense" and not any(k.startswith("cheap") for k in cold.ledger)
    assert cold.artifact.dense and cold.artifact.version == 1 and cold.artifact.emb.shape[0] == len(cold.artifact.texts)
    warm = asyncio.run(b.pour(r0.artifact, wl.texts[500:1000], votes=True, emb=E[500:1000]))
    assert warm.diagnostics["mode"] == "pour_warm" and warm.diagnostics["dense"] == "dense"
    expert_all = [client._answer("expert", t, "single") for t in wl.texts[500:1000]]
    for r in (cold, warm):
        assert evaluate(r, wl.gold[500:1000], expert_all)["realized_disagreement_with_expert"] <= 0.08
    # pour scores come from the PRE-update bottle (trained on earlier batches only)
    pre = r0.artifact.predict(wl.texts[500:1000], emb=E[500:1000])
    bottled = [i for i, p in enumerate(cold.provenance) if p == "bottle"]
    assert all(cold.answers[i] == pre[0][i] for i in bottled)
    with pytest.raises(ValueError):                                    # width mismatch on pour, no embedder
        asyncio.run(b.pour(r0.artifact, wl.texts[1000:1100], votes=False, emb=np.zeros((100, 5), np.float32)))
    with pytest.raises(ValueError):                                    # row mismatch on run
        asyncio.run(b.run(wl.texts[:100], emb=E[:99]))


def test_pour_without_emb_on_dense_bottle_equals_tfidf_bottle_and_flags():
    wl, client = _toy()
    E = hashed_embed(wl.texts)
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=5))
    r0 = asyncio.run(b.run(wl.texts[:500], emb=E[:500]))
    with pytest.warns(UserWarning, match="dense_missing"):
        a = asyncio.run(b.pour(r0.artifact, wl.texts[500:1000], votes=False))
    t = asyncio.run(b.pour(r0.artifact.with_embeddings(None), wl.texts[500:1000], votes=False))
    assert a.answers == t.answers and a.provenance == t.provenance and a.certificate == t.certificate
    assert a.diagnostics["dense"] == "dense_missing" and not a.artifact.dense
    assert t.diagnostics["dense"] == "off"


def test_embedder_called_once_per_batch_before_calibration():
    wl, client = _toy(900)
    events = []
    real_chat = client.chat

    async def chat(alias, messages, **kw):
        events.append(("chat", kw.get("phase")))
        return await real_chat(alias, messages, **kw)
    client.chat = chat

    def embedder(texts):
        events.append(("embed", len(texts)))
        return hashed_embed(texts)
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=1, embedder=embedder,
                                           embedder_id="hash16"))
    r0 = asyncio.run(b.run(wl.texts[:400]))
    assert [e for e in events if e[0] == "embed"] == [("embed", 400)] and events[0] == ("embed", 400)
    assert r0.artifact.dense and r0.artifact.embedder_id == "hash16"
    events.clear()
    r1 = asyncio.run(b.pour(r0.artifact, wl.texts[400:900], votes=True))
    assert [e for e in events if e[0] == "embed"] == [("embed", 500)] and events[0] == ("embed", 500)
    assert r1.diagnostics["dense"] == "dense" and r1.artifact.dense


def test_embedder_reembeds_bottle_memory_on_mismatch_or_tfidf_bottle():
    wl, client = _toy(1000)
    b0 = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=3))
    r0 = asyncio.run(b0.run(wl.texts[:400]))                           # TF-IDF bottle (no embeddings)
    calls = []

    def emb32(texts):
        calls.append(len(texts))
        return hashed_embed(texts, 32)
    b1 = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", seed=3, embedder=emb32, embedder_id="h32"))
    r1 = asyncio.run(b1.pour(r0.artifact, wl.texts[400:700], votes=False))
    assert r1.diagnostics["dense_memory_reembedded"] and r1.diagnostics["dense"] == "dense"
    assert calls == [300, len(r0.artifact.texts)] and r1.artifact.dense and r1.artifact.embedder_id == "h32"
    # a different embedder: memory re-embedded with it, never mixed
    b2 = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", seed=3, embedder=hashed_embed, embedder_id="h16"))
    r2 = asyncio.run(b2.pour(r1.artifact, wl.texts[700:1000], votes=False))
    assert r2.diagnostics["dense_memory_reembedded"] and r2.artifact.embedder_id == "h16" and r2.artifact.emb_dim == DIM
    # explicit embeddings from a different embedder and no embedder to fix it: refused
    b3 = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", seed=3, embedder_id="h32"))
    with pytest.raises(ValueError, match="mismatch"):
        asyncio.run(b3.pour(r2.artifact, wl.texts[700:800], votes=False, emb=hashed_embed(wl.texts[700:800])))


def test_embedder_failure_falls_back_to_tfidf():
    wl, client = _toy(600)

    def broken(texts):
        raise RuntimeError("embedding endpoint down")
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=1, embedder=broken))
    with pytest.warns(UserWarning, match="embedder failed"):
        r = asyncio.run(b.run(wl.texts[:600]))
    assert "embedding endpoint down" in r.diagnostics["embedder_error"] and not r.artifact.dense
    assert r.certificate["certified"] or r.coverage() == 0.0


def test_async_embedder_is_awaited():
    wl, client = _toy(400)

    async def aemb(texts):
        return hashed_embed(texts)
    b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=1, embedder=aemb))
    r = asyncio.run(b.run(wl.texts))
    assert r.artifact.dense and r.diagnostics["dense"] == "dense"


def test_dense_pour_validity_simulation():
    """Dense cold pours keep the per-batch guarantee: realized disagreement > eps in at most ~delta of batches."""
    wl, client = _toy(1300, seed=21, acc={"cheap": 0.8, "expert": 0.93})
    E = hashed_embed(wl.texts)
    base = asyncio.run(Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=0))
                       .run(wl.texts[:300], emb=E[:300]))
    expert_all = [client._answer("expert", t, "single") for t in wl.texts[300:]]
    R, viol, cov = 30, 0, []
    for r in range(R):
        b = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", seed=100 + r, export=False))
        res = asyncio.run(b.pour(base.artifact, wl.texts[300:], votes=False, emb=E[300:]))
        viol += evaluate(res, None, expert_all)["realized_disagreement_with_expert"] > 0.05
        cov.append(res.coverage())
    assert viol / R <= 0.1 + 0.1 and np.mean(cov) > 0.1


# ----------------------------------------------------------------------------------------------- client + CLI
def _mock_vec(t, dim):
    h = int(hashlib.md5(str(t).encode()).hexdigest(), 16)
    return [float((h >> (4 * j)) % 7 + 1) for j in range(dim)]       # positive, never zero, unnormalised


def _emb_response(inputs, dim=6, shuffle=True):
    data = [{"index": i, "embedding": _mock_vec(t, dim)} for i, t in enumerate(inputs)]
    if shuffle:
        data = data[::-1]
    return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 7 * len(inputs)}})


def test_client_embed_mock_transport(tmp_path, monkeypatch):
    async def _noop(*_):
        return None
    monkeypatch.setattr(asyncio, "sleep", _noop)
    seen = []

    def h(req):
        assert req.url.path.endswith("/embeddings")
        body = json.loads(req.content)
        seen.append(len(body["input"]))
        return _emb_response(body["input"])
    spec = ModelSpec("emb", ["mock-embedder"], base_url="http://mock.invalid/v1", price_in=1.0)
    c = LLMClient({"emb": spec}, cache_path=tmp_path / "c.sqlite")
    c.http = httpx.AsyncClient(transport=httpx.MockTransport(h))
    texts = [f"text number {i} word{i % 5}" for i in range(150)]
    E = asyncio.run(c.embed("emb", texts, batch_size=64))
    assert E.shape == (150, 6) and E.dtype == np.float32 and sorted(seen) == [22, 64, 64]
    assert np.allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)          # normalised by default
    ref = np.array([_mock_vec(t, 6) for t in texts])
    assert np.allclose(E, ref / np.linalg.norm(ref, axis=1, keepdims=True), atol=1e-5)   # index order restored
    row = c.ledger.as_dict()["embed"]
    assert row["calls"] == 3 and row["prompt_tokens"] == 7 * 150 and row["usd"] == pytest.approx(7 * 150 / 1e6)
    E2 = asyncio.run(c.embed("emb", texts[:10] + ["a new one"]))           # cache: only the new text is requested
    assert seen[-1] == 1 and np.allclose(E2[:10], E[:10])
    c2 = LLMClient({"emb": spec})
    c2.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(404, text="no such model")))
    with pytest.raises(EmbeddingError, match="404"):
        asyncio.run(c2.embed("emb", ["x"]))
    c3 = LLMClient({"emb": spec})
    c3.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"data": []})))
    with pytest.raises(EmbeddingError):
        asyncio.run(c3.embed("emb", ["x", "y"]))


def _label_of(text):
    return "billing" if ("invoice" in text or "refund" in text) else "shipping"


def _cli_handler(counter):
    def h(req):
        body = json.loads(req.content)
        if req.url.path.endswith("/embeddings"):
            counter["embed"] += 1
            return _emb_response(body["input"], dim=8)
        counter["chat"] += 1
        user = body["messages"][-1]["content"]
        if user.startswith("1. "):
            parts = [p.split(". ", 1)[1] for p in user.split("\n\n")]
            text = "\n".join(f"{i + 1}: {_label_of(t)}" for i, t in enumerate(parts))
        else:
            text = _label_of(user)
        return httpx.Response(200, json={"choices": [{"message": {"content": text}}],
                                         "usage": {"prompt_tokens": 20, "completion_tokens": 2}})
    return h


def test_cli_embed_model_run_and_pour_with_mock_transport(tmp_path, monkeypatch):
    from bottlekit import __main__ as cli
    rng = random.Random(0)
    words = {"billing": ["invoice", "refund"], "shipping": ["parcel", "courier"]}
    recs = []
    for i in range(700):
        lab = rng.choice(["billing", "shipping"])
        recs.append({"text": f"{rng.choice(words[lab])} {rng.choice(['hello', 'please', 'urgent'])} order{i}", "label": lab})
    (tmp_path / "items.jsonl").write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    (tmp_path / "models.yaml").write_text(
        "defaults: {base_url: 'http://mock.invalid/v1', max_concurrency: 8}\n"
        "models:\n  c: {id: cheap-m, price_in: 0.05, price_out: 0.1}\n"
        "  e: {id: expert-m, price_in: 2.0, price_out: 8.0}\n  emb: {id: mock-embedder, price_in: 0.02}\n",
        encoding="utf-8")
    counter = {"embed": 0, "chat": 0}
    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(_cli_handler(counter)), **kw))
    common = ["--jsonl", str(tmp_path / "items.jsonl"), "--labels", "billing,shipping", "--expert", "e",
              "--models", str(tmp_path / "models.yaml"), "--label-key", ""]

    def run_cli(args):
        monkeypatch.setattr(sys, "argv", ["bottlekit"] + args)
        cli.main()
    run_cli(["run", "--cheap", "c", "--n", "400", "--embed-model", "emb", "--save-artifact", str(tmp_path / "b0.pkl"),
             "--out", str(tmp_path / "r0.json")] + common)
    r0 = json.loads((tmp_path / "r0.json").read_text(encoding="utf-8"))
    assert r0["artifact"]["dense"] and r0["artifact"]["embedder_id"] == "mock-embedder" and r0["artifact"]["emb_dim"] == 8
    assert r0["ledger"]["embed"]["calls"] >= 1 and r0["ledger"]["embed"]["usd"] > 0
    assert r0["diagnostics"]["dense"] == "dense"
    run_cli(["pour", "--artifact", str(tmp_path / "b0.pkl"), "--offset", "400", "--n", "300", "--votes", "no",
             "--embed-model", "emb", "--save-artifact", str(tmp_path / "b1.pkl"), "--out", str(tmp_path / "r1.json")] + common)
    r1 = json.loads((tmp_path / "r1.json").read_text(encoding="utf-8"))
    assert r1["diagnostics"]["dense"] == "dense" and r1["artifact"]["dense"] and r1["artifact"]["version"] == 1
    assert "embed" in r1["ledger"] and not any(k.startswith("cheap") for k in r1["ledger"])
    # off by default: no --embed-model -> no embedding request at all, TF-IDF bottle
    n_embed = counter["embed"]
    run_cli(["run", "--cheap", "c", "--n", "300", "--out", str(tmp_path / "r2.json")] + common)
    r2 = json.loads((tmp_path / "r2.json").read_text(encoding="utf-8"))
    assert counter["embed"] == n_embed and "embed" not in r2["ledger"] and "dense" not in r2["diagnostics"]
    with pytest.raises(SystemExit):
        run_cli(["run", "--cheap", "c", "--n", "100", "--embed-model", "nope"] + common)


def test_failed_memory_reembedding_falls_back_to_tfidf():
    """If the embedder works for the new batch but fails to re-embed a dense bottle's memory (width
    mismatch), pour() degrades to TF-IDF features with a warning instead of raising."""
    wl, client = _toy(1000)
    b0 = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=3,
                                            embedder=hashed_embed, embedder_id="h16"))
    r0 = asyncio.run(b0.run(wl.texts[:400]))
    assert r0.artifact.dense and r0.artifact.emb_dim == DIM
    calls = []

    def flaky32(texts):
        calls.append(len(texts))
        if len(calls) > 1:
            raise RuntimeError("embedding service down")
        return hashed_embed(texts, 32)
    b1 = Bottler(client, SPEC, BottleConfig(cheap="cheap", expert="expert", seed=3, embedder=flaky32, embedder_id="h32"))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r1 = asyncio.run(b1.pour(r0.artifact, wl.texts[400:700], votes=False))
    assert "embedder_error" in r1.diagnostics and "dense_fallback" in r1.diagnostics
    assert len(r1.answers) == 300 and r1.coverage() >= 0.0
