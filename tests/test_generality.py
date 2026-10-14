"""Generality pass (v0.9): every new knob defaults to the v0.8 value, so outputs, hashes and bytes stay identical.

Pins in tests/data/v08_pins.json and the v0.8 session checkpoint in tests/data/v08_session/ were produced by the FROZEN
v0.8 code; these tests compare the current package to
them. Mock LLM only, no servers.
  (0) defaults = the v0.8 literals; config_hash / fingerprint / prompts / parser / save() bytes / pipeline outputs pinned
  (0) the config-hash trap: v0.9 fields at their default are not hashed; a v0.8 checkpoint resumes bit for bit
  (4) CLI defaults = BottleConfig defaults; every field has a flag
  (2) non-default knobs run end to end on the mock client
  (3) a non-default parser or prompt changes the fingerprint (memo off); hooks parse what they should
"""

import argparse
import asyncio
import base64
import dataclasses
import hashlib
import inspect
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
import bottlekit.certify  # noqa: E402,F401
C = sys.modules["bottlekit.certify"]       # the package attribute `certify` is the function
import bottlekit.economics as EC  # noqa: E402
import bottlekit.harness as H  # noqa: E402
import bottlekit.labeling as LB  # noqa: E402
import bottlekit.students as ST  # noqa: E402
import bottlekit.tasks as T  # noqa: E402
from bottlekit import (BottleArtifact, BottleConfig, Bottler, BottleSession, LLMClient, ModelSpec, TaskSpec,  # noqa: E402
                       batch_tail, cross_fit, min_certified_size, parse_batch, parse_label)
from bottlekit.economics import gain_prior_for, per_item_costs  # noqa: E402
from bottlekit.labeling import label_batched, single_max_tokens  # noqa: E402
from bottlekit.session import _norm, config_hash  # noqa: E402
from bottlekit.tasks import batch_max_tokens, batch_messages, single_messages  # noqa: E402
from test_bottlekit import SPEC, MockClient, toy_workload  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
PINS = json.loads((DATA / "v08_pins.json").read_text(encoding="utf-8"))
WL = toy_workload(1400, seed=77)
GOLD = {t[:400]: g for t, g in zip(WL.texts, WL.gold)}
LETTERS = TaskSpec("esci", "Judge.", {"E": "Exact: match", "S": "Substitute: partial", "C": "Complement: goes with",
                                      "I": "Irrelevant: no"})
V09_FIELDS = ("student", "crossfit_folds", "max_memory", "n_build_min", "n_build_per_class", "gain_fed_per_class",
              "gain_starved_per_class", "gain_starved_mult", "early_frac", "forecast_hysteresis", "forecast_gain_decay",
              "forecast_fit_budget", "forecast_max_folds", "forecast_max_rows", "forecast_canpay_z", "cost_basis")
NON_DEFAULT = {"student": {"C": 1.0}, "crossfit_folds": 3, "max_memory": 5000, "n_build_min": 100,
               "n_build_per_class": 5, "gain_fed_per_class": 30, "gain_starved_per_class": 2, "gain_starved_mult": 3.0,
               "early_frac": 0.2, "forecast_hysteresis": 0.01, "forecast_gain_decay": 0.5, "forecast_fit_budget": 3,
               "forecast_max_folds": 4, "forecast_max_rows": 1000, "forecast_canpay_z": 1.0, "cost_basis": "tokens"}


def client(acc=None):
    return MockClient(GOLD, acc=acc or {"cheap": 0.82, "expert": 0.95})


def j(o):
    return json.dumps(_norm(o), sort_keys=True)


def sha(s):
    return hashlib.sha256(s.encode("utf-8") if isinstance(s, str) else s).hexdigest()


def cfg_from_pin(name):
    kw = dict(PINS["cfg_kwargs"][name])
    return BottleConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()})


def view(res):
    d = dict(res.diagnostics)
    d.pop("wall_time_s", None)
    return j({"answers": res.answers, "provenance": res.provenance, "certificate": res.certificate,
              "ledger": res.ledger, "diagnostics": d})


# ----------------------------------------------------------------------------------------------- (0) defaults
def test_defaults_equal_the_v08_literals_and_duplicates_have_one_source():
    c = BottleConfig(cheap="c", expert="e")
    v08 = dict(mode="auto", k=20, contexts=3, warmup=0, balanced=True, weighted=True, fusion="prob", n_build=None,
               eps=0.05, delta=0.1, calibration="sequential", certificate="rectified", n_cal=300, looks=None,
               spending=None, design="single", design_c=1.25, tune=True, gain_prior=0.15, max_expert_none=0.2,
               memo=False, memo_allowance="novel", forecast=True, forecast_run=False, chunking="cold",
               chunk_cold_ratio=4.0, chunk_tax=0.30, chunk_growth=1.0, max_chunks=None, forecast_v2="full",
               forecast_folds="auto")
    assert {k: getattr(c, k) for k in v08} == v08
    assert {k: getattr(c, k) for k in V09_FIELDS} == {
        "student": None, "crossfit_folds": 5, "max_memory": 200_000, "n_build_min": 600, "n_build_per_class": 20,
        "gain_fed_per_class": 40, "gain_starved_per_class": 5, "gain_starved_mult": 2.0, "early_frac": 0.25,
        "forecast_hysteresis": 0.005, "forecast_gain_decay": 0.8, "forecast_fit_budget": 2, "forecast_max_folds": 5,
        "forecast_max_rows": None, "forecast_canpay_z": 0.0, "cost_basis": "usd"}
    # the generality fields are ONE contiguous block; every field after the v0.8 ones is tagged since=0.9 (later
    # v0.9 report-only fields, e.g. the health report's, may follow the generality block)
    names = [f.name for f in dataclasses.fields(BottleConfig)]
    i0 = names.index(V09_FIELDS[0])
    assert tuple(names[i0:i0 + len(V09_FIELDS)]) == V09_FIELDS
    tagged = [f.name for f in dataclasses.fields(BottleConfig) if f.metadata.get("since")]
    assert tagged == names[i0:] and set(V09_FIELDS) <= set(tagged)
    # one definition each
    sig = lambda f, p: inspect.signature(f).parameters[p].default  # noqa: E731
    assert C.DEFAULT_DESIGN_C == 1.25 and c.design_c == C.DEFAULT_DESIGN_C
    for f, p in ((C.single_look_design, "c"), (C.single_look_size, "design_c"), (C.min_certified_size, "design_c"),
                 (C.min_batch_for_tax, "design_c")):
        assert sig(f, p) == C.DEFAULT_DESIGN_C
    assert sig(C.tune_size, "window") == sig(C.min_certified_size, "window") == C.TUNE_WINDOW == 0.1
    assert c.chunk_tax == C.DEFAULT_MIN_TAX == sys.modules["bottlekit.session"].DEFAULT_TAX == 0.30
    assert sig(EC.expected_cost, "n_build") == EC.N_BUILD_MIN == c.n_build_min == 600
    assert sig(cross_fit, "folds") == ST.CROSS_FIT_FOLDS == H.XFIT_MAX_FOLDS == c.crossfit_folds == 5
    assert sig(BottleArtifact.__init__, "C") == sig(ST.TfidfLR.__init__, "C") == ST.DEFAULT_C == 4.0
    assert sig(BottleArtifact.__init__, "max_memory") == H.DEFAULT_MAX_MEMORY == c.max_memory
    assert sig(LB.balanced_orders, "early_frac") == sig(LB.vote_label, "early_frac") == c.early_frac
    assert LB.n_bins_for(c.early_frac) == sig(LB.position_reliability, "n_bins") == 4
    assert C.report_deltas(0.1) == (0.5, 0.1, 0.05, 0.01) and C.report_deltas() == (0.5, 0.05, 0.01)
    assert LB.STREAM_KEYS == {"calibration": 1, "crossfit": 2, "forecast": 7, "contexts": 11, "partition": 13,
                              "health": 19, "reseed": 0x5E55}
    assert H.FORECAST_STREAM_KEY == 7 and H.PARTITION_STREAM_KEY == 13
    assert sys.modules["bottlekit.session"].RESEED_TAG == 0x5E55
    assert H.cal_stream_key(4) == (4, 1)
    art = BottleArtifact(SPEC, WL.texts[:30], WL.gold[:30], version=2)
    assert H.cal_stream_key(4, art) == (4, 1, 3, 30)


def test_config_hash_is_pinned_and_skips_v09_fields_at_their_default():
    for name in PINS["config_hash"]:
        assert config_hash(cfg_from_pin(name)) == PINS["config_hash"][name], name
    base = BottleConfig(cheap="c", expert="e")
    h0 = config_hash(base)
    seen = {h0}
    for k, v in NON_DEFAULT.items():
        h = config_hash(dataclasses.replace(base, **{k: v}))
        assert h not in seen, k                # every non-default v0.9 knob enters the hash
        seen.add(h)
    # back at the default (another object, equal value): the v0.8 hash again
    assert config_hash(dataclasses.replace(base, student=None, cost_basis="usd", crossfit_folds=5)) == h0


def test_fingerprints_prompts_and_parser_are_pinned():
    c = client()
    assert Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert")).fingerprint() == PINS["fingerprint"]["mock_default"]
    c2 = client()
    c2.models["expert"] = ModelSpec("expert", ["x1", "x2"], price_in=2.0, price_out=8.0, max_tokens=300,
                                    extra_body={"reasoning_effort": "low"}, system_prefix="/no_think")
    assert Bottler(c2, SPEC, BottleConfig(cheap="cheap", expert="expert")).fingerprint() == \
        PINS["fingerprint"]["mock_reasoning"]
    assert Bottler(c, LETTERS, BottleConfig(cheap="cheap", expert="expert")).fingerprint() == PINS["fingerprint"]["letters"]
    t1, t2 = WL.texts[0], "  multi\n\nline   item " + "x" * 900
    pr = PINS["prompts"]
    got = {"single": single_messages(SPEC, t1), "single_long": single_messages(SPEC, t2),
           "batch": batch_messages(SPEC, [t1, t2]), "batch_letters": batch_messages(LETTERS, [t1]),
           "batch_max_tokens": [batch_max_tokens(SPEC, n) for n in (1, 5, 20, 25)],
           "batch_max_tokens_letters": batch_max_tokens(LETTERS, 20),
           "single_max_tokens": [single_max_tokens(c, "expert", SPEC), single_max_tokens(c2, "expert", SPEC)],
           "max_label_tokens": [SPEC.max_label_tokens, LETTERS.max_label_tokens],
           "per_item_costs": [list(per_item_costs(SPEC, c.models["cheap"], c.models["expert"], a, k, w))
                              for a, k, w in ((80.0, 20, 0), (900.0, 25, 2))],
           "per_item_costs_reasoning": list(per_item_costs(SPEC, c.models["cheap"], c2.models["expert"], 120.0, 20, 0)),
           "gain_prior_for": [gain_prior_for(0.15, nb, K) for nb in (100, 600, 3000) for K in (2, 4, 20, 77, 150)]}
    assert j(got) == j(pr)
    REPLIES = ["alpha", "  Beta. ", "**gamma**", "delta x", "delta_x", "The answer is beta", "", "unknown",
               "alpha or beta", "Answer: gamma", "final label = beta", "<think>hmm alpha</think>gamma",
               "reasoning...</think> beta", "Category - delta x", "Etiqueta: beta", "答案：gamma",
               "LABEL: alpha\nexplanation", "x\nbeta", "`alpha`", "Output: 'beta'"]
    LREPLIES = ["Substitute", "E", "irrelevant", "S.", "(C)", "the answer is I", "e", "E or S", "Exact", "Ι",
                "Answer: S"]
    BATCH = ["1: alpha\n**2**: beta\n3) gamma\n4 - delta x\nnoise line\n5: zzz", "1: alpha, 2: beta, 3: gamma",
             "alpha\nbeta\ngamma", "Item 1: beta\n#2 gamma\n[3] alpha", "1：alpha\n2：beta\n3：gamma",
             "<think>1: alpha</think>\n1: beta\n2: gamma\n3: delta x"]
    assert [parse_label(SPEC, r) for r in REPLIES] == PINS["parse_label"]
    assert [parse_label(LETTERS, r) for r in LREPLIES] == PINS["parse_label_letters"]
    assert [parse_batch(SPEC, r, 3) for r in BATCH] == PINS["parse_batch"]
    assert T.parser_config(SPEC) == {} and T.spec_extras(SPEC) == {}
    assert T.spec_to_plain(SPEC) == {"name": SPEC.name, "instruction": SPEC.instruction, "labels": SPEC.labels,
                                     "max_chars": SPEC.max_chars}


def test_save_bytes_and_default_pipeline_outputs_are_pinned(tmp_path):
    a0 = BottleArtifact(SPEC, WL.texts[:200], WL.gold[:200])
    a0.save(tmp_path / "a0.pkl")
    # health=False: the v0.9 health report adds only report-only data (diagnostics['health'] and a history row in the
    # bottle); its on/off byte-identity of answers, certificates and ledgers is tested in test_health.py
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=3, health=False)
    bot = Bottler(client(), SPEC, cfg)
    r = asyncio.run(bot.run(WL.texts[:300]))
    r.artifact.save(tmp_path / "a1.pkl")
    same_protocol = pickle.DEFAULT_PROTOCOL == PINS["python_pickle_protocol"]
    for name, path in (("handbuilt", "a0.pkl"), ("run", "a1.pkl")):
        got, ref = (tmp_path / path).read_bytes(), base64.b64decode(PINS["save"][name + "_b64"])
        assert pickle.loads(got) == pickle.loads(ref), name       # same payload (keys, order-free values)
        assert list(pickle.loads(got)) == list(pickle.loads(ref))  # same key order
        if same_protocol:
            assert got == ref and sha(got) == PINS["save"][name + "_sha256"], name
    pw = asyncio.run(bot.pour(r.artifact, WL.texts[300:500], votes=True))
    pc = asyncio.run(bot.pour(pw.artifact, WL.texts[500:700], votes=False))
    rs = asyncio.run(Bottler(client(), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="student", seed=4, health=False)
                             ).run(WL.texts[700:1000]))
    pin = PINS["pipeline"]
    assert sha(view(r)) == pin["run_fused"]
    assert sha(view(pw)) == pin["pour_warm"]
    assert sha(view(pc)) == pin["pour_cold"]
    assert sha(view(rs)) == pin["run_student"]
    assert sha(j(pc.artifact.forecast)) == pin["pour_cold_artifact_forecast"]


# ----------------------------------------------------------------------------------------------- (0) v0.8 checkpoint
def _copy_session(tmp_path):
    d = tmp_path / "v08"
    shutil.copytree(DATA / "v08_session", d)
    return d / "stream.json"


def test_v08_checkpoint_resumes_bit_for_bit(tmp_path):
    p = _copy_session(tmp_path)
    cfg = dataclasses.replace(cfg_from_pin("session"), health=False)   # report-only: not in config_hash
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # the recorded lineage path is the generator's temp dir
        s = BottleSession.resume(p, client(), SPEC, cfg)
    assert len(s.batches) == 2 and s.artifact is not None
    lo, hi = PINS["session"]["subs"][2]
    asyncio.run(s.submit(WL.texts[lo:hi]))
    rec = s.batches[2]
    got = {k: rec[k] for k in ("answers", "provenance", "certificate", "ledger", "kind", "N", "submissions", "streams")}
    got["diagnostics"] = {k: v for k, v in rec["diagnostics"].items() if k != "wall_time_s"}
    exp = json.loads((DATA / "v08_session" / "expected_batch3.json").read_text(encoding="utf-8"))
    assert j(got) == json.dumps(exp, sort_keys=True)
    art = s.artifact
    assert sha(j({"v": art.version, "texts": art.texts, "labels": art.labels, "src": list(art.src),
                  "fp": art.fingerprint, "forecast": art.forecast})) == PINS["session"]["artifact_mem_sha"]
    # the v0.8 bottle bytes in the checkpoint load and re-save to the same payload
    st = json.loads((DATA / "v08_session" / "stream.json").read_text(encoding="utf-8"))
    blob = base64.b64decode(st["artifact"]["b64"])
    (tmp_path / "old.pkl").write_bytes(blob)
    old = BottleArtifact.load(tmp_path / "old.pkl")
    old.save(tmp_path / "again.pkl")
    again = (tmp_path / "again.pkl").read_bytes()
    assert pickle.loads(again) == pickle.loads(blob) and list(pickle.loads(again)) == list(pickle.loads(blob))
    if pickle.DEFAULT_PROTOCOL == PINS["python_pickle_protocol"]:   # v0.8's own re-save bytes (pickle memo sharing
        assert sha(again) == PINS["session"]["resave_sha256"]       # differs from the original's, in v0.8 too)
    assert old.student_cfg is None and old.max_memory == 200_000


def test_v08_checkpoint_refuses_a_non_default_v09_knob(tmp_path):
    p = _copy_session(tmp_path)
    cfg = dataclasses.replace(cfg_from_pin("session"), early_frac=0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="cfg hash"):
            BottleSession.resume(p, client(), SPEC, cfg)


# ----------------------------------------------------------------------------------------------- (4) CLI
def _cli_parser():
    from bottlekit.__main__ import _common
    ap = argparse.ArgumentParser()
    _common(ap)
    return ap


def test_cli_defaults_equal_bottleconfig_defaults_and_every_field_has_a_flag():
    from bottlekit.__main__ import _CFG_FLAGS, _CFG_SWITCHES, CLI_SKIPPED_FIELDS, _cfg
    ap = _cli_parser()
    a = ap.parse_args(["--expert", "e"])
    assert _cfg(a) == BottleConfig(cheap=None, expert="e")
    for _flag, dest, *_ in _CFG_FLAGS + _CFG_SWITCHES:
        assert ap.get_default(dest) is None, dest               # no re-declared default
    covered = {f[2] for f in _CFG_FLAGS} | {s[2] for s in _CFG_SWITCHES} | {"cheap", "expert", "mode", "student",
                                                                          "embedder_id"}
    missing = {f.name for f in dataclasses.fields(BottleConfig)} - covered - set(CLI_SKIPPED_FIELDS)
    assert not missing, missing
    a = ap.parse_args(["--expert", "e", "--crossfit-folds", "3", "--memo", "--no-forecast", "--student-c", "1.5",
                       "--student", '{"char": false}', "--looks", "100,200", "--spending", "1,4",
                       "--forecast-folds", "3", "--cost-basis", "tokens", "--early-frac", "0.2", "--max-memory", "999",
                       "--design-c", "1.0", "--chunking", "off", "--forecast-max-rows", "50", "--embed-id", "m1",
                       "--budget", "40", "--no-balance"])
    c = _cfg(a)
    assert (c.crossfit_folds, c.memo, c.forecast, c.student, c.looks, c.spending, c.forecast_folds, c.cost_basis,
            c.early_frac, c.max_memory, c.design_c, c.chunking, c.forecast_max_rows, c.embedder_id, c.expert_budget,
            c.balanced) == (3, True, False, {"char": False, "C": 1.5}, (100, 200), (1.0, 4.0), 3, "tokens", 0.2, 999,
                            1.0, "off", 50, "m1", 40, False)


def test_cli_main_parser_builds_and_run_mode_defaults():
    from bottlekit.__main__ import _cfg
    from bottlekit import __main__ as cli
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run")
    cli._common(r)
    r.add_argument("--cheap", required=True)
    r.add_argument("--mode", default=None)
    a = ap.parse_args(["run", "--expert", "e", "--cheap", "c"])
    assert _cfg(a) == BottleConfig(cheap="c", expert="e")


# ----------------------------------------------------------------------------------------------- (2) knobs end to end
def test_non_default_knobs_run_end_to_end_on_the_mock_client(tmp_path):
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", seed=5,
                       student={"C": 1.0, "char_analyzer": "char", "word_ngrams": [1, 1], "max_iter": 500},
                       crossfit_folds=3, early_frac=0.2, max_memory=650, forecast_hysteresis=0.05,
                       forecast_gain_decay=0.5, forecast_canpay_z=1.0, forecast_max_rows=50, forecast_fit_budget=2.5,
                       forecast_max_folds=3, cost_basis="tokens")
    c = client()
    bot = Bottler(c, SPEC, cfg)
    r = asyncio.run(bot.run(WL.texts[:300]))
    assert r.certificate["certified"] and len(r.answers) == 300
    art = r.artifact
    assert art.C == 1.0 and art.student_cfg == {"char_analyzer": "char", "word_ngrams": [1, 1], "max_iter": 500}
    assert art.max_memory == 650 and art.student.char_analyzer == "char" and art.student.word_ngrams == (1, 1)
    art.save(tmp_path / "b.pkl")
    back = BottleArtifact.load(tmp_path / "b.pkl")
    assert back.C == 1.0 and back.student_cfg == art.student_cfg and back.max_memory == 650
    assert pickle.loads((tmp_path / "b.pkl").read_bytes())["student"] == art.student_cfg
    pw = asyncio.run(bot.pour(art, WL.texts[300:600], votes=True))
    f = pw.artifact.forecast
    assert f["costs"]["vote_tokens_per_item"] > 0 and f["costs"]["expert_tokens_per_call"] > 0
    pa = asyncio.run(bot.pour(pw.artifact, WL.texts[600:900], votes="auto"))
    why = pa.diagnostics["votes_decision"]
    assert why.get("basis") in ("measured", "can_pay_bound", "prior")
    if why.get("basis") == "measured":
        assert why["hysteresis"] == pytest.approx(0.05 * why["c_exp_call"])
    xf = why.get("xfit")
    if isinstance(xf, dict):
        assert xf["n_other_rows"] <= 50 and xf["folds"] <= 3
    assert len(pa.artifact.texts) <= 650 and pa.artifact.student_cfg == art.student_cfg
    # a DEFAULT Bottler pouring this bottle keeps the bottle's own student and cap
    p2 = asyncio.run(Bottler(client(), SPEC, BottleConfig(cheap="cheap", expert="expert", seed=6)).pour(
        pa.artifact, WL.texts[900:1100], votes=False))
    assert p2.artifact.student_cfg == art.student_cfg and p2.artifact.C == 1.0 and p2.artifact.max_memory == 650
    # student mode with the n_build knobs (auto mode with a price ratio that selects student)
    rs = asyncio.run(Bottler(client(), SPEC, BottleConfig(cheap="cheap", expert="expert", mode="student", seed=7,
                                                          n_build_min=120, n_build_per_class=10)).run(WL.texts[:400]))
    assert rs.diagnostics["n_build"] == 120
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", student={"bogus": 1})
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", crossfit_folds=1)
    with pytest.raises(ValueError):
        BottleConfig(cheap="c", expert="e", cost_basis="eur")


def test_gain_prior_knobs_and_zero_price_warning():
    assert gain_prior_for(0.15, 600, 150) == 0.3 and gain_prior_for(0.15, 600, 4) == 0.15
    assert gain_prior_for(0.15, 600, 150, mult=3.0) == pytest.approx(0.45)
    assert gain_prior_for(0.15, 600, 30, fed=20, starved=10) == pytest.approx(0.15)
    c = client()
    c.models["expert"] = ModelSpec("expert", ["expert"])         # no price (a local model)
    b = Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert"))
    with pytest.warns(UserWarning, match="no price"):
        mode, info = b.plan(WL.texts[:100])
    assert mode == "student" and info["price_warning"] == "no_prices"
    bt = Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert", cost_basis="tokens"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mode_t, info_t = bt.plan(WL.texts[:100])
    assert "price_warning" not in info_t and np.isfinite(info_t["ratio"])
    with pytest.warns(UserWarning, match="no price"):
        use, why = b._votes_pay(WL.texts[:100])
    assert use is False and why["price_warning"] == "no_prices"
    # priced models: no flag (default outputs unchanged)
    _, info_p = Bottler(client(), SPEC, BottleConfig(cheap="cheap", expert="expert")).plan(WL.texts[:100])
    assert "price_warning" not in info_p


# ----------------------------------------------------------------------------------------------- (3) hooks
def test_non_default_parser_or_prompt_changes_the_fingerprint_and_disables_the_memo(tmp_path):
    c = client()
    fp0 = Bottler(c, SPEC, BottleConfig(cheap="cheap", expert="expert")).fingerprint()
    variants = {"prompt_single": "{instruction}\nEtiquetas:\n{labels}\nResponde solo con la etiqueta.",
                "answer_prefixes": ("etiqueta", "respuesta"), "think_pattern": r"<razonamiento>.*?</razonamiento>",
                "letter_class": "A-ZΑ-Ω", "fullwidth_punct": True, "single_chars_factor": 3,
                "chars_per_token": 1.2}
    fps = set()
    for k, v in variants.items():
        sp = dataclasses.replace(SPEC, **{k: v})
        fp = Bottler(c, sp, BottleConfig(cheap="cheap", expert="expert")).fingerprint()
        assert fp != fp0 and fp not in fps, k
        fps.add(fp)
    # the cheap-side batch template is not part of the expert request
    sp_b = dataclasses.replace(SPEC, prompt_batch="{instruction}\n{labels}\nOne line per item: '<n>: <label>'.")
    assert Bottler(c, sp_b, BottleConfig(cheap="cheap", expert="expert")).fingerprint() == fp0
    assert batch_messages(sp_b, ["x"])[0]["content"] != batch_messages(SPEC, ["x"])[0]["content"]
    # memo: a bottle built under the default parser is not reused under another parser
    cfg = BottleConfig(cheap="cheap", expert="expert", mode="fused", memo=True, seed=8)
    r = asyncio.run(Bottler(client(), SPEC, cfg).run(WL.texts[:300]))
    same = asyncio.run(Bottler(client(), SPEC, cfg).pour(r.artifact, WL.texts[:200], votes=False))
    assert same.diagnostics.get("memo") == "on" and same.diagnostics["n_memo"] > 0
    sp_p = dataclasses.replace(SPEC, answer_prefixes=("etiqueta",))
    other = asyncio.run(Bottler(client(), sp_p, cfg).pour(r.artifact, WL.texts[:200], votes=False))
    assert str(other.diagnostics.get("memo", "")).startswith("off: prompt fingerprint mismatch")
    # a non-default spec round-trips through a bottle, a default one saves the v0.8 four keys only
    a = BottleArtifact(sp_p, WL.texts[:50], WL.gold[:50])
    assert T.spec_to_plain(sp_p)["answer_prefixes"] == ["etiqueta"]
    a.save(tmp_path / "spec.pkl")
    assert BottleArtifact.load(tmp_path / "spec.pkl").spec == sp_p


def test_language_hooks_parse_what_they_should():
    # think pattern: another reasoning wrapper
    rz = dataclasses.replace(SPEC, think_pattern=r"<razonamiento>.*?</razonamiento>")
    reply = "<razonamiento>alpha</razonamiento>beta"
    assert parse_label(SPEC, reply) is None and parse_label(rz, reply) == "beta"
    # answer prefixes (regex level; the default regex is the v0.8 one)
    es = dataclasses.replace(SPEC, answer_prefixes=("etiqueta", "respuesta final"))
    assert T._prefix_rx(SPEC) == T._PREFIX
    import re
    assert re.sub(T._prefix_rx(es), "", "Respuesta  final: beta") == "beta"
    assert re.sub(T._prefix_rx(es), "", "Answer: beta") == "Answer: beta"
    # letter class: Greek single-letter labels
    gr = TaskSpec("gr", "Judge.", {"Α": "first", "Β": "second", "Γ": "third"})
    gr2 = dataclasses.replace(gr, letter_class="Α-Ω")
    r = "η απάντηση είναι Β"
    assert parse_label(gr, r) is None and parse_label(gr2, r) == "Β"
    # full-width punctuation after item numbers
    fw = dataclasses.replace(SPEC, fullwidth_punct=True)
    assert parse_batch(SPEC, "1：beta\n3：alpha", 3) == [None, None, None]
    assert parse_batch(fw, "1：beta\n3：alpha", 3) == ["beta", None, "alpha"]
    # prompt templates and the expert truncation factor
    sp = dataclasses.replace(SPEC, prompt_single="{instruction} | {labels}", single_chars_factor=1)
    m = single_messages(sp, "y" * 2000)
    assert m[0]["content"] == f"{SPEC.instruction} | {SPEC.label_block()}" and len(m[1]["content"]) == SPEC.max_chars
    assert H.expert_key(sp, "y" * 2000) != H.expert_key(SPEC, "y" * 2000)
    # chars per token: label reply budget for CJK-like ratios (default: the v0.8 // 3 + 2)
    cjk = TaskSpec("zh", "x", {"体育新闻": "a", "财经": "b"}, chars_per_token=1.0)
    assert cjk.max_label_tokens == 4 + 2 and TaskSpec("zh", "x", dict(cjk.labels)).max_label_tokens == 4 // 3 + 2
    with pytest.raises(ValueError):
        TaskSpec("x", "y", {"a": "a", "b": "b"}, chars_per_token=0)


def test_model_and_client_hooks():
    seen = []

    class Spy(MockClient):
        async def chat(self, alias, messages, max_tokens=256, **kw):
            seen.append((alias, max_tokens))
            return await super().chat(alias, messages, max_tokens=max_tokens, **kw)
    c = Spy(GOLD)
    texts = WL.texts[:40]
    asyncio.run(label_batched(c, "cheap", SPEC, texts, list(range(40)), 20))
    base = {mt for _, mt in seen}
    assert base == {batch_max_tokens(SPEC, 20)}
    seen.clear()
    c.models["cheap"] = ModelSpec("cheap", ["cheap"], price_in=0.05, price_out=0.1, batch_extra_tokens=300)
    asyncio.run(label_batched(c, "cheap", SPEC, texts, list(range(40)), 20))
    assert {mt for _, mt in seen} == {batch_max_tokens(SPEC, 20) + 300}
    # expected_reply_tokens overrides the 0.5 x max_tokens reasoning share in cost estimates only
    e1 = ModelSpec("e", ["e"], price_in=1.0, price_out=1.0, max_tokens=300)
    e2 = ModelSpec("e", ["e"], price_in=1.0, price_out=1.0, max_tokens=300, expected_reply_tokens=150)
    ch = ModelSpec("c", ["c"], price_in=1.0, price_out=1.0)
    assert per_item_costs(SPEC, ch, e1, 100.0, 20) == per_item_costs(SPEC, ch, e2, 100.0, 20)
    # LLMClient: gateway headers and retry policy are options with the v0.8 defaults
    lc = LLMClient({"a": ModelSpec("a", ["m"])})
    assert lc._headers(lc.models["a"], "p") == {"Content-Type": "application/json", "X-Exp": "p"}
    lc2 = LLMClient({"a": ModelSpec("a", ["m"])}, phase_header=None, cache_header=None, backoff_cap=1.0,
                    non_retryable=(400,), map_limit=2)
    assert lc2._headers(lc2.models["a"], "p") == {"Content-Type": "application/json"}
    assert sys.modules["bottlekit.llm"]._backoff(10, 1.0) == 1.0 and lc2.non_retryable == (400,)

    async def one(i):
        return i
    assert asyncio.run(lc2.map([one(i) for i in range(5)])) == list(range(5))
    asyncio.run(lc.close())
    asyncio.run(lc2.close())


def test_batch_tail_uses_delta_and_min_size_threads_the_window():
    designs = [(60, 300, 5), (80, 500, 8)]
    d0 = batch_tail(designs, 800, 0.05)
    assert d0 == batch_tail(designs, 800, 0.05, delta=0.1) == batch_tail(designs, 800, 0.05,
                                                                         deltas=(0.5, 0.1, 0.05, 0.01))
    d2 = batch_tail(designs, 800, 0.05, delta=0.2)
    assert "0.2" in d2["eps_at_delta"] and "0.1" not in d2["eps_at_delta"] and d2["tail"] == d0["tail"]
    assert [min_certified_size(e, 0.1) for e in (0.05, 0.1)] == [355, 190]
    eps, w = 0.05, 0.2
    N = min_certified_size(eps, 0.1, window=w)

    def share(n_):
        looks, sp = C.single_look_design(n_, eps)
        return C.tuned_looks(list(looks), C.spend(list(looks), sp), n_, eps, 0.1, "rectified", window=w)[-1] / n_
    start = int(4 * C.DEFAULT_DESIGN_C ** 2 / eps)
    assert share(N) <= 0.30 and all(share(n_) > 0.30 for n_ in range(start, N))
