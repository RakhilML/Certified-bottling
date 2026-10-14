"""CLI.

  python -m bottlekit tasks
  python -m bottlekit run --task trec --n 1000 --offset 0 --cheap cheap --expert expert [--mode auto|fused|student]
         [--k 20 --contexts 3 --warmup 0 --calibration sequential|fixed --design single|staged --eps 0.05
          --delta 0.1 --budget 300]
         [--save-artifact bottle.pkl] [--eval-gold] [--eval-expert-all] [--out result.json]
  python -m bottlekit pour --artifact bottle.pkl --task trec --n 1000 --offset 1000 --expert expert
         [--cheap cheap --votes auto|yes|no] [--save-artifact next.pkl]
  python -m bottlekit run --jsonl my.jsonl --text-key text --label-key label --labels A,B,C --instruction "..." ...
  python -m bottlekit run --jsonl my.jsonl --labels-file labels.txt ...   (one label per line, optional 'name: description')
  ... --embed-model <alias>   (run and pour; off by default): dense student/bottle features from an OpenAI-compatible
                              /v1/embeddings model in models.yaml; costed under ledger phase 'embed'
Every BottleConfig field except the callable embedder has a flag (see _CFG_FLAGS). Flag defaults are None and mean
"the BottleConfig default" (help texts show it), so the CLI never re-declares a default.
"""

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

from .certify import NONE
from .economics import COST_BASES
from .harness import (CERTIFICATES, CHUNKINGS, DESIGNS, FORECAST_V2, MEMO_ALLOWANCES, MODES, BottleArtifact,
                      BottleConfig, Bottler, evaluate)
from .labeling import single_max_tokens
from .llm import DEFAULT_GLOBAL_CONCURRENCY, LLMClient, load_models
from .tasks import REGISTRY, TaskSpec, from_jsonl, load_workload, parse_label, single_messages

HERE = Path(__file__).resolve().parent
REGISTRY_DEFAULT_N = 1000     # registry tasks: items when --n is not given
DEFAULT_INSTRUCTION = "Classify the item."   # --jsonl tasks without --instruction
CFG_DEFAULTS = {f.name: f.default for f in dataclasses.fields(BottleConfig) if f.default is not dataclasses.MISSING}


def _folds(v: str):
    return v if v == "auto" else int(v)


def _ints(v: str) -> tuple:
    return tuple(int(x) for x in v.split(","))


def _floats(v: str) -> tuple:
    return tuple(float(x) for x in v.split(","))


# (flag, namespace dest, BottleConfig field, type, choices, help). Every default is None = the BottleConfig default.
_CFG_FLAGS = [
    ("--k", "k", "k", int, None, "real items per cheap call"),
    ("--contexts", "contexts", "contexts", int, None, "voting contexts (fused mode, warm pours)"),
    ("--warmup", "warmup", "warmup", int, None, "throwaway items per cheap call (cold-start fix)"),
    ("--fusion", "fusion", "fusion", str, ["agree", "prob"], "fused score"),
    ("--n-build", "n_build", "n_build", int, None, "student mode: cheap labels (None: max(n_build_min, "
                                                    "n_build_per_class*classes))"),
    ("--calibration", "calibration", "calibration", str, ["sequential", "fixed"], "calibration procedure"),
    ("--n-cal", "n_cal", "n_cal", int, None, "fixed calibration size (--calibration fixed)"),
    ("--looks", "looks", "looks", _ints, None, "explicit sequential looks, e.g. 100,200,400 (default: --design)"),
    ("--spending", "spending", "spending", _floats, None, "Bonferroni weights per look for --looks, e.g. 1,4"),
    ("--design", "design", "design", str, list(DESIGNS),
     "default design when --looks is not given: single (one full-delta look of ~design_c*sqrt(N/eps) items, "
     "<= N/2) or staged (100, 300, 900, ... with delta shares 1:4:4)"),
    ("--design-c", "design_c", "design_c", float, None, "single-look size constant (label-free; fixed before any data is seen)"),
    ("--certificate", "certificate", "certificate", str, list(CERTIFICATES),
     "rectified: bound the errors that reach the output; hypergeometric: bound every bad item, calibration items included"),
    ("--max-expected-eps", "max_expected_eps", "max_expected_eps", float, None,
     "optional, stricter only: also cap the worst-case EXPECTED share of unlabelled bottled answers != expert "
     "(lowers k; label-free). Default: off"),
    ("--patience", "patience", "patience", int, None, "optional heuristic stop of staged looks"),
    ("--target-coverage", "target_coverage", "target_coverage", float, None, "stop looks at this coverage"),
    ("--budget", "budget", "expert_budget", int, None, "max expert escalation calls (reports eps achieved; the k-rule and the budget test each run at delta/2)"),
    ("--eps", "eps", "eps", float, None, "allowed share of answers != expert"),
    ("--delta", "delta", "delta", float, None, "failure probability of the certificate"),
    ("--seed", "seed", "seed", int, None, "random seed (calibration draws, orders, folds; registry tasks: also the "
                                          "workload shuffle, so keep it equal for run and pour)"),
    ("--gain-prior", "gain_prior", "gain_prior", float, None, "auto mode: prior coverage gain of voting"),
    ("--max-expert-none", "max_expert_none", "max_expert_none", float, None,
     "abort if more calibration expert answers than this share are unusable"),
    ("--memo-allowance", "memo_allowance", "memo_allowance", str, list(MEMO_ALLOWANCES), "memo pours: allowance m"),
    ("--chunking", "chunking", "chunking", str, list(CHUNKINGS),
     "right-sized pours: 'cold' splits a COLD pour of >= chunk_cold_ratio x the bottle's memory into chained chunks "
     "(whole-batch guarantee at delta_batch <= delta; warm pours never split); 'auto' plans every pour; "
     "'off' never splits (run() never splits)"),
    ("--chunk-cold-ratio", "chunk_cold_ratio", "chunk_cold_ratio", float, None, "'cold' rule: split iff N >= ratio x M"),
    ("--chunk-tax", "chunk_tax", "chunk_tax", float, None, "N_min: smallest batch with calibration share <= tax"),
    ("--chunk-growth", "chunk_growth", "chunk_growth", float, None, "chunk j = max(N_min, growth x memory)"),
    ("--max-chunks", "max_chunks", "max_chunks", int, None, "cap on the number of chunks"),
    ("--forecast-v2", "forecast_v2", "forecast_v2", str, list(FORECAST_V2), "votes='auto' forecast v2 parts"),
    ("--forecast-folds", "forecast_folds", "forecast_folds", _folds, None, "'auto' or an int >= 2"),
    ("--crossfit-folds", "crossfit_folds", "crossfit_folds", int, None, "folds of run()'s cross-fit (>= 2)"),
    ("--max-memory", "max_memory", "max_memory", int, None, "memory cap of exported bottles"),
    ("--n-build-min", "n_build_min", "n_build_min", int, None, "student mode: minimum cheap labels"),
    ("--n-build-per-class", "n_build_per_class", "n_build_per_class", int, None, "student mode: cheap labels/class"),
    ("--gain-fed-per-class", "gain_fed_per_class", "gain_fed_per_class", float, None, "gain prior: no boost from here"),
    ("--gain-starved-per-class", "gain_starved_per_class", "gain_starved_per_class", float, None,
     "gain prior: full boost below this many labels per class"),
    ("--gain-starved-mult", "gain_starved_mult", "gain_starved_mult", float, None, "gain prior: full boost factor"),
    ("--early-frac", "early_frac", "early_frac", float, None, "cold-start zone share of a cheap batch"),
    ("--forecast-hysteresis", "forecast_hysteresis", "forecast_hysteresis", float, None, "votes='auto' margin x c_exp"),
    ("--forecast-gain-decay", "forecast_gain_decay", "forecast_gain_decay", float, None, "stale warm gain decay/age"),
    ("--forecast-fit-budget", "forecast_fit_budget", "forecast_fit_budget", float, None,
     "forecast cross-fit work budget (bottle fits)"),
    ("--forecast-max-folds", "forecast_max_folds", "forecast_max_folds", int, None, "max 'auto' forecast folds"),
    ("--forecast-max-rows", "forecast_max_rows", "forecast_max_rows", int, None,
     "cap on the pre-batch memory rows of a forecast cross-fit (default: all)"),
    ("--forecast-canpay-z", "forecast_canpay_z", "forecast_canpay_z", float, None, "can-pay ceiling uses a + z*se"),
    ("--cost-basis", "cost_basis", "cost_basis", str, list(COST_BASES), "decision costs: usd (prices) or tokens"),
    ("--chunk-spend", "chunk_spend", "chunk_spend", str, ["chunk", "batch"],
     "chunked pours: 'chunk' certifies every chunk at (eps, delta); 'batch' spends the batch's delta across chunks"),
    ("--chunk-alpha-max", "chunk_alpha_max", "chunk_alpha_max", float, None,
     "chunk_spend='batch': ceiling on the common per-chunk level alpha*"),
    ("--health-level", "health_level", "health_level", float, None, "health report: interval level (report-only)"),
    ("--history-cap", "history_cap", "history_cap", int, None, "health report: learning-curve rows kept"),
    ("--health-max-rows", "health_max_rows", "health_max_rows", int, None,
     "health report: drift-statistics subsample size (default: 20,000)"),
    ("--health-low-percentile", "health_low_percentile", "health_low_percentile", float, None,
     "health report: low-tail percentile of the drift statistics"),
]
# (flag, namespace dest, BottleConfig field, value set when the flag is given)
_CFG_SWITCHES = [
    ("--no-balance", "no_balance", "balanced", False), ("--no-weights", "no_weights", "weighted", False),
    ("--no-tune", "no_tune", "tune", False), ("--memo", "memo", "memo", True),
    ("--no-forecast", "no_forecast", "forecast", False), ("--forecast-run", "forecast_run", "forecast_run", True),
    ("--no-fallback", "no_fallback", "fallback", False), ("--no-export", "no_export", "export", False),
    ("--no-health", "no_health", "health", False),
]
CLI_SKIPPED_FIELDS = ("embedder", "dense")   # a callable (use --embed-model) and an unused field


def _conv(conv, v):
    """Apply a flag's converter to a value that may still be a string (hand-built Namespaces) or already converted."""
    if isinstance(v, str) and conv is not str:
        return conv(v)
    return tuple(v) if conv in (_ints, _floats) and isinstance(v, list) else v


def _labels(a) -> dict[str, str]:
    labels: dict[str, str] = {}
    if a.labels_file:
        for line in Path(a.labels_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, _, desc = line.partition(":")
            labels[name.strip()] = desc.strip() or name.strip()
    for l in (a.labels or "").split(","):
        if l.strip():
            labels.setdefault(l.strip(), l.strip())
    if len(labels) < 2:
        sys.exit("custom workloads need at least 2 labels (--labels A,B or --labels-file)")
    return labels


def _workload(a):
    mc = getattr(a, "max_chars", None)
    if a.jsonl:
        spec = TaskSpec(name=Path(a.jsonl).stem, instruction=a.instruction or DEFAULT_INSTRUCTION, labels=_labels(a),
                        **({} if mc is None else {"max_chars": int(mc)}))
        wl = from_jsonl(spec, a.jsonl, text_key=a.text_key, label_key=a.label_key or None)
        return wl.subset(a.offset, a.n)
    wl = load_workload(a.task, n=a.n if a.n is not None else REGISTRY_DEFAULT_N, offset=a.offset, seed=a.seed or 0)
    if mc is not None:
        wl.spec.max_chars = int(mc)
    return wl


def _cfg(a) -> BottleConfig:
    """BottleConfig from parsed flags: only flags that were given (not None) are passed, so every other field keeps
    its BottleConfig default. Accepts Namespaces built by hand (a missing attribute counts as not given)."""
    kw = {"cheap": getattr(a, "cheap", None), "expert": a.expert}
    if getattr(a, "mode", None):
        kw["mode"] = a.mode
    for _flag, dest, fld, conv, _ch, _h in _CFG_FLAGS:
        v = getattr(a, dest, None)
        if v is not None and not (fld in ("fusion", "chunking", "chunk_spend", "looks", "spending") and v == ""):
            kw[fld] = _conv(conv, v)
    for _flag, dest, fld, val in _CFG_SWITCHES:
        if getattr(a, dest, None):
            kw[fld] = val
    st = getattr(a, "student", None)
    st = dict(json.loads(st) if isinstance(st, str) else st) if st else {}
    if getattr(a, "embedder_id", None):
        kw["embedder_id"] = a.embedder_id
    if getattr(a, "student_c", None) is not None:
        st["C"] = float(a.student_c)
    if st:
        kw["student"] = st
    return BottleConfig(**kw)


def _attach_embedder(cfg: BottleConfig, client: LLMClient, alias: str | None) -> BottleConfig:
    """--embed-model: a fixed, label-free embedding model; called once per batch before any calibration draw."""
    if not alias:
        return cfg
    if alias not in client.models:
        sys.exit(f"--embed-model {alias!r} is not in the models file")
    cfg.embedder = lambda texts: client.embed(alias, texts, phase="embed")
    cfg.embedder_id = cfg.embedder_id or client.models[alias].ids[0]    # --embed-id overrides the first replica id
    return cfg


async def cmd_run(a):
    models = load_models(a.models)
    client = LLMClient(models, global_concurrency=a.global_concurrency, cache_path=a.cache)
    wl = _workload(a)
    bot = Bottler(client, wl.spec, _attach_embedder(_cfg(a), client, getattr(a, "embed_model", None)))
    t0 = time.time()
    if a.cmd == "pour":
        votes = {"auto": "auto", "yes": True, "no": False}[a.votes]
        res = await bot.pour(BottleArtifact.load(a.artifact), wl.texts, votes=votes)
    else:
        res = await bot.run(wl.texts)
    cfg_out = {f.name: getattr(bot.cfg, f.name) for f in dataclasses.fields(bot.cfg) if f.name != "embedder"}
    out = {"task": wl.spec.name, "N": len(wl.items), "config": vars(a), "bottle_config": cfg_out,
           "certificate": res.certificate,
           "ledger": res.ledger, "diagnostics": res.diagnostics, "wall_min": (time.time() - t0) / 60}
    pipeline_usd = sum(v["usd"] for v in res.ledger.values())
    out["pipeline_usd"] = pipeline_usd
    if a.save_artifact and res.artifact is not None:
        res.artifact.save(a.save_artifact)
        out["artifact"] = {"path": a.save_artifact, "version": res.artifact.version, "memory": len(res.artifact.texts),
                           "dense": res.artifact.dense, "embedder_id": res.artifact.embedder_id,
                           "emb_dim": res.artifact.emb_dim}
    gold = wl.gold if (a.eval_gold or a.eval_expert_all) and all(g is not None for g in wl.gold) else None
    expert_all = None
    if a.eval_expert_all:  # evaluation only: expert labels EVERY item (exact realized disagreement); not pipeline cost
        mt = single_max_tokens(client, a.expert, wl.spec)
        rs = await client.map([client.chat(a.expert, single_messages(wl.spec, t), max_tokens=mt, phase="EVAL_expert_all")
                               for t in wl.texts])
        expert_all = [parse_label(wl.spec, r.text) or NONE for r in rs]
        out["expert_all_usd_est"] = sum(r.usd for r in rs)  # price of an expert call on every item (cache-independent)
        out["cost_frac_vs_expert_all"] = pipeline_usd / max(1e-12, out["expert_all_usd_est"])
    if gold is not None or expert_all is not None:
        out["eval"] = evaluate(res, gold, expert_all)
    await client.close()
    text = json.dumps(out, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


def _common(r):
    r.add_argument("--task", default=None)
    r.add_argument("--jsonl", default=None)
    r.add_argument("--text-key", default="text")
    r.add_argument("--label-key", default="label", help="gold label field (evaluation only); '' for none")
    r.add_argument("--labels", default="")
    r.add_argument("--labels-file", default=None)
    r.add_argument("--instruction", default=None, help=f"jsonl tasks (default: {DEFAULT_INSTRUCTION!r})")
    r.add_argument("--n", type=int, default=None,
                   help=f"items (registry default {REGISTRY_DEFAULT_N}; jsonl default all; 0 = all)")
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--max-chars", type=int, default=None,
                   help="per-item truncation of batched prompts (expert prompts: x TaskSpec.single_chars_factor); "
                        "default: the TaskSpec default")
    r.add_argument("--expert", required=True)
    for flag, dest, fld, conv, choices, hlp in _CFG_FLAGS:
        r.add_argument(flag, dest=dest, type=conv, default=None, choices=choices,
                       help=f"{hlp} (BottleConfig.{fld}; default {CFG_DEFAULTS[fld]!r})")
    for flag, dest, fld, val in _CFG_SWITCHES:
        r.add_argument(flag, dest=dest, action="store_true", default=None,
                       help=f"set BottleConfig.{fld}={val} (default {CFG_DEFAULTS[fld]!r})")
    r.add_argument("--student", default=None,
                   help="JSON dict of student (TfidfLR) settings, e.g. '{\"C\": 1.0, \"char_analyzer\": \"char\"}' "
                        "(BottleConfig.student; default: the built-in student settings)")
    r.add_argument("--student-c", type=float, default=None, help="shortcut for the student's C (default 4.0)")
    r.add_argument("--models", default=str(HERE / "models.yaml"),
                   help="models file (default: the example models.yaml shipped with the package)")
    r.add_argument("--cache", default=None, help="sqlite response cache (identical requests are answered from it)")
    r.add_argument("--global-concurrency", type=int, default=DEFAULT_GLOBAL_CONCURRENCY)
    r.add_argument("--save-artifact", default=None)
    r.add_argument("--embed-model", default=None,
                   help="models.yaml alias of an OpenAI-compatible embedding model: TF-IDF + dense student and bottle "
                        "(off by default; the model must be fixed and never tuned on this workload's expert labels)")
    r.add_argument("--embed-id", dest="embedder_id", default=None,
                   help="embedder identity recorded in bottles (default: the --embed-model's first replica id)")
    r.add_argument("--eval-gold", action="store_true", help="evaluation only: accuracy against the gold labels")
    r.add_argument("--eval-expert-all", action="store_true",
                   help="evaluation only: ALSO call the expert on every item (costs as much as not bottling; use "
                        "--cache to reuse the pipeline's expert answers)")
    r.add_argument("--out", default=None, help="write the result JSON here (it is always printed)")


def main():
    ap = argparse.ArgumentParser(prog="bottlekit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    r = sub.add_parser("run")
    _common(r)
    r.add_argument("--cheap", required=True)
    r.add_argument("--mode", default=None, choices=list(MODES), help=f"default {CFG_DEFAULTS['mode']!r}")
    p = sub.add_parser("pour")
    _common(p)
    p.add_argument("--artifact", required=True)
    p.add_argument("--cheap", default=None, help="cheap model for a WARM pour (votes fused with the bottle)")
    p.add_argument("--votes", default="auto", choices=["auto", "yes", "no"], help="warm pour: auto (economics rule), yes, no")
    a = ap.parse_args()
    if a.cmd == "tasks":
        for k, v in REGISTRY.items():
            print(f"{k:16} {v['hf']}")
        return
    if not a.task and not a.jsonl:
        sys.exit("--task or --jsonl is required")
    asyncio.run(cmd_run(a))


if __name__ == "__main__":
    main()
