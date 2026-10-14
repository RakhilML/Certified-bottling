"""Offline replay benchmark for bottlekit.

The benchmark scores a bottlekit package on stored model outputs. It makes no model calls and needs no network.

    python benchmarks/replay_bench.py --data DIR [--pkg PATH] [--tasks a,b] [--quick | --seeds N] [--dense] [--out FILE]

DIR holds one folder per task (see benchmarks/README.md):
    DIR/<task>/spec.json     task instruction and labels
    DIR/<task>/items.jsonl   {"id", "text", "gold"} in replay order
    DIR/<task>/votes.jsonl   {"id", "r1", "r2", "r3"}: the cheap model's label for the item in 3 batch contexts
    DIR/<task>/expert.jsonl  {"id", "label"}: the expert model's single-item label
    DIR/<task>/emb.npy       optional (n_items, d) float array, row i = item i (needed for --dense)

Replay rules:
    cheap  : the n-th time the package asks the cheap model about an item (inside a batch), the reply for that item
             is its label from context (n mod 3). Votes do not depend on the batch the package builds, so the
             benchmark does not measure position effects inside a batch.
    expert : the reply is the item's stored expert label, or "I am not sure." when there is none.
    tokens : prompt tokens = prompt characters // 4; completion tokens = max(1, reply characters // 4).

Scenarios (each repeated over seeds; a seed changes the calibration draws, the batch orders and the folds):
    cold@0.05, cold@0.02 : Bottler.run on all N items (--quick skips cold@0.02)
    reuse                : run on block A (items 0 to N/3) to make a bottle; pour block B with votes=False
                           (reuse_pour1_cold) and with votes=True (reuse_pour1_warm), both from that bottle; then pour
                           block C from the reuse_pour1_cold bottle with votes=False (reuse_pour2_cold)

Metrics per run:
    bottled      : share of items answered by the bottle (no expert call)
    cost_frac    : run cost / cost of asking the expert about every item (each item billed as its replayed reply)
    disagreement : share of final answers that differ from the expert label of the same item
    violation    : disagreement > eps of the scenario
    gold_acc     : share of final answers equal to the gold label
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib
import inspect
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
N_ITEMS = 1200
REUSE_EPS = 0.05
ITEM_RE = re.compile(r"^(\d+)\. ", re.M)
SPEC_CORE = ("name", "instruction", "labels")


# ----------------------------------------------------------------------------------------------- data
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def task_names(data: Path) -> list[str]:
    return sorted(p.name for p in data.iterdir() if p.is_dir() and (p / "items.jsonl").is_file())


def load_task(data: Path, task: str, n_items: int, need_emb: bool):
    """Returns (items, spec dict, votes, expert labels, embeddings or None) for the first n items of a task."""
    d = data / task
    for name in ("spec.json", "items.jsonl", "votes.jsonl", "expert.jsonl"):
        if not (d / name).is_file():
            sys.exit(f"{task}: missing {d / name}")
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    items = read_jsonl(d / "items.jsonl")
    n = 3 * (min(n_items, len(items)) // 3)
    if n == 0:
        sys.exit(f"{task}: items.jsonl has fewer than 3 items")
    items = items[:n]
    ids = [str(it["id"]) for it in items]
    if len(set(ids)) != n:
        sys.exit(f"{task}: item ids are not unique")

    vote_rows = {str(v["id"]): v for v in read_jsonl(d / "votes.jsonl")}
    missing = [i for i in ids if i not in vote_rows]
    if missing:
        sys.exit(f"{task}: votes.jsonl has no row for {len(missing)} item(s), e.g. {missing[0]!r}")
    votes = [{k: vote_rows[i].get(k) for k in ("r1", "r2", "r3")} for i in ids]

    expert_rows = {str(r["id"]): r.get("label") for r in read_jsonl(d / "expert.jsonl")}
    expert = [expert_rows.get(i) for i in ids]
    no_expert = sum(x is None for x in expert)
    if no_expert:
        print(f"{task}: {no_expert} item(s) have no expert label (replayed as an unusable answer)", flush=True)

    emb = None
    if need_emb:
        f = d / "emb.npy"
        if not f.is_file():
            sys.exit(f"{task}: --dense needs {f}")
        emb = np.load(f)
        if len(emb) < n:
            sys.exit(f"{task}: emb.npy has {len(emb)} rows, need {n}")
        emb = emb[:n]
    return items, spec, votes, expert, emb


def make_spec(bk, task: str, spec_d: dict):
    """bottlekit TaskSpec from spec.json. Keys other than name/instruction/labels must be TaskSpec fields."""
    labels = spec_d["labels"]
    if isinstance(labels, list):
        labels = {str(x): str(x) for x in labels}
    fields = {f.name for f in dataclasses.fields(bk.TaskSpec)} if dataclasses.is_dataclass(bk.TaskSpec) else set()
    extra = {k: v for k, v in spec_d.items() if k not in SPEC_CORE}
    unknown = sorted(set(extra) - fields)
    if unknown:
        sys.exit(f"{task}: spec.json keys {unknown} are not TaskSpec fields of this bottlekit")
    return bk.TaskSpec(name=spec_d.get("name", task), instruction=spec_d["instruction"], labels=labels, **extra)


# ----------------------------------------------------------------------------------------------- replay client
class ReplayClient:
    """Implements the LLMClient surface bottlekit uses (models, ledger, chat, map) by replaying stored outputs."""

    def __init__(self, bk, spec, texts, votes, expert_labels, prices):
        self.bk = bk
        self.models = {"cheap": bk.ModelSpec("cheap", ["cheap"], price_in=prices[0], price_out=prices[1], max_concurrency=8),
                       "expert": bk.ModelSpec("expert", ["expert"], price_in=prices[2], price_out=prices[3], max_concurrency=8)}
        self.ledger = importlib.import_module(bk.__name__ + ".llm")._Ledger()
        self.max_chars = spec.max_chars
        self.single_chars = getattr(spec, "single_limit", spec.max_chars * 2)   # item truncation of expert prompts
        self.key_batch = {self._key(t, self.max_chars): i for i, t in enumerate(texts)}
        self.key_single = {self._key(t, self.single_chars): i for i, t in enumerate(texts)}
        self.votes, self.expert = votes, expert_labels
        self.seen = Counter()
        self.lookup_stats = Counter()

    @staticmethod
    def _key(t, n):
        return " ".join(str(t).split())[:n]

    def _lookup(self, text, table, n):
        k = self._key(text, n)
        if k in table:
            self.lookup_stats["exact"] += 1
            return table[k]
        for kk, i in table.items():          # tolerate small prompt-format changes (e.g. extra truncation)
            if kk.startswith(k[:200]) or k.startswith(kk[:200]):
                self.lookup_stats["prefix"] += 1
                return i
        self.lookup_stats["miss"] += 1
        return None

    async def chat(self, alias, messages, max_tokens=256, temperature=0.0, phase="default", seed=None, **_):
        user = messages[-1]["content"]
        prompt_chars = sum(len(m["content"]) for m in messages)
        if alias == "cheap":
            starts = [m.start() for m in ITEM_RE.finditer(user)] + [len(user)]
            lines = []
            for j in range(len(starts) - 1):
                chunk = user[starts[j]:starts[j + 1]].strip()
                num, body = chunk.split(". ", 1)
                i = self._lookup(body, self.key_batch, self.max_chars)
                lab = None
                if i is not None:
                    c = self.seen[i] % 3
                    self.seen[i] += 1
                    lab = self.votes[i][f"r{c + 1}"]
                lines.append(f"{num}: {lab if lab else '???'}")
            text = "\n".join(lines)
        else:
            i = self._lookup(user, self.key_single, self.single_chars)
            lab = self.expert[i] if i is not None else None
            text = lab if lab else "I am not sure."
        spec = self.models[alias]
        pt, ct = prompt_chars // 4, max(1, len(text) // 4)
        r = self.bk.ChatResult(text=text, ok=True, model=alias, prompt_tokens=pt, completion_tokens=ct, usd=spec.usd(pt, ct))
        self.ledger.add(phase, r)
        return r

    async def map(self, coros, limit=64):
        return await asyncio.gather(*[(c() if callable(c) else c) for c in coros])


def _peek(client, text, table, n):
    """ReplayClient._lookup without touching lookup_stats (the cost baseline must not change the run's counters)."""
    k = client._key(text, n)
    if k in table:
        return table[k]
    for kk, i in table.items():
        if kk.startswith(k[:200]) or k.startswith(kk[:200]):
            return i
    return None


def expert_reply_text(client, user):
    """The reply ReplayClient.chat returns for an expert call whose user message is `user`."""
    i = _peek(client, user, client.key_single, client.single_chars)
    lab = client.expert[i] if i is not None else None
    return lab if lab else "I am not sure."


def expert_all_usd(client, spec, texts, bk):
    """Cost of asking the expert about every item, billed exactly as ReplayClient bills an expert call on that item.

    Prompt tokens: characters of the package's single-item prompt // 4. Completion tokens: max(1, len(reply) // 4),
    where reply is the text the client returns for that prompt (the item's expert label, or "I am not sure." when the
    lookup finds no item or the label is empty). So a run that sends every item to the expert scores cost_frac = 1.
    """
    tasks_mod = bk.tasks if hasattr(bk, "tasks") else importlib.import_module(bk.__name__ + ".tasks")
    tot = 0.0
    for t in texts:
        msgs = tasks_mod.single_messages(spec, t)
        pc = sum(len(m["content"]) for m in msgs)
        reply = expert_reply_text(client, msgs[-1]["content"])
        tot += client.models["expert"].usd(pc // 4, max(1, len(reply) // 4))
    return tot


# ----------------------------------------------------------------------------------------------- scenarios
def one(bk, res, texts, gold, expert_labels, client, spec, eps):
    ev = bk.evaluate(res, gold, expert_labels)
    usd = sum(v["usd"] for v in res.ledger.values())
    return {"bottled": res.coverage(), "cost_frac": usd / max(1e-12, expert_all_usd(client, spec, texts, bk)),
            "disagreement": ev["realized_disagreement_with_expert"], "violation": ev["realized_disagreement_with_expert"] > eps,
            "gold_acc": ev.get("final_gold_acc")}


async def run_task(bk, data, task, expert_mode, seeds, quick, dense, prices, n_items):
    items, spec_d, votes, expert_raw, emb = load_task(data, task, n_items, dense)
    spec = make_spec(bk, task, spec_d)
    texts = [it["text"] for it in items]
    gold_raw = [it.get("gold") for it in items]
    has_gold = [g is not None for g in gold_raw]
    if any(has_gold) and not all(has_gold):
        sys.exit(f"{task}: give 'gold' for every item or for none")
    gold = gold_raw if all(has_gold) else None
    if expert_mode == "gold":
        if gold is None:
            sys.exit(f"{task}: --expert gold needs gold labels in items.jsonl")
        expert_labels = gold
    else:
        none = getattr(bk, "NONE", "__none__")
        expert_labels = [x or none for x in expert_raw]
    n = len(texts)
    k = n // 3
    A, B, C = slice(0, k), slice(k, 2 * k), slice(2 * k, 3 * k)
    gsl = (lambda sl: gold[sl]) if gold is not None else (lambda sl: None)
    out = defaultdict(list)
    lookups = Counter()
    pour_takes_emb = "emb" in inspect.signature(bk.Bottler.pour).parameters   # --dense reaches pours when supported
    for s in range(seeds):
        for eps in ((0.05,) if quick else (0.05, 0.02)):
            client = ReplayClient(bk, spec, texts, votes, expert_labels, prices)
            res = await bk.Bottler(client, spec, bk.BottleConfig(cheap="cheap", expert="expert", eps=eps, seed=s)).run(
                texts, emb if dense else None)
            out[f"cold@{eps}"].append(one(bk, res, texts, gold, expert_labels, client, spec, eps))
            lookups.update(client.lookup_stats)
        # reuse
        client = ReplayClient(bk, spec, texts, votes, expert_labels, prices)
        b = bk.Bottler(client, spec, bk.BottleConfig(cheap="cheap", expert="expert", eps=REUSE_EPS, seed=1000 + s))
        r0 = await b.run(texts[A], emb[A] if dense else None)
        out["reuse_cold_start"].append(one(bk, r0, texts[A], gsl(A), expert_labels[A], client, spec, REUSE_EPS))
        pe = (lambda sl: {"emb": emb[sl]}) if (dense and pour_takes_emb) else (lambda sl: {})
        r_cold = await b.pour(r0.artifact, texts[B], votes=False, **pe(B))
        out["reuse_pour1_cold"].append(one(bk, r_cold, texts[B], gsl(B), expert_labels[B], client, spec, REUSE_EPS))
        r_warm = await b.pour(r0.artifact, texts[B], votes=True, **pe(B))
        out["reuse_pour1_warm"].append(one(bk, r_warm, texts[B], gsl(B), expert_labels[B], client, spec, REUSE_EPS))
        r2 = await b.pour(r_cold.artifact, texts[C], votes=False, **pe(C))
        out["reuse_pour2_cold"].append(one(bk, r2, texts[C], gsl(C), expert_labels[C], client, spec, REUSE_EPS))
        lookups.update(client.lookup_stats)
    return out, dict(lookups), n


def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def summarise(per_task):
    summ = {"per_task": {}, "overall": {}}
    agg = defaultdict(list)
    alarm = 0.0
    for task, sc in per_task.items():
        summ["per_task"][task] = {}
        for name, runs in sc.items():
            d = {"bottled": float(np.mean([r["bottled"] for r in runs])),
                 "cost_frac": float(np.mean([r["cost_frac"] for r in runs])),
                 "disagreement": float(np.mean([r["disagreement"] for r in runs])),
                 "violation_rate": float(np.mean([r["violation"] for r in runs])),
                 "gold_acc": _mean([r["gold_acc"] for r in runs]),
                 "runs": len(runs)}
            summ["per_task"][task][name] = d
            agg[name].append(d)
            alarm = max(alarm, d["violation_rate"])
    for name, ds in agg.items():
        summ["overall"][name] = {k: _mean([d[k] for d in ds]) for k in ("bottled", "cost_frac", "violation_rate", "gold_acc")}
    keys = [k for k in summ["overall"] if k != "reuse_cold_start"]
    summ["score_mean_bottled"] = float(np.mean([summ["overall"][k]["bottled"] for k in keys]))
    summ["score_mean_cost_frac"] = float(np.mean([summ["overall"][k]["cost_frac"] for k in keys]))
    summ["validity_alarm_max_violation_rate"] = alarm
    return summ


# ----------------------------------------------------------------------------------------------- main
def import_bottlekit(pkg: str):
    sys.path.insert(0, str(Path(pkg).resolve()))
    for m in [m for m in sys.modules if m == "bottlekit" or m.startswith("bottlekit.")]:
        del sys.modules[m]
    return importlib.import_module("bottlekit")


def main():
    ap = argparse.ArgumentParser(description="Offline replay benchmark for bottlekit (no model calls).")
    ap.add_argument("--data", required=True, help="data directory with one folder per task (see benchmarks/README.md)")
    ap.add_argument("--pkg", default=str(REPO), help="directory that contains the bottlekit/ package to benchmark "
                                                     "(default: this repository)")
    ap.add_argument("--tasks", default=None, help="comma-separated task folders (default: every folder in --data)")
    ap.add_argument("--seeds", type=int, default=None, help="seeds per scenario (default: 30, or 8 with --quick)")
    ap.add_argument("--quick", action="store_true", help="8 seeds; skips cold@0.02 (the reuse scenarios still run)")
    ap.add_argument("--n", type=int, default=N_ITEMS, help="items per task: the first N of items.jsonl, rounded "
                                                          "down to a multiple of 3 (default: 1200)")
    ap.add_argument("--expert", choices=("replay", "gold"), default="replay",
                    help="'replay': expert.jsonl labels; 'gold': the gold labels act as the expert")
    ap.add_argument("--dense", action="store_true", help="pass emb.npy rows as emb= to run() and to pour()")
    ap.add_argument("--cheap-price-in", type=float, default=0.05, help="USD per million input tokens")
    ap.add_argument("--cheap-price-out", type=float, default=0.10, help="USD per million output tokens")
    ap.add_argument("--expert-price-in", type=float, default=0.425, help="USD per million input tokens")
    ap.add_argument("--expert-price-out", type=float, default=2.55, help="USD per million output tokens")
    ap.add_argument("--out", default=None, help="write the full JSON summary to this file")
    a = ap.parse_args()

    data = Path(a.data).resolve()
    if not data.is_dir():
        sys.exit(f"--data: {data} is not a directory")
    tasks = a.tasks.split(",") if a.tasks else task_names(data)
    if not tasks:
        sys.exit(f"--data: no task folder with items.jsonl in {data}")
    bk = import_bottlekit(a.pkg)
    pkg_dir = Path(bk.__file__).resolve().parent
    print("benchmarking", pkg_dir, flush=True)
    seeds = a.seeds or (8 if a.quick else 30)
    prices = (a.cheap_price_in, a.cheap_price_out, a.expert_price_in, a.expert_price_out)

    per_task, lookups, n_used = {}, {}, {}
    t0 = time.time()
    for task in tasks:
        per_task[task], lookups[task], n_used[task] = asyncio.run(
            run_task(bk, data, task, a.expert, seeds, a.quick, a.dense, prices, a.n))
        miss = lookups[task].get("miss", 0)
        note = f" ({miss} replay lookups found no item)" if miss else ""
        print(task, "done", f"{time.time() - t0:.0f}s{note}", flush=True)

    summ = summarise(per_task)
    summ["replay_lookups"] = lookups
    summ["config"] = {"pkg": str(pkg_dir), "data": str(data), "tasks": tasks, "n_items": n_used, "seeds": seeds,
                      "quick": a.quick, "expert": a.expert, "dense": a.dense, "prices": prices,
                      "wall_s": time.time() - t0,
                      "expert_all_billing": "each item billed as its replayed expert reply: "
                                            "prompt chars // 4 in, max(1, len(reply) // 4) out"}
    text = json.dumps(summ, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
    print(json.dumps({"overall": summ["overall"], "score_mean_bottled": summ["score_mean_bottled"],
                      "score_mean_cost_frac": summ["score_mean_cost_frac"],
                      "validity_alarm": summ["validity_alarm_max_violation_rate"]}, indent=1))


if __name__ == "__main__":
    main()
