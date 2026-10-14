# Replay benchmark

`replay_bench.py` scores a bottlekit package on stored model outputs. It makes no model calls and needs no network, so a
run is free and gives the same numbers each time.

Use it to compare two versions of the code, or two settings, on the same data.

## Data format

The benchmark reads one folder per task. Your data stays outside the repository (`/data/` is in `.gitignore`).

```
DIR/
  <task>/
    spec.json      {"name": "...", "instruction": "...", "labels": {"label": "description", ...}}
    items.jsonl    {"id": "...", "text": "...", "gold": "..."}       one line per item, in replay order
    votes.jsonl    {"id": "...", "r1": "...", "r2": "...", "r3": "..."}
    expert.jsonl   {"id": "...", "label": "..."}
    emb.npy        optional: (n_items, d) float array, row i = item i (only for --dense)
```

| File | Content |
|---|---|
| `spec.json` | The task instruction and the label set. `labels` can also be a list of names. Other keys must be `TaskSpec` fields. |
| `items.jsonl` | The items. `gold` is optional: give it for every item or for none. |
| `votes.jsonl` | The label of the cheap model for each item in 3 different batch contexts. Use `null` for no usable answer. |
| `expert.jsonl` | The single-item label of the expert model. A missing or empty label counts as an unusable expert answer. |
| `emb.npy` | Fixed text embeddings (for example a sentence encoder). Never fit them on the expert labels. |

To make the data, call the cheap model on random batches of the items 3 times (3 contexts) and the expert model once
per item. Record the parsed labels.

## Run

```bash
python benchmarks/replay_bench.py --data DIR --quick            # 8 seeds per scenario
python benchmarks/replay_bench.py --data DIR                    # 30 seeds per scenario
python benchmarks/replay_bench.py --data DIR --dense             # add the embeddings to the student
python benchmarks/replay_bench.py --data DIR --pkg ../other     # score another copy of the package
```

| Option | Meaning |
|---|---|
| `--data DIR` | The data folder (required). |
| `--pkg PATH` | The folder that holds the `bottlekit/` package to score. Default: this repository. |
| `--tasks a,b` | Only these task folders. Default: every folder in `DIR`. |
| `--quick` | 8 seeds, and no `cold@0.02` scenario. |
| `--seeds N` | Seeds per scenario. Default: 30, or 8 with `--quick`. |
| `--n N` | Items per task: the first N items, rounded down to a multiple of 3. Default: 1200. |
| `--expert replay` or `gold` | The expert answers come from `expert.jsonl`, or the gold labels act as the expert. |
| `--cheap-price-in/out`, `--expert-price-in/out` | List prices in USD per million tokens. Defaults: 0.05 / 0.10 and 0.425 / 2.55. |
| `--out FILE` | Write the full JSON summary to a file. |

## How the replay works

| Model | Reply |
|---|---|
| Cheap | The n-th time the package asks about an item, the reply is the item's label from context (n mod 3). |
| Expert | The item's stored expert label, or "I am not sure." when there is none. |
| Tokens | Prompt tokens = prompt characters // 4. Completion tokens = max(1, reply characters // 4). |

The cheap votes do not depend on the batches that the package builds. Thus the benchmark does not measure the
position effects inside a batch.

## Scenarios

Each scenario runs once per seed. A seed changes the calibration samples, the batch orders and the folds.

| Scenario | What runs |
|---|---|
| `cold@0.05`, `cold@0.02` | `Bottler.run` on all items with eps 0.05 or 0.02. |
| `reuse_cold_start` | `run` on the first third of the items. This makes the bottle. |
| `reuse_pour1_cold` | `pour` of the second third with `votes=False` (the bottle alone). |
| `reuse_pour1_warm` | `pour` of the second third with `votes=True` (fresh cheap votes and the bottle). |
| `reuse_pour2_cold` | `pour` of the last third from the bottle of `reuse_pour1_cold`, with `votes=False`. |

## Metrics

| Metric | Meaning |
|---|---|
| `bottled` | Share of items that the bottle answers (no expert call). |
| `cost_frac` | Cost of the run divided by the cost of asking the expert about every item. A run that sends every item to the expert scores 1. |
| `disagreement` | Share of final answers that are not the same as the expert label of the item. |
| `violation_rate` | Share of runs with `disagreement > eps`. |
| `gold_acc` | Share of final answers equal to the gold label (when gold labels are given). |

The summary gives these means per task and scenario, and over all tasks. It also gives:

* `score_mean_bottled` and `score_mean_cost_frac`: the means over the scenarios, without `reuse_cold_start`;
* `validity_alarm_max_violation_rate`: the largest violation rate over all (task, scenario) pairs.

At `delta = 0.1` a violation rate near 0.1 is expected. A rate above 0.2 over 30 seeds is a warning sign.

## The paper's numbers

The paper used this benchmark with 4 tasks (agnews, emotion, esci, banking77), 1,200 items each, and `--quick`. The
replay data of the paper is not part of this repository.
