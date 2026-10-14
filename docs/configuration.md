# Configuration

All settings are fields of `BottleConfig`. Every CLI flag of `run` and `pour` sets one field (for example `--eps 0.02`
sets `BottleConfig.eps`). A flag that you do not give keeps the `BottleConfig` default. `bottlekit run --help` shows
every flag with its default.

**Validity rule.** Every setting is label-free. Any value that you fix **before** a workload keeps the guarantee. A
value that you choose after you look at the expert answers of that workload does not.

## Models

Models are aliases in a models file (see `bottlekit/models.yaml`). Copy the file, edit it, and pass it with
`--models my_models.yaml` or `load_models("my_models.yaml")`.

| Field of an entry | Meaning |
|---|---|
| `id` or `ids` | The model name on the server. `ids` lists replicas of the same model. |
| `base_url` | Any OpenAI-compatible server: vLLM, LM Studio, OpenRouter, OpenAI or a proxy. |
| `api_key_env` | The NAME of the environment variable that holds the API key. Never put the key in the file. |
| `price_in`, `price_out` | USD per million tokens. The cost ledger and the `auto` decisions use them. |
| `max_concurrency` | Concurrent requests to this model (default 2). |
| `system_prefix` | Text in front of the system prompt (for example `/no_think` for Qwen3 models). |
| `extra_body` | Extra JSON fields for each request (for example `{reasoning_effort: low}`). |
| `max_tokens` | Reply budget of one expert call (reasoning models need more). |

## Core fields

| Field | Default | Meaning |
|---|---|---|
| `cheap`, `expert` | (required) | Model aliases. `pour` with `votes=False` needs no cheap model. |
| `eps` | 0.05 | Allowed share of final answers that differ from the expert. |
| `delta` | 0.1 | Allowed probability that the share is above eps. |
| `mode` | `auto` | `fused`, `student` or `auto` (see [concepts](concepts.md#modes)). |
| `seed` | 0 | Seed of every random draw (calibration sample, batch orders, folds). |
| `expert_budget` | None | Maximum escalation calls. See [concepts](concepts.md#expert-budget). |
| `fallback` | True | If the expert gives no usable answer, keep the bottled answer (flagged in the provenance). |
| `max_expert_none` | 0.2 | Stop with `ExpertFailure` if a larger share of calibration answers is unusable. |
| `export` | True | Build a reusable bottle from the certified output. |
| `cost_basis` | `usd` | `usd`: decisions use prices. `tokens`: decisions use token volumes (price 1 per token). |

## Cheap votes and student

| Field | Default | Meaning |
|---|---|---|
| `k` | 20 | Items per cheap call. |
| `contexts` | 3 | Votes per item, each from a different random batch. |
| `warmup` | 0 | Throw-away items at the start of each cheap call. |
| `balanced` | True | No item sits in the first quarter of a batch more than once. |
| `weighted` | True | Weight each vote by the measured reliability of its batch position (no labels needed). |
| `fusion` | `prob` | Fused score: vote share + student probability of the answer (`agree`: + student confidence if it agrees). |
| `n_build` | None | Student mode: items that the cheap model labels (None: max(600, 20 x classes)). |
| `student` | None | Student settings as a dict, for example `{"C": 1.0, "char_analyzer": "char"}`. |
| `crossfit_folds` | 5 | Folds of the cross-fit. |
| `embedder`, `embedder_id` | None | A fixed function from texts to vectors. It adds dense features to the student. |

## Certificate

| Field | Default | Meaning |
|---|---|---|
| `certificate` | `rectified` | `rectified`: bound the errors that reach the output. `hypergeometric`: bound every bad item, sampled items included (stricter). |
| `calibration` | `sequential` | `sequential`: pre-specified looks. `fixed`: one sample of `n_cal` items. |
| `design` | `single` | `single`: one look of about 1.25 x sqrt(N / eps) items at the full delta. `staged`: looks of 100, 300, 900, ... |
| `design_c` | 1.25 | The size constant of the single look. |
| `tune` | True | Tune the sample size to the saw-tooth shape of the exact test (before any label). |
| `looks`, `spending` | None | Your own look sizes and their shares of delta. |
| `max_expected_eps` | None | Also cap the worst-case expected error share (stricter only). |

## Reuse

| Field | Default | Meaning |
|---|---|---|
| `forecast` | True | Record a measured warm/cold forecast in each bottle. `pour(votes="auto")` uses it. |
| `forecast_v2` | `full` | Extra inputs of that decision: `off`, `canpay`, `pour1` or `full`. |
| `gain_prior` | 0.15 | Expected gain of cheap votes when no forecast is available. |
| `memo` | False | Answer recurring items from the expert answers in the bottle. This assumes a deterministic expert. |
| `memo_allowance` | `novel` | Error allowance of a memo pour: `novel` (the new items alone meet eps) or `batch`. |
| `chunking` | `cold` | Split a cold pour much larger than the bottle (N >= 4 x memory) into chunks that update the bottle in turn. |
| `chunk_spend` | `chunk` | `batch`: spend the whole delta across the chunks of a split pour. |
| `max_memory` | 200,000 | Rows that a bottle keeps (newest first). |

## Reports

| Field | Default | Meaning |
|---|---|---|
| `health` | True | Add a health report to each workload: agreement with the expert on the sample (exact intervals), a lower bound on the precision of the bottled answers, drift statistics, and one row of the bottle's learning curve. It never changes an answer or a decision. |
| `health_level` | 0.95 | Level of the agreement intervals. It is a report level, not 1 - delta. |
| `history_cap` | 500 | Learning-curve rows that a bottle keeps. |

## Task definition

```python
from bottlekit import TaskSpec

spec = TaskSpec(
    name="tickets",
    instruction="Classify the support ticket.",
    labels={"billing": "billing: charges, invoices, refunds",
            "delivery": "delivery: shipping, parcels, tracking"},
)
```

Optional `TaskSpec` fields change the prompts and the parser for other languages or models (`prompt_single`,
`prompt_batch`, `answer_prefixes`, `think_pattern`, `chars_per_token`, ...). The expert-side fields enter the expert
fingerprint, so a bottle never reuses expert answers made with another prompt.

Load items with `from_records`, `from_jsonl` or `from_csv`. `load_workload(name)` loads a public task of the registry
(`bottlekit tasks` lists them; it needs `pip install "bottlekit[datasets]"`).
