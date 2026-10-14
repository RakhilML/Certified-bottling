"""Tasks and workloads.

A TaskSpec describes ONE narrow classification task (instruction + label set with optional descriptions).
A Workload is a list of items {"id", "text", "gold"?}. Gold is only ever used for evaluation.

Sources:
  - REGISTRY: declarative HuggingFace specs (add a dict entry; no code)
  - from_records / from_jsonl / from_csv: any custom workload
Prompt builders and parsers are shared by every labeling mode (single, batched, warm-up).
Parsing is deliberately conservative: an ambiguous reply maps to None (never to a guessed label).
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
from dataclasses import dataclass, field
import math
from functools import cached_property
from pathlib import Path

DATA_DIR = Path(os.environ.get("BOTTLEKIT_DATA_DIR") or Path(__file__).resolve().parents[1] / "data")   # registry cache
NONE = "__none__"  # sentinel for "no usable answer"; never counts as agreement

# Prompt and reply-budget constants (English-shaped). TaskSpec fields below override them per task.
DEFAULT_MAX_CHARS = 700         # per-item truncation in batched prompts (chars: language-dependent)
SINGLE_CHARS_FACTOR = 2         # single (expert) prompts truncate at max_chars * this
LABEL_CHARS_PER_TOKEN = 3       # label reply budget: longest label chars // this + LABEL_TOKENS_PAD
LABEL_TOKENS_PAD = 2
BATCH_TOKENS_PER_ITEM_MIN = 14  # batch reply budget: max(MIN, label tokens + PAD) per item + BASE
BATCH_TOKENS_PER_ITEM_PAD = 6
BATCH_TOKENS_BASE = 60
PROMPT_SINGLE = ("{instruction}\nAllowed labels:\n{labels}\n"
                 "Respond with ONLY the label name, exactly as written above, and nothing else.")
PROMPT_BATCH = ("{instruction}\nAllowed labels:\n{labels}\n"
                "You will receive a numbered list of items. For EACH item output exactly one line in the form "
                "`<number>: <label>` using ONLY the allowed label names. Output nothing else.")


@dataclass
class TaskSpec:
    name: str
    instruction: str
    labels: dict[str, str]                 # label name -> description (description may equal the name)
    max_chars: int = DEFAULT_MAX_CHARS     # per-item truncation inside batched prompts (single prompts: 2x)
    # ---- language/model hooks: every default gives the standard prompts and parser byte for byte. A non-default
    # value of an EXPERT-side hook (prompt_single, the parser hooks, single_chars_factor, chars_per_token) enters the
    # expert request fingerprint, so a bottle's stored expert answers are never memo-reused under another prompt or
    # parser. Fix them before the batch; never choose them on calibration labels.
    prompt_single: str | None = None       # system prompt template of expert (single-item) calls over {instruction},
                                           # {labels}; None = PROMPT_SINGLE (English)
    prompt_batch: str | None = None        # system prompt template of cheap batched calls; None = PROMPT_BATCH. Replies
                                           # are still parsed as numbered lines '<number>: <label>'
    answer_prefixes: tuple | None = None   # reply prefixes stripped before matching, e.g. ("etiqueta", "respuesta");
                                           # case-insensitive phrases followed by ':', '=', '-' (or '：' with
                                           # fullwidth_punct). None = the English list (final) answer|label|...
    think_pattern: str | None = None       # regex of reasoning blocks removed before parsing; None = <think>...</think>
    letter_class: str = "A-Z"              # regex character class of single-letter label sets (e.g. "Α-Ω" for Greek)
    fullwidth_punct: bool = False          # also accept full-width '：．）' after item numbers and answer prefixes
    single_chars_factor: int = SINGLE_CHARS_FACTOR   # expert prompts truncate items at max_chars * this
    chars_per_token: float | None = None   # None: the English ratios (3 for the label reply budget, 3.8 for cost
                                           # estimates); set ~1-1.5 for CJK so reply budgets are not truncated

    def __post_init__(self):
        if len(self.labels) < 2:
            raise ValueError(f"task {self.name!r}: need at least 2 labels, got {list(self.labels)}")
        if len({k.lower() for k in self.labels}) != len(self.labels):
            raise ValueError(f"task {self.name!r}: labels differ only by case: {list(self.labels)}")
        if self.answer_prefixes is not None:
            self.answer_prefixes = tuple(str(p) for p in self.answer_prefixes)
        if self.chars_per_token is not None and not self.chars_per_token > 0:
            raise ValueError("chars_per_token must be > 0 (or None)")
        if not int(self.single_chars_factor) >= 1:
            raise ValueError("single_chars_factor must be an int >= 1")

    @property
    def single_limit(self) -> int:
        """Item truncation of single (expert) prompts: max_chars * single_chars_factor (1400 by default)."""
        return self.max_chars * self.single_chars_factor

    @property
    def names(self) -> list[str]:
        return list(self.labels)

    @cached_property
    def max_label_tokens(self) -> int:
        longest = max(len(n) for n in self.labels)
        if self.chars_per_token is None:
            return longest // LABEL_CHARS_PER_TOKEN + LABEL_TOKENS_PAD
        return int(math.ceil(longest / self.chars_per_token)) + LABEL_TOKENS_PAD

    def label_block(self) -> str:
        out = []
        for k, v in self.labels.items():
            same = (not v) or _norm(v) == _norm(k)
            out.append(f"- {k}" if same else f"- {k}: {v}")
        return "\n".join(out)


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", s.strip().lower()).strip(" .")


# fields of TaskSpec beyond the core four (name, instruction, labels, max_chars) and their defaults
SPEC_EXTRA_DEFAULTS = {"prompt_single": None, "prompt_batch": None, "answer_prefixes": None, "think_pattern": None,
                       "letter_class": "A-Z", "fullwidth_punct": False, "single_chars_factor": SINGLE_CHARS_FACTOR,
                       "chars_per_token": None}
EXPERT_SIDE_EXTRAS = ("prompt_single", "answer_prefixes", "think_pattern", "letter_class", "fullwidth_punct",
                      "single_chars_factor", "chars_per_token")   # change the expert request or its parsed answer


def spec_extras(spec: TaskSpec, keys=None) -> dict:
    """Non-default optional TaskSpec fields as plain values ({} for a default spec). Saved with bottles and
    hashed into the expert fingerprint only when non-empty, so default specs keep their core bytes and hashes."""
    out = {}
    for k in (SPEC_EXTRA_DEFAULTS if keys is None else keys):
        v = getattr(spec, k, SPEC_EXTRA_DEFAULTS[k])
        if v != SPEC_EXTRA_DEFAULTS[k]:
            out[k] = list(v) if isinstance(v, tuple) else v
    return out


def spec_to_plain(spec: TaskSpec) -> dict:
    """The spec as saved in bottles and session states: the core four keys plus any non-default optional field."""
    return {"name": spec.name, "instruction": spec.instruction, "labels": spec.labels, "max_chars": spec.max_chars,
            **spec_extras(spec)}


def _clean_item(text: str, limit: int) -> str:
    """One line per item: collapse whitespace (embedded blank lines / numbered lists would corrupt numbering)."""
    return " ".join(str(text).split())[:limit]


# ----------------------------------------------------------------------------------------------- prompts
def single_messages(spec: TaskSpec, text: str) -> list[dict]:
    sys = (spec.prompt_single or PROMPT_SINGLE).format(instruction=spec.instruction, labels=spec.label_block())
    return [{"role": "system", "content": sys}, {"role": "user", "content": _clean_item(text, spec.single_limit)}]


def batch_messages(spec: TaskSpec, texts: list[str]) -> list[dict]:
    sys = (spec.prompt_batch or PROMPT_BATCH).format(instruction=spec.instruction, labels=spec.label_block())
    body = "\n\n".join(f"{i + 1}. {_clean_item(t, spec.max_chars)}" for i, t in enumerate(texts))
    return [{"role": "system", "content": sys}, {"role": "user", "content": body}]


def batch_max_tokens(spec: TaskSpec, n_items: int, extra: int = 0) -> int:
    """Reply budget of one batched call of n_items (+ `extra`, e.g. ModelSpec.batch_extra_tokens for reasoning)."""
    return (max(BATCH_TOKENS_PER_ITEM_MIN, spec.max_label_tokens + BATCH_TOKENS_PER_ITEM_PAD) * n_items
            + BATCH_TOKENS_BASE + int(extra))


# ----------------------------------------------------------------------------------------------- parsing
PARSER_VERSION = "2"          # bump whenever parse_label / parse_batch can map a reply differently (memo fingerprint)
_THINK = re.compile(r"(?is)<think>.*?</think>|^.*?</think>")
_PREFIX = r"(?i)^(final\s+)?(answer|label|category|class|intent|output)\s*[:=\-]\s*"


def parser_config(spec: TaskSpec) -> dict:
    """Non-default parser hooks of `spec` ({} = the default parser). Part of the expert fingerprint when non-empty."""
    return spec_extras(spec, ("answer_prefixes", "think_pattern", "letter_class", "fullwidth_punct"))


def _strip_think(reply: str, spec: TaskSpec | None = None) -> str:
    pat = getattr(spec, "think_pattern", None)
    rx = _THINK if pat is None else re.compile(pat, re.I | re.S)
    return rx.sub("", reply or "").strip()


def _prefix_rx(spec: TaskSpec) -> str:
    pre = getattr(spec, "answer_prefixes", None)
    fw = getattr(spec, "fullwidth_punct", False)
    if pre is None and not fw:
        return _PREFIX
    alts = (r"(final\s+)?(answer|label|category|class|intent|output)" if pre is None
            else "(" + "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in pre) + ")")
    return r"(?i)^" + alts + (r"\s*[:=\-：]\s*" if fw else r"\s*[:=\-]\s*")


def _match_line(spec: TaskSpec, r: str) -> str | None:
    r = r.strip().strip("`*\"'").strip()
    r = re.sub(_prefix_rx(spec), "", r).strip().strip("`*\"'").strip()
    if not r:
        return None
    norm_names = {_norm(n): n for n in spec.names}
    if _norm(r) in norm_names:
        return norm_names[_norm(r)]
    # description match (e.g. reply "science and technology" for label "Sci/Tech")
    for n, d in spec.labels.items():
        if d and _norm(d) != _norm(n) and _norm(r) in (_norm(d), _norm(d.split(":")[0])):
            return n
    if all(len(n) == 1 for n in spec.names):  # single-letter label sets (A-D, E/S/C/I, ...)
        for n, d in spec.labels.items():
            head = _norm(d.split(":")[0]) if d else ""
            if head and head != _norm(n) and re.match(re.escape(head) + r"\b", _norm(r)):
                return n
        cls = getattr(spec, "letter_class", "A-Z")
        if cls == "A-Z":
            letters = set(re.findall(r"(?<![A-Za-z0-9'])([A-Z])(?![A-Za-z0-9'])", r))
        else:
            letters = set(re.findall(r"(?<![\w'])([" + cls + r"])(?![\w'])", r))
        letters &= set(spec.names)
        if len(letters) == 1:
            return letters.pop()
        if (re.fullmatch(r"[A-Za-z]\W*", r) if cls == "A-Z" else re.fullmatch(r"(?i)[" + cls + r"]\W*", r)) \
                and r[0].upper() in spec.names:
            return r[0].upper()
        return None
    rn = _norm(r)
    hits = [n for n in spec.names if re.search(r"(?<![\w])" + re.escape(_norm(n)) + r"(?![\w])", rn)]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:  # only NESTED names (e.g. 'card' inside 'card arrival') resolve to the longest
        hits.sort(key=lambda h: len(_norm(h)), reverse=True)
        if all(_norm(h) in _norm(hits[0]) for h in hits[1:]):
            return hits[0]
    return None


def parse_label(spec: TaskSpec, reply: str | None) -> str | None:
    """Map a free-text reply to a label name; None if it cannot be mapped unambiguously.
    Strips <think> blocks, then tries the first line, the last line, then the whole reply."""
    r = _strip_think(reply or "", spec)
    if not r:
        return None
    lines = [l for l in r.splitlines() if l.strip()]
    for cand in ([lines[0], lines[-1]] if lines else []) + [r.replace("\n", " ")]:
        got = _match_line(spec, cand)
        if got is not None:
            return got
    return None


_LINE = re.compile(r"^\s*(?:[-•]\s+)?[*_]*\s*(?:item\s*|#|\[)?\s*(\d+)\s*\]?[*_]*\s*(?:[:.)\-–=]|\s)\s*(.+?)\s*$", re.I)
_LINE_FW = re.compile(r"^\s*(?:[-•]\s+)?[*_]*\s*(?:item\s*|#|\[)?\s*(\d+)\s*\]?[*_]*\s*(?:[:.)\-–=：．）]|\s)\s*(.+?)\s*$",
                      re.I)


def parse_batch(spec: TaskSpec, reply: str | None, n: int) -> list[str | None]:
    """Parse '<number>: <label>' lines (tolerant of bullets, 'Item 3:', '#3', '[3]', '3 label', one-line
    comma lists). Falls back to positional parsing when nothing is numbered and there are exactly n lines."""
    text = _strip_think(reply or "", spec)
    text = re.sub(r",\s*(?=\d+\s*[:.)\]])", "\n", text)  # '1: a, 2: b' -> lines
    got: dict[int, str | None] = {}
    numbered = 0
    line_rx = _LINE_FW if getattr(spec, "fullwidth_punct", False) else _LINE
    for line in text.splitlines():
        m = line_rx.match(line)
        if m:
            numbered += 1
            pos = int(m.group(1)) - 1
            if 0 <= pos < n and pos not in got:
                got[pos] = parse_label(spec, m.group(2))
    if numbered == 0:
        lines = [l for l in text.splitlines() if l.strip()]
        if len(lines) == n:
            return [parse_label(spec, l) for l in lines]
    return [got.get(i) for i in range(n)]


# ----------------------------------------------------------------------------------------------- custom workloads
@dataclass
class Workload:
    spec: TaskSpec
    items: list[dict] = field(default_factory=list)

    @property
    def texts(self) -> list[str]:
        return [it["text"] for it in self.items]

    @property
    def gold(self) -> list[str | None]:
        return [it.get("gold") for it in self.items]

    def subset(self, start: int, n: int | None) -> "Workload":
        return Workload(self.spec, self.items[start:start + n] if n else self.items[start:])


def from_records(spec: TaskSpec, records: list[dict], text_key: str = "text", label_key: str | None = "label",
                 id_key: str | None = None) -> Workload:
    items = []
    for i, r in enumerate(records):
        g = r.get(label_key) if label_key else None
        items.append({"id": str(r[id_key]) if id_key else str(i), "text": str(r[text_key]),
                      "gold": (str(g).strip() if g not in (None, "") else None)})
    return Workload(spec, items)


def from_jsonl(spec: TaskSpec, path: str | Path, **kw) -> Workload:
    return from_records(spec, [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()], **kw)


def from_csv(spec: TaskSpec, path: str | Path, **kw) -> Workload:
    with Path(path).open(encoding="utf-8", newline="") as f:
        return from_records(spec, list(csv.DictReader(f)), **kw)


# ----------------------------------------------------------------------------------------------- HF registry
# Each entry: hf id, splits, how to build text (template over row fields), how to get the label NAME.
#   label: {"field": f}                    -> use row[f] as the label name
#          {"field": f, "names": [...]}    -> row[f] is an int index into names
#          {"field": f, "classlabel": True}-> read names from the dataset's ClassLabel feature
REGISTRY: dict[str, dict] = {
    "agnews": {"hf": "fancyzhx/ag_news", "splits": ["train", "test"], "text": "{text}",
               "label": {"field": "label", "names": ["World", "Sports", "Business", "Sci/Tech"]},
               "instruction": "Classify the news article into exactly one topic.",
               "descriptions": {"World": "world news, politics, international affairs", "Sports": "sports",
                                "Business": "business, economy, companies, markets", "Sci/Tech": "science and technology"}},
    "emotion": {"hf": "dair-ai/emotion", "splits": ["train", "validation", "test"], "text": "{text}",
                "label": {"field": "label", "classlabel": True},
                "instruction": "Classify the main emotion expressed in the text."},
    "banking77": {"hf": "mteb/banking77", "splits": ["train", "test"], "text": "{text}", "label": {"field": "label_text"},
                  "instruction": "Classify the bank customer's message into exactly one intent."},
    "trec": {"hf": "SetFit/TREC-QC", "splits": ["train", "test"], "text": "{text}", "label": {"field": "label_coarse_text"},
             "instruction": "Classify what kind of answer the question is asking for."},
    "dbpedia": {"hf": "fancyzhx/dbpedia_14", "splits": ["test"], "text": "{title}. {content}",
                "label": {"field": "label", "classlabel": True},
                "instruction": "Classify the Wikipedia entity described by the text into exactly one category."},
    "sst2": {"hf": "SetFit/sst2", "splits": ["train", "validation", "test"], "text": "{text}", "label": {"field": "label_text"},
             "instruction": "Classify the sentiment of the movie review sentence."},
    "sst5": {"hf": "SetFit/sst5", "splits": ["train", "validation", "test"], "text": "{text}", "label": {"field": "label_text"},
             "instruction": "Rate the sentiment of the movie review sentence."},
    "tweet_sentiment": {"hf": "mteb/tweet_sentiment_extraction", "splits": ["train", "test"], "text": "{text}",
                        "label": {"field": "label_text"}, "instruction": "Classify the sentiment of the tweet."},
    "clinc150": {"hf": "clinc/clinc_oos", "config": "plus", "splits": ["train", "validation", "test"], "text": "{text}",
                 "label": {"field": "intent", "classlabel": True},
                 "instruction": "Classify the user's request to a virtual assistant into exactly one intent."},
}


def _build_hf(name: str) -> tuple[list[dict], TaskSpec]:
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError(f"registry task {name!r} is not cached in {DATA_DIR} and downloading it needs the "
                          "'datasets' package: pip install datasets (or set BOTTLEKIT_DATA_DIR to a directory that "
                          f"holds bk_{name}.jsonl and bk_{name}.spec.json)") from err
    e = REGISTRY[name]
    rows, names_seen = [], []
    names = None
    for split in e["splits"]:
        ds = load_dataset(e["hf"], e.get("config"), split=split)
        lab = e["label"]
        names = lab.get("names")
        if lab.get("classlabel"):
            names = ds.features[lab["field"]].names
        for i, r in enumerate(ds):
            v = r[lab["field"]]
            g = names[int(v)] if names is not None else str(v)
            rows.append({"id": f"{split}-{i}", "text": e["text"].format(**r), "gold": g})
            if g not in names_seen:
                names_seen.append(g)
    order = names if names is not None else sorted(names_seen)
    desc = e.get("descriptions", {})
    spec = TaskSpec(name=name, instruction=e["instruction"], labels={n: desc.get(n, n.replace("_", " ")) for n in order})
    return rows, spec


def load_workload(name: str, n: int | None = None, offset: int = 0, seed: int = 0) -> Workload:
    """Deterministically shuffled workload from the registry (cached to data/bk_<name>.jsonl)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path, spec_path = DATA_DIR / f"bk_{name}.jsonl", DATA_DIR / f"bk_{name}.spec.json"
    if not path.exists() and name not in REGISTRY:
        raise ValueError(f"unknown task {name!r}: registry tasks are {sorted(REGISTRY)}; for your own data use "
                         "from_jsonl / from_csv / from_records (CLI: --jsonl)")
    if not path.exists():
        rows, spec = _build_hf(name)
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        spec_path.write_text(json.dumps({"name": spec.name, "instruction": spec.instruction, "labels": spec.labels}, ensure_ascii=False), encoding="utf-8")
    items = [json.loads(l) for l in path.open(encoding="utf-8")]
    sd = json.loads(spec_path.read_text(encoding="utf-8"))
    random.Random(seed).shuffle(items)
    items = items[offset:offset + n] if n else items[offset:]
    return Workload(TaskSpec(**sd), items)
