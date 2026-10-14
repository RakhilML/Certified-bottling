"""Offline demo: bottlekit end to end with a simulated cheap model and a simulated expert.

No model server, no network and no API key are necessary. A small simulator replaces the two models:
  * the cheap model is correct on about 85% of the items (less at the start of each batch),
  * the expert is correct on about 95% of the items.

The demo does a cold start on 1,200 items and then reuses the bottle on 1,200 new items (a cold pour).

    python examples/offline_demo.py
"""
import asyncio
import hashlib
import random
import tempfile
from pathlib import Path

from bottlekit import (BottleArtifact, BottleConfig, Bottler, ChatResult, LLMClient, ModelSpec, TaskSpec, evaluate,
                       from_records)

SPEC = TaskSpec(
    name="tickets",
    instruction="Classify the support ticket.",
    labels={"billing": "billing: charges, invoices, refunds",
            "delivery": "delivery: shipping, parcels, tracking",
            "account": "account: login, password, profile",
            "technical": "technical: errors, crashes, bugs"},
)
WORDS = {"billing": ["invoice", "refund", "charged", "payment"],
         "delivery": ["parcel", "shipping", "tracking", "courier"],
         "account": ["login", "password", "profile", "email"],
         "technical": ["error", "crash", "bug", "freeze"]}
FILLER = ["please", "help", "today", "again", "order", "still", "since", "monday"]


def make_tickets(n: int, seed: int) -> list[dict]:
    """Short synthetic tickets: two words of the true topic, one random topic word, filler and an id."""
    rng = random.Random(seed)
    every_word = sum(WORDS.values(), [])
    rows = []
    for i in range(n):
        label = rng.choice(list(WORDS))
        tokens = rng.sample(WORDS[label], 2) + [rng.choice(every_word)] + rng.sample(FILLER, 2) + [f"t{seed}-{i}"]
        rng.shuffle(tokens)
        rows.append({"text": " ".join(tokens), "label": label})
    return rows


class SimulatedClient(LLMClient):
    """An LLMClient whose chat() answers from the gold labels with a fixed accuracy per model (no HTTP calls).

    Single-item prompts get one answer. Batched prompts ("1. item", "2. item", ...) get one numbered answer per item,
    and the first quarter of each batch is less accurate (the cold start of a weak model)."""

    def __init__(self, gold: dict[str, str], accuracy: dict[str, float], cold_start_drop: float = 0.15):
        super().__init__({"cheap": ModelSpec("cheap", ["cheap"], price_in=0.05, price_out=0.10),
                          "expert": ModelSpec("expert", ["expert"], price_in=2.00, price_out=8.00)})
        self.gold, self.accuracy, self.drop = gold, accuracy, cold_start_drop

    def answer(self, alias: str, text: str, salt: str, position: int | None = None, batch_size: int = 0) -> str:
        truth = self.gold[text]
        rng = random.Random(int(hashlib.md5(f"{alias}|{text}|{salt}".encode()).hexdigest()[:8], 16))
        acc = self.accuracy[alias]
        if position is not None and position < max(1, batch_size // 4):
            acc -= self.drop
        return truth if rng.random() < acc else rng.choice([n for n in SPEC.names if n != truth])

    async def chat(self, alias, messages, max_tokens=256, temperature=0.0, phase="default", seed=None):
        user = messages[-1]["content"]
        if user.startswith("1. "):                                   # a batched prompt
            items = [part.split(". ", 1)[1] for part in user.split("\n\n")]
            salt = hashlib.md5(user.encode()).hexdigest()
            text = "\n".join(f"{i + 1}: {self.answer(alias, t, salt, i, len(items))}" for i, t in enumerate(items))
            prompt_tokens, completion_tokens = 40 * len(items), 6 * len(items)
        else:                                                         # a single-item prompt
            text = self.answer(alias, user, "single")
            prompt_tokens, completion_tokens = 60, 3
        result = ChatResult(text=text, ok=True, model=alias, prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            usd=self.models[alias].usd(prompt_tokens, completion_tokens))
        self.ledger.add(phase, result)
        return result


def report(title: str, client: SimulatedClient, result, rows: list[dict]) -> None:
    texts = [r["text"] for r in rows]
    gold = [r["label"] for r in rows]
    expert_all = [client.answer("expert", t, "single") for t in texts]      # evaluation only
    ev = evaluate(result, gold, expert_all)
    spent = sum(row["usd"] for row in result.ledger.values())
    expert_every_item = len(texts) * client.models["expert"].usd(60, 3)
    cert = result.certificate
    print(f"\n== {title}")
    print(f"  guarantee          : {cert['guarantee']}")
    print(f"  calibration sample : {cert['n_cal']} of {len(texts)} items")
    print(f"  bottled share      : {result.coverage():.3f} (answered without the expert)")
    print(f"  cost               : {spent / expert_every_item:.3f} of asking the expert about every item")
    print(f"  realized disagreement with the expert: {ev['realized_disagreement_with_expert']:.3f}")


async def main() -> None:
    first, second = make_tickets(1200, seed=1), make_tickets(1200, seed=2)
    client = SimulatedClient({r["text"]: r["label"] for r in first + second},
                             accuracy={"cheap": 0.85, "expert": 0.95})
    cfg = BottleConfig(cheap="cheap", expert="expert", eps=0.05, delta=0.1, seed=0)

    workload = from_records(SPEC, first)
    cold = await Bottler(client, SPEC, cfg).run(workload.texts)
    report("cold start (cheap votes + student + certificate)", client, cold, first)

    poured = await Bottler(client, SPEC, cfg).pour(cold.artifact, [r["text"] for r in second], votes=False)
    report("reuse: cold pour of the bottle on new items", client, poured, second)

    path = Path(tempfile.gettempdir()) / "tickets_bottle.pkl"
    poured.artifact.save(path)
    bottle = BottleArtifact.load(path)
    print(f"\nThe updated bottle ({len(bottle.texts)} certified answers in memory) is saved to {path}")
    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
