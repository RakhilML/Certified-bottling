"""Quick start with real models on an OpenAI-compatible endpoint (vLLM, LM Studio, OpenRouter, OpenAI, ...).

1. Copy bottlekit/models.yaml to my_models.yaml and set the model ids, base_url and prices of your provider.
   Put an API key in an environment variable and name that variable in `api_key_env` (never in the file).
2. Put your items in a JSONL file, one object per line: {"text": "..."} (a "label" field is optional and is used
   only for evaluation).
3. Run:
       python examples/quickstart_endpoint.py my_models.yaml my_items.jsonl

This script calls your models and costs money on a paid API. Start with a few hundred items.
"""
import asyncio
import json
import sys

from bottlekit import BottleConfig, Bottler, LLMClient, TaskSpec, from_jsonl, load_models

SPEC = TaskSpec(
    name="tickets",
    instruction="Classify the support ticket.",
    labels={"billing": "billing: charges, invoices, refunds",
            "delivery": "delivery: shipping, parcels, tracking",
            "account": "account: login, password, profile",
            "technical": "technical: errors, crashes, bugs"},
)


async def main(models_path: str, items_path: str) -> None:
    workload = from_jsonl(SPEC, items_path)
    client = LLMClient(load_models(models_path), cache_path=".bottlekit_cache.sqlite")
    cfg = BottleConfig(cheap="cheap", expert="expert", eps=0.05, delta=0.1)
    try:
        result = await Bottler(client, SPEC, cfg).run(workload.texts)
    finally:
        await client.close()
    print(json.dumps(result.certificate, indent=2, default=str))
    print(f"bottled share: {result.coverage():.3f}")
    print(f"spent: {sum(row['usd'] for row in result.ledger.values()):.4f} USD")
    if result.artifact is not None:
        result.artifact.save("bottle.pkl")
        print("bottle saved to bottle.pkl: reuse it with Bottler(...).pour(BottleArtifact.load('bottle.pkl'), new_texts)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python examples/quickstart_endpoint.py MODELS.yaml ITEMS.jsonl")
    asyncio.run(main(sys.argv[1], sys.argv[2]))
