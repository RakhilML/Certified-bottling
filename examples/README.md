# Examples

| Script | Needs | What it does |
|---|---|---|
| `offline_demo.py` | Nothing (no models, no network) | A simulator replaces the cheap model and the expert. A cold start on 1,200 items, then a cold pour of the bottle on 1,200 new items. |
| `quickstart_endpoint.py` | Your models file and a JSONL file of items | A cold start on your items with your models. It saves the bottle to `bottle.pkl`. |

```bash
python examples/offline_demo.py
python examples/quickstart_endpoint.py my_models.yaml my_items.jsonl
```

`quickstart_endpoint.py` calls your models. On a paid API it costs money, so start with a few hundred items.
