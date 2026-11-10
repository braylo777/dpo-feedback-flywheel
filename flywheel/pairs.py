"""Convert thumbs-up/down feedback into DPO preference pairs."""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

from .store import Rated


def normalize_prompt(p: str) -> str:
    return " ".join(p.strip().lower().split())


def build_pairs(rated: list[Rated], exclude_prompts: set[str] | None = None, max_pairs_per_prompt: int = 4) -> list[dict]:
    """For each prompt with >=1 up-voted and >=1 down-voted response, emit (chosen, rejected) pairs.

    - Prompts are grouped after whitespace/case normalisation.
    - Identical chosen/rejected texts are skipped (contradictory votes on the same text).
    - `exclude_prompts` keeps held-out eval prompts out of training data.
    """
    exclude = {normalize_prompt(p) for p in (exclude_prompts or set())}
    groups: dict[str, dict] = defaultdict(lambda: {"prompt": None, "up": [], "down": []})
    for r in rated:
        key = normalize_prompt(r.prompt)
        if key in exclude:
            continue
        g = groups[key]
        g["prompt"] = g["prompt"] or r.prompt.strip()
        (g["up"] if r.vote == 1 else g["down"]).append(r.response.strip())

    pairs = []
    for key in sorted(groups):
        g = groups[key]
        ups = list(dict.fromkeys(g["up"]))
        downs = list(dict.fromkeys(g["down"]))
        n = 0
        for c, rj in itertools.product(ups, downs):
            if c == rj:
                continue
            pairs.append({"prompt": g["prompt"], "chosen": c, "rejected": rj})
            n += 1
            if n >= max_pairs_per_prompt:
                break
    return pairs


def to_conversational(pair: dict) -> dict:
    """TRL conversational preference format."""
    return {
        "prompt": [{"role": "user", "content": pair["prompt"]}],
        "chosen": [{"role": "assistant", "content": pair["chosen"]}],
        "rejected": [{"role": "assistant", "content": pair["rejected"]}],
    }


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
