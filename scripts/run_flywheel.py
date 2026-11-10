"""End-to-end flywheel run on SYNTHETIC feedback.

1. Base model answers the train prompts (greedy + 1 sample) -> on-policy turns.
2. Templated candidate replies are logged too (direct / sycophantic / hedging).
3. A simulated-user rubric gives thumbs (10% label noise) -> SQLite feedback DB.
4. Build preference pairs (held-out prompts excluded); check the schedule gate.
5. Train DPO LoRA; evaluate base vs DPO on two held-out preference sets and on
   held-out generations. Write results/<run>.json (committed); adapters go to
   checkpoints/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flywheel import evaluate, schedule, synth  # noqa: E402
from flywheel.pairs import build_pairs, write_jsonl  # noqa: E402
from flywheel.store import FeedbackStore  # noqa: E402


def gen_all(model, tok, prompts, sample=False, bs=8):
    from flywheel.modeling import generate_batch

    out = []
    for i in range(0, len(prompts), bs):
        out.extend(generate_batch(model, tok, prompts[i : i + bs], sample=sample))
    return out


def pref_block(model, tok, sets):
    return {
        name: evaluate.preference_scores(model, tok, pairs)
        for name, pairs in sets.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--db", default=None)
    ap.add_argument("--state", default=str(ROOT / "data" / "schedule_state.json"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "run"))
    ap.add_argument("--results", default=str(ROOT / "results" / "run.json"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--min-new-pairs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import torch

    from flywheel.modeling import load_model

    t0 = time.time()
    torch.manual_seed(a.seed)
    train_items, heldout_items = synth.split_items(seed=0)
    train_prompts = [p for _, p, _, _ in train_items]
    gen_set = [(p, gold) for _, p, _, gold in heldout_items]

    base, tok = load_model(a.model)
    greedy = gen_all(base, tok, train_prompts)
    sampled = gen_all(base, tok, train_prompts, sample=True)

    # Fresh DB per run (no deletes): data/synthetic_feedback.<timestamp>.db, gitignored.
    db = a.db or str(
        ROOT / "data" / f"synthetic_feedback.{time.strftime('%Y%m%d-%H%M%S')}.db"
    )
    store = FeedbackStore(db)
    seed_templated = synth.seed_store(store, train_items, noise=a.noise)
    seed_on_policy = synth.seed_store_on_policy(
        store,
        list(zip(train_prompts, greedy)) + list(zip(train_prompts, sampled)),
        noise=a.noise,
    )
    pairs = build_pairs(store.rated(), exclude_prompts={p for p, _ in gen_set})
    write_jsonl(pairs, ROOT / "data" / "train_pairs.synthetic.jsonl")
    print(f"feedback: {store.stats()}  train pairs: {len(pairs)}")

    st = schedule.load_state(a.state)
    due, why = schedule.is_due(st, len(pairs), min_new_pairs=a.min_new_pairs)
    print(f"schedule gate: due={due} ({why})")
    if not (due or a.force):
        return

    base_gen = evaluate.generation_eval(base, tok, gen_set)
    base_out = {s["prompt"]: s["output"] for s in base_gen["samples"]}
    sets = {
        "templated": synth.heldout_pairs(heldout_items),
        "on_policy": synth.on_policy_heldout_pairs(heldout_items, base_out),
    }
    for name, ps in sets.items():
        write_jsonl(ps, ROOT / "data" / f"heldout_pairs.{name}.synthetic.jsonl")
    base_rows = pref_block(base, tok, sets)
    print(
        "base:",
        {k: evaluate.pref_accuracy(v) for k, v in base_rows.items()},
        f"pass={base_gen['rubric_pass_rate']:.3f}",
    )
    del base
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    from flywheel.train import train_dpo

    tr = train_dpo(
        pairs, a.model, a.out, beta=a.beta, epochs=a.epochs, lr=a.lr, seed=a.seed
    )
    print(
        f"trained: steps={tr['steps']} loss={tr['train_loss']:.4f} wall={tr['wall_seconds']:.0f}s"
    )

    dpo, tok = load_model(a.model, tr["adapter_dir"])
    dpo_rows = pref_block(dpo, tok, sets)
    dpo_gen = evaluate.generation_eval(dpo, tok, gen_set)
    print(
        "dpo:",
        {k: evaluate.pref_accuracy(v) for k, v in dpo_rows.items()},
        f"pass={dpo_gen['rubric_pass_rate']:.3f}",
    )

    schedule.save_state(schedule.mark_ran(st, len(pairs)), a.state)
    tr.pop("adapter_dir", None)
    out = {
        "data_label": "SYNTHETIC: simulated-user thumbs from flywheel/rubric.py on templated + model-generated replies; not real users",
        "date": time.strftime("%Y-%m-%d"),
        "machine": f"{platform.machine()} {platform.system()} {platform.mac_ver()[0]}",
        "torch": torch.__version__,
        "seed": a.seed,
        "seed_feedback": {"templated": seed_templated, "on_policy": seed_on_policy},
        "feedback_stats": store.stats(),
        "n_train_pairs": len(pairs),
        "n_heldout_prompts": len(gen_set),
        "train": tr,
        "heldout_preference": {
            name: {
                "n": len(sets[name]),
                "base": evaluate.pref_accuracy(base_rows[name]),
                "dpo": evaluate.pref_accuracy(dpo_rows[name]),
                "dpo_implicit_reward": evaluate.implicit_reward_accuracy(
                    dpo_rows[name], base_rows[name], a.beta
                ),
            }
            for name in sets
        },
        "heldout_generation": {"base": base_gen, "dpo": dpo_gen},
        "total_wall_seconds": time.time() - t0,
    }
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    Path(a.results).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.results}")


if __name__ == "__main__":
    main()
