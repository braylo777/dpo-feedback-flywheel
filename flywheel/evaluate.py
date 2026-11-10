"""Before/after evaluation on a held-out preference set and held-out generations."""
from __future__ import annotations

import statistics

from . import rubric


def preference_scores(model, tok, pairs: list[dict]) -> list[dict]:
    from .modeling import response_logprob

    rows = []
    for p in pairs:
        lp_c, len_c = response_logprob(model, tok, p["prompt"], p["chosen"])
        lp_r, len_r = response_logprob(model, tok, p["prompt"], p["rejected"])
        rows.append({"lp_c": lp_c, "len_c": len_c, "lp_r": lp_r, "len_r": len_r})
    return rows


def pref_accuracy(rows: list[dict]) -> dict:
    """Length-normalised log-prob accuracy: P(mean logp chosen > mean logp rejected)."""
    wins = [r["lp_c"] / r["len_c"] > r["lp_r"] / r["len_r"] for r in rows]
    return {"n": len(rows), "acc_len_norm": sum(wins) / len(rows)}


def implicit_reward_accuracy(policy_rows: list[dict], ref_rows: list[dict], beta: float) -> dict:
    """DPO implicit reward r = beta*(logp_policy - logp_ref). Accuracy = P(r_chosen > r_rejected)."""
    margins = []
    for p, r in zip(policy_rows, ref_rows):
        r_ch = beta * (p["lp_c"] - r["lp_c"])
        r_rej = beta * (p["lp_r"] - r["lp_r"])
        margins.append(r_ch - r_rej)
    return {"acc": sum(m > 0 for m in margins) / len(margins), "mean_margin": statistics.fmean(margins)}


def score_outputs(prompts_with_gold: list[tuple[str, str | None]], outs: list[str]) -> dict:
    samples, passes, words, opener, phrase, gold_hits, gold_n = [], 0, [], 0, 0, 0, 0
    for (p, gold), o in zip(prompts_with_gold, outs):
        ok = rubric.passes(o)
        passes += ok
        words.append(rubric.word_count(o))
        opener += rubric.has_filler_opener(o)
        phrase += rubric.has_filler_phrase(o)
        if gold is not None:
            gold_n += 1
            gold_hits += gold.lower() in o.lower()
        samples.append({"prompt": p, "output": o, "passes_rubric": ok})
    n = len(outs)
    return {
        "n": n,
        "rubric_pass_rate": passes / n,
        "mean_words": statistics.fmean(words),
        "median_words": statistics.median(words),
        "filler_opener_rate": opener / n,
        "filler_phrase_rate": phrase / n,
        "gold_answer_rate": (gold_hits / gold_n) if gold_n else None,
        "gold_n": gold_n,
        "samples": samples,
    }


def generation_eval(model, tok, prompts_with_gold, batch_size: int = 8, max_new_tokens: int = 96) -> dict:
    from .modeling import generate_batch

    prompts = [p for p, _ in prompts_with_gold]
    outs: list[str] = []
    for i in range(0, len(prompts), batch_size):
        outs.extend(generate_batch(model, tok, prompts[i:i + batch_size], max_new_tokens))
    return score_outputs(prompts_with_gold, outs)
