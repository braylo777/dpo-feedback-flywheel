"""SYNTHETIC feedback generator. Every row it creates is tagged source='synthetic'.

Builds templated prompts with a known gold answer, three candidate replies
(direct / sycophantic-verbose / hedging), and simulated-user thumbs from
`rubric.vote` with a configurable label-noise rate. Train and held-out prompts
use disjoint entities so the eval never sees a training prompt.
"""
from __future__ import annotations

import random

from . import rubric
from .store import FeedbackStore

CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Germany", "Berlin"),
    ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo"),
    ("Greece", "Athens"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Chile", "Santiago"), ("Cuba", "Havana"),
    ("Russia", "Moscow"), ("China", "Beijing"), ("India", "New Delhi"), ("Mexico", "Mexico City"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Argentina", "Buenos Aires"), ("Colombia", "Bogota"), ("Hungary", "Budapest"), ("Belgium", "Brussels"),
]
TERMS = [
    ("photosynthesis", "the process plants use to turn light, water and carbon dioxide into sugar and oxygen"),
    ("a CPU", "the chip that executes a computer's instructions"),
    ("an API", "a defined interface that lets one program request services from another"),
    ("gravity", "the force that pulls masses toward each other"),
    ("inflation", "a general rise in prices that lowers what money can buy"),
    ("a vaccine", "a preparation that trains the immune system to recognise a pathogen"),
    ("an atom", "the smallest unit of a chemical element"),
    ("DNA", "the molecule that carries genetic instructions in living things"),
    ("a mortgage", "a loan used to buy property, secured by that property"),
    ("a compiler", "a program that translates source code into machine code"),
    ("evaporation", "the change of a liquid into a gas below its boiling point"),
    ("a firewall", "a system that filters network traffic by security rules"),
    ("an ecosystem", "a community of organisms interacting with their environment"),
    ("a recession", "a significant decline in economic activity lasting months"),
    ("encryption", "encoding data so only authorised parties can read it"),
    ("a neuron", "a nerve cell that transmits electrical signals"),
    ("an algorithm", "a finite sequence of steps that solves a problem"),
    ("erosion", "the wearing away of land by water, wind or ice"),
    ("a protein", "a large molecule made of amino acid chains"),
    ("bandwidth", "the maximum rate of data transfer across a network"),
]
OPENERS = ["Great question!", "What a great question!", "Certainly!", "Sure! I'd be happy to help with that.",
           "Absolutely!", "Of course!", "Thank you for asking!"]
PADDING = [
    "This is something many people wonder about, and it is worth taking a moment to think it through carefully.",
    "There are a lot of interesting details here, so let me walk you through the answer step by step.",
    "It is always good to be curious about topics like this one, because learning never really stops.",
    "Of course, there is much more that could be said about this, depending on the context you have in mind.",
]
CLOSERS = ["I hope this helps!", "Let me know if you have any other questions!", "Feel free to ask if you need more detail!"]
HEDGES = ["As an AI, I can't be completely certain, but I believe", "I'm not entirely sure, but as an AI language model I think"]


def _items(rng: random.Random):
    items = []
    for country, cap in CAPITALS:
        items.append(("capital", f"What is the capital of {country}?", f"The capital of {country} is {cap}.", cap))
    for term, d in TERMS:
        items.append(("define", f"What is {term}?", f"{term[0].upper() + term[1:]} is {d}.", None))
    for _ in range(40):
        a, b = rng.randint(11, 99), rng.randint(11, 99)
        items.append(("add", f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}.", str(a + b)))
    for _ in range(30):
        km = rng.randint(2, 40)
        items.append(("convert", f"How many metres are in {km} kilometres?", f"{km} kilometres is {km * 1000} metres.", str(km * 1000)))
    # dedupe by prompt (random arithmetic can collide)
    seen, out = set(), []
    for it in items:
        if it[1] not in seen:
            seen.add(it[1]); out.append(it)
    return out


def candidates(direct: str, rng: random.Random) -> list[str]:
    verbose = " ".join([rng.choice(OPENERS), rng.choice(PADDING), direct, rng.choice(PADDING), rng.choice(CLOSERS)])
    body = direct[0].lower() + direct[1:]
    hedge = f"{rng.choice(HEDGES)} {body} {rng.choice(CLOSERS)}"
    return [direct, verbose, hedge]


def split_items(seed: int = 0, heldout_frac: float = 0.25):
    rng = random.Random(seed)
    items = _items(rng)
    rng.shuffle(items)
    k = int(len(items) * heldout_frac)
    return items[k:], items[:k]  # train, heldout


def heldout_pairs(heldout, seed: int = 1) -> list[dict]:
    """Noise-free held-out preference pairs (chosen = rubric-passing reply)."""
    rng = random.Random(seed)
    pairs = []
    for _, prompt, direct, _ in heldout:
        c = candidates(direct, rng)
        for rej in c[1:]:
            assert rubric.passes(c[0]) and not rubric.passes(rej)
            pairs.append({"prompt": prompt, "chosen": c[0], "rejected": rej})
    return pairs


def seed_store(store: FeedbackStore, train_items, noise: float = 0.1, seed: int = 2) -> dict:
    """Log synthetic turns + simulated thumbs. Returns counts incl. how many votes were flipped by noise."""
    rng = random.Random(seed)
    flipped = n = 0
    for _, prompt, direct, _ in train_items:
        for resp in candidates(direct, rng):
            tid = store.add_turn(prompt, resp, source="synthetic")
            v = rubric.vote(resp)
            if rng.random() < noise:
                v, flipped = -v, flipped + 1
            store.vote(tid, v)
            n += 1
    return {"votes": n, "flipped_by_noise": flipped, "noise": noise}


def seed_store_on_policy(store: FeedbackStore, outputs: list[tuple[str, str]], model_version: str = "base",
                         noise: float = 0.1, seed: int = 3) -> dict:
    """Log the MODEL's own replies (prompt, output) with simulated thumbs. Still synthetic votes."""
    rng = random.Random(seed)
    flipped = up = 0
    for prompt, out in outputs:
        if not out.strip():
            continue
        tid = store.add_turn(prompt, out, model_version=model_version, source="synthetic")
        v = rubric.vote(out)
        if rng.random() < noise:
            v, flipped = -v, flipped + 1
        up += v == 1
        store.vote(tid, v)
    return {"votes": len(outputs), "up": up, "flipped_by_noise": flipped, "noise": noise}


def on_policy_heldout_pairs(heldout, base_outputs: dict[str, str]) -> list[dict]:
    """Hard held-out set: chosen = direct reply, rejected = the BASE model's own greedy reply when it fails the rubric."""
    pairs = []
    for _, prompt, direct, _ in heldout:
        out = base_outputs.get(prompt, "")
        if out.strip() and not rubric.passes(out):
            pairs.append({"prompt": prompt, "chosen": direct, "rejected": out})
    return pairs
