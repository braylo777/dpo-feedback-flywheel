"""Offline tests: no model download needed."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.server import create_app  # noqa: E402
from flywheel import evaluate, rubric, schedule, synth  # noqa: E402
from flywheel.pairs import (
    build_pairs,
    read_jsonl,
    to_conversational,
    write_jsonl,
)  # noqa: E402
from flywheel.store import FeedbackStore, Rated  # noqa: E402


def R(prompt, resp, vote):
    return Rated(0, prompt, resp, vote, "synthetic", "base")


def test_store_vote_latest_wins():
    s = FeedbackStore()
    t = s.add_turn("q", "a")
    s.vote(t, 1)
    s.vote(t, -1)
    assert [r.vote for r in s.rated()] == [-1]
    assert s.stats() == {"turns": 1, "up": 0, "down": 1}


def test_store_rejects_bad_input():
    s = FeedbackStore()
    with pytest.raises(ValueError):
        s.add_turn("", "a")
    with pytest.raises(ValueError):
        s.vote(s.add_turn("q", "a"), 0)
    with pytest.raises(KeyError):
        s.vote(999, 1)


def test_pairs_need_up_and_down():
    pairs = build_pairs([R("Q?", "good", 1), R("Q?", "bad", -1), R("Only up?", "x", 1)])
    assert pairs == [{"prompt": "Q?", "chosen": "good", "rejected": "bad"}]


def test_pairs_normalise_prompt_and_skip_identical():
    pairs = build_pairs(
        [
            R("What is X?", "same", 1),
            R("what is  x?", "same", -1),
            R("WHAT IS X?", "other", -1),
        ]
    )
    assert len(pairs) == 1 and pairs[0]["rejected"] == "other"


def test_pairs_exclude_heldout_and_cap():
    rated = [R("Q", f"up{i}", 1) for i in range(5)] + [
        R("Q", f"dn{i}", -1) for i in range(5)
    ]
    assert len(build_pairs(rated, max_pairs_per_prompt=3)) == 3
    assert build_pairs(rated, exclude_prompts={"q"}) == []


def test_conversational_format_and_jsonl(tmp_path):
    c = to_conversational({"prompt": "p", "chosen": "c", "rejected": "r"})
    assert c["prompt"][0]["role"] == "user" and c["chosen"][0]["role"] == "assistant"
    write_jsonl([c], tmp_path / "x.jsonl")
    assert read_jsonl(tmp_path / "x.jsonl") == [c]


def test_rubric():
    assert rubric.passes("The capital of France is Paris.")
    assert not rubric.passes("Great question! The capital of France is Paris.")
    assert not rubric.passes("Paris. I hope this helps!")
    assert not rubric.passes(" ".join(["word"] * 41))
    assert not rubric.passes("   ")


def test_synth_split_is_disjoint_and_heldout_pairs_valid():
    train, held = synth.split_items(seed=0)
    assert not {p for _, p, _, _ in train} & {p for _, p, _, _ in held}
    hp = synth.heldout_pairs(held)
    assert len(hp) == 2 * len(held)
    assert all(
        rubric.passes(p["chosen"]) and not rubric.passes(p["rejected"]) for p in hp
    )


def test_synth_seed_store_labels_synthetic_and_noise():
    train, _ = synth.split_items(seed=0)
    s = FeedbackStore()
    info = synth.seed_store(s, train, noise=0.0)
    rated = s.rated()
    assert info["flipped_by_noise"] == 0 and all(r.source == "synthetic" for r in rated)
    assert all(r.vote == rubric.vote(r.response) for r in rated)
    s2 = FeedbackStore()
    assert synth.seed_store(s2, train, noise=0.5)["flipped_by_noise"] > 0


def test_on_policy_heldout_pairs_only_failing_outputs():
    _, held = synth.split_items(seed=0)
    p0, p1 = held[0][1], held[1][1]
    outs = {p0: "Great question! " + "blah " * 50, p1: "Short answer."}
    pairs = synth.on_policy_heldout_pairs(held, outs)
    assert [p["prompt"] for p in pairs] == [p0]


def test_schedule_gate():
    st = schedule.State()
    assert schedule.is_due(st, 10, now=1000, min_new_pairs=50)[0] is False
    assert schedule.is_due(st, 60, now=1000, min_new_pairs=50)[0] is True
    st2 = schedule.mark_ran(st, 60, now=1000)
    assert (
        schedule.is_due(st2, 200, now=1000 + 60, min_interval_s=3600)[0] is False
    )  # cool-down
    assert schedule.is_due(st2, 200, now=1000 + 7200, min_interval_s=3600)[0] is True
    assert (
        schedule.is_due(st2, 100, now=1000 + 7200, min_new_pairs=50)[0] is False
    )  # 40 new only


def test_schedule_state_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    assert schedule.load_state(p) == schedule.State()
    st = schedule.mark_ran(schedule.State(), 5, now=1.0)
    schedule.save_state(st, p)
    assert schedule.load_state(p) == st


def test_eval_metrics():
    rows = [
        {"lp_c": -2, "len_c": 2, "lp_r": -10, "len_r": 5},
        {"lp_c": -9, "len_c": 3, "lp_r": -2, "len_r": 2},
    ]
    assert evaluate.pref_accuracy(rows)["acc_len_norm"] == 0.5
    pol = [{"lp_c": -1, "lp_r": -5}]
    ref = [{"lp_c": -2, "lp_r": -4}]
    ir = evaluate.implicit_reward_accuracy(pol, ref, beta=0.1)
    assert ir["acc"] == 1.0 and ir["mean_margin"] == pytest.approx(0.2)
    s = evaluate.score_outputs(
        [("q", "Paris"), ("q2", None)], ["Paris is it.", "Sure! " + "x " * 50]
    )
    assert (
        s["rubric_pass_rate"] == 0.5
        and s["gold_answer_rate"] == 1.0
        and s["filler_opener_rate"] == 0.5
    )


def test_chat_app_flow():
    store = FeedbackStore()
    c = TestClient(create_app(store, lambda p: f"echo: {p}"))
    assert "thumbs" in c.get("/").text
    r = c.post("/api/chat", json={"prompt": "hi"}).json()
    assert r["reply"] == "echo: hi"
    assert c.post(
        "/api/feedback", json={"turn_id": r["turn_id"], "vote": 1}
    ).json() == {"ok": True}
    assert (
        c.post("/api/feedback", json={"turn_id": r["turn_id"], "vote": 5}).status_code
        == 422
    )
    assert c.post("/api/feedback", json={"turn_id": 999, "vote": 1}).status_code == 404
    assert c.post("/api/chat", json={"prompt": ""}).status_code == 422
    assert c.get("/api/stats").json() == {"turns": 1, "up": 1, "down": 0}
    assert store.rated()[0].source == "human"
