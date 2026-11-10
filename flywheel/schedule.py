"""Decide when a scheduled run should retrain. Pure logic + a small JSON state file."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class State:
    last_run_ts: float = 0.0
    pairs_at_last_run: int = 0
    runs: int = 0


def load_state(path: str | Path) -> State:
    p = Path(path)
    if not p.exists():
        return State()
    return State(**json.loads(p.read_text()))


def save_state(state: State, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(state), indent=2))


def is_due(state: State, n_pairs: int, now: float | None = None,
           min_new_pairs: int = 50, min_interval_s: float = 6 * 3600) -> tuple[bool, str]:
    """Retrain only when enough NEW pairs arrived AND the cool-down passed."""
    now = time.time() if now is None else now
    new = n_pairs - state.pairs_at_last_run
    if new < min_new_pairs:
        return False, f"only {new} new pairs (< {min_new_pairs})"
    elapsed = now - state.last_run_ts
    if state.runs > 0 and elapsed < min_interval_s:
        return False, f"cool-down: {elapsed:.0f}s since last run (< {min_interval_s:.0f}s)"
    return True, f"{new} new pairs; due"


def mark_ran(state: State, n_pairs: int, now: float | None = None) -> State:
    return State(last_run_ts=time.time() if now is None else now, pairs_at_last_run=n_pairs, runs=state.runs + 1)
