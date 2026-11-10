"""SQLite store for chat turns and thumbs-up/down feedback."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'base',
    source TEXT NOT NULL DEFAULT 'human',      -- 'human' | 'synthetic'
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id INTEGER NOT NULL REFERENCES turns(id),
    vote INTEGER NOT NULL CHECK (vote IN (-1, 1)),
    created_at REAL NOT NULL
);
"""


@dataclass
class Rated:
    turn_id: int
    prompt: str
    response: str
    vote: int
    source: str
    model_version: str


class FeedbackStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def add_turn(self, prompt: str, response: str, model_version: str = "base", source: str = "human") -> int:
        if not prompt.strip() or not response.strip():
            raise ValueError("prompt and response must be non-empty")
        if source not in ("human", "synthetic"):
            raise ValueError("source must be 'human' or 'synthetic'")
        cur = self.conn.execute(
            "INSERT INTO turns(prompt, response, model_version, source, created_at) VALUES (?,?,?,?,?)",
            (prompt, response, model_version, source, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def vote(self, turn_id: int, vote: int) -> None:
        if vote not in (-1, 1):
            raise ValueError("vote must be -1 or 1")
        row = self.conn.execute("SELECT id FROM turns WHERE id=?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown turn {turn_id}")
        # Latest vote wins: replace any earlier vote on the same turn.
        self.conn.execute("DELETE FROM feedback WHERE turn_id=?", (turn_id,))
        self.conn.execute(
            "INSERT INTO feedback(turn_id, vote, created_at) VALUES (?,?,?)", (turn_id, vote, time.time())
        )
        self.conn.commit()

    def rated(self) -> list[Rated]:
        rows = self.conn.execute(
            """SELECT t.id, t.prompt, t.response, f.vote, t.source, t.model_version
               FROM turns t JOIN feedback f ON f.turn_id = t.id ORDER BY t.id"""
        ).fetchall()
        return [Rated(*r) for r in rows]

    def stats(self) -> dict:
        n_turns = self.conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        up = self.conn.execute("SELECT COUNT(*) FROM feedback WHERE vote=1").fetchone()[0]
        down = self.conn.execute("SELECT COUNT(*) FROM feedback WHERE vote=-1").fetchone()[0]
        return {"turns": n_turns, "up": up, "down": down}
