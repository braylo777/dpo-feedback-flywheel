"""Tiny chat app: ask a question, get a reply, click thumbs-up/down. Feedback lands in SQLite."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from flywheel.store import FeedbackStore

INDEX = (Path(__file__).parent / "index.html").read_text()


class ChatIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)


class FeedbackIn(BaseModel):
    turn_id: int
    vote: int


def make_hf_generator(model_dir: str, adapter_dir: str | None = None) -> Callable[[str], str]:
    from flywheel.modeling import generate_one, load_model

    model, tok = load_model(model_dir, adapter_dir)
    return lambda prompt: generate_one(model, tok, prompt)


def create_app(store: FeedbackStore, generate: Callable[[str], str], model_version: str = "base") -> FastAPI:
    app = FastAPI(title="dpo-feedback-flywheel chat")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX

    @app.post("/api/chat")
    def chat(body: ChatIn):
        reply = generate(body.prompt).strip() or "(empty reply)"
        tid = store.add_turn(body.prompt, reply, model_version=model_version, source="human")
        return {"turn_id": tid, "reply": reply, "model_version": model_version}

    @app.post("/api/feedback")
    def feedback(body: FeedbackIn):
        if body.vote not in (-1, 1):
            raise HTTPException(422, "vote must be -1 or 1")
        try:
            store.vote(body.turn_id, body.vote)
        except KeyError:
            raise HTTPException(404, "unknown turn")
        return {"ok": True}

    @app.get("/api/stats")
    def stats():
        return store.stats()

    return app


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    model_id = os.environ.get("MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
    adapter = os.environ.get("ADAPTER")
    store = FeedbackStore(os.environ.get("DB", "data/feedback.db"))
    gen = make_hf_generator(model_id, adapter)
    uvicorn.run(create_app(store, gen, model_version="dpo" if adapter else "base"), host="127.0.0.1", port=8000)
