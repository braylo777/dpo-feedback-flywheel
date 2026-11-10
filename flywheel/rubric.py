"""Simulated-user rubric. SYNTHETIC: this stands in for real users' thumbs.

The simulated users like direct, short answers and dislike sycophantic openers,
filler sign-offs and rambling. This is a stated, checkable preference so the
before/after eval measures whether DPO moved the model toward it.
"""
from __future__ import annotations

import re

FILLER_OPENERS = (
    "great question", "what a great", "certainly", "sure", "of course", "absolutely",
    "i'd be happy", "i would be happy", "as an ai", "thank you for asking", "excellent question",
)
FILLER_PHRASES = (
    "i hope this helps", "let me know if you have any", "feel free to ask", "happy to help",
    "as an ai language model", "great question",
)
MAX_WORDS = 40


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def has_filler_opener(text: str) -> bool:
    head = text.strip().lower().lstrip("\"'*# ")
    return any(head.startswith(f) for f in FILLER_OPENERS)


def has_filler_phrase(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in FILLER_PHRASES)


def passes(text: str) -> bool:
    """True if a simulated user would give thumbs-up."""
    if not text.strip():
        return False
    return word_count(text) <= MAX_WORDS and not has_filler_opener(text) and not has_filler_phrase(text)


def vote(text: str) -> int:
    return 1 if passes(text) else -1
