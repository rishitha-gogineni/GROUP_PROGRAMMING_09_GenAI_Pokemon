"""SDK response adapter: deck phase, response validation and safe fallback. Owner: Chelsi.

Official response rules (sample_submission/main.py):
- When obs["select"] is None (initial phase) the agent returns the 60-card deck.
- Otherwise it returns a list of option indices with
  0 <= i < len(select["option"]), no duplicates, and
  select["minCount"] <= len(list) <= select["maxCount"].
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence

from .contracts import Observation

DECK_SIZE = 60
Policy = Callable[[dict], list]


def read_deck(path: str | Path) -> list[int]:
    """Read a deck file with one card ID per line (official deck.csv format)."""
    values = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    deck = [int(v) for v in values]
    if len(deck) != DECK_SIZE:
        raise ValueError(f"{path}: expected {DECK_SIZE} card IDs, found {len(deck)}")
    return deck


def is_deck_phase(observation: Observation) -> bool:
    return observation.get("select") is None


def validation_error(observation: Observation, response: Sequence[int]) -> str | None:
    """Return a reason string if `response` is not a legal selection, else None."""
    select = observation["select"]
    n = len(select["option"])
    if not isinstance(response, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in response):
        return "response must be list[int]"
    if len(set(response)) != len(response):
        return "duplicate indices"
    if any(i < 0 or i >= n for i in response):
        return f"index out of range 0..{n - 1}"
    if not (select["minCount"] <= len(response) <= select["maxCount"]):
        return f"count {len(response)} not in [{select['minCount']}, {select['maxCount']}]"
    return None


def random_valid_response(observation: Observation, rng: random.Random | None = None) -> list[int]:
    """Uniformly random legal response (also the safe fallback)."""
    rng = rng or random
    select = observation["select"]
    return rng.sample(range(len(select["option"])), select["maxCount"])


def top_k(scores: Sequence[float], k: int) -> list[int]:
    """Indices of the k highest scores; ties broken by lower index (deterministic)."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def safe_agent(policy: Policy, deck: list[int]) -> Policy:
    """Wrap a policy: return the deck in the deck phase, and fall back to a random
    legal response if the policy crashes or returns an illegal selection.
    `wrapped.stats` counts how often the fallback was needed."""

    stats = {"decisions": 0, "fallbacks": 0, "errors": []}

    def wrapped(observation: dict) -> list[int]:
        if is_deck_phase(observation):
            return list(deck)
        stats["decisions"] += 1
        try:
            response = [int(i) for i in policy(observation)]
            reason = validation_error(observation, response)
        except Exception as exc:  # keep playing; record the failure
            response, reason = None, f"{type(exc).__name__}: {exc}"
        if reason is None:
            return response
        stats["fallbacks"] += 1
        if len(stats["errors"]) < 20:
            stats["errors"].append(reason)
        return random_valid_response(observation)

    wrapped.stats = stats
    return wrapped


def agent(observation: Observation) -> list[int]:
    """Not a standalone entry point: build one with `safe_agent(policy, deck)`."""
    raise NotImplementedError("Use safe_agent(policy, deck); see agent_package/main.py.")
