"""Team B local game runner. Owner: Chelsi.

Verified behaviour of the local engine (cg wrapper, see team_b/README.md):
- `battle_start(deck0, deck1)` plays ONE game. `current.round` stays 1 and the
  battle ends at the first RESULT log, so a best-of-three match must be built
  here: `run_match` plays games until one side has 2 wins (max 3 games).
- `current.result`: -1 running, 0/1 winning player index, 2 draw.
- Every decision is made by the player `current.yourIndex`; both agents are
  called through the same loop.
- Shuffles/coin flips use the engine's internal RNG, which is NOT seedable from
  Python. Seeds below only control the agents' own randomness, so repeated runs
  are statistically, not exactly, reproducible.

Seat protocol: game 1 agent A is player 0, game 2 player 1, game 3 player 0.
Across matches the starting seat alternates too, so A and B get equal seats.
Who goes first inside a game is decided by the engine's coin flip / IS_FIRST choice.
"""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import engine
from .adapter import checked_response, random_valid_response

Agent = Callable[[dict], list]
RESULT_LOG = 23
MAX_STEPS = 20000


@dataclass
class GameRecord:
    winner: int  # player index 0/1, 2 = draw, -1 = aborted
    reason: int | None
    turns: int
    steps: int
    seconds: float
    invalid: list[int] = field(default_factory=lambda: [0, 0])  # per player index
    decision_ms: list[float] = field(default_factory=lambda: [0.0, 0.0])  # total per player
    decisions: list[int] = field(default_factory=lambda: [0, 0])
    error: str | None = None


def play_game(agent0: Agent, agent1: Agent, deck0: list[int], deck1: list[int],
              max_steps: int = MAX_STEPS, recorder: Callable[[dict, list[int]], None] | None = None) -> GameRecord:
    """Play one game. An illegal response is counted and replaced by a random
    legal one so the game can finish. `recorder(obs, response)` sees every decision."""
    g = engine.game()
    agents = (agent0, agent1)
    t0 = time.perf_counter()
    obs, start = g.battle_start(list(deck0), list(deck1))
    if obs is None:
        return GameRecord(-1, None, 0, 0, 0.0, error=f"battle_start failed: player {start.errorPlayer} type {start.errorType}")
    rec = GameRecord(-1, None, 0, 0, 0.0)
    try:
        while obs["current"] is not None and obs["current"]["result"] == -1:
            if rec.steps >= max_steps:
                rec.error = f"aborted after {max_steps} steps"
                break
            if obs["select"] is None:
                rec.error = "select is None during a running game"
                break
            p = obs["current"]["yourIndex"]
            d0 = time.perf_counter()
            try:
                response, bad = checked_response(obs, agents[p](obs))
            except Exception as exc:
                bad = f"{type(exc).__name__}: {exc}"
            rec.decision_ms[p] += (time.perf_counter() - d0) * 1000.0
            rec.decisions[p] += 1
            if bad is not None:
                rec.invalid[p] += 1
                response = random_valid_response(obs)
            if recorder is not None:
                recorder(obs, response)
            obs = g.battle_select(response)
            rec.steps += 1
        if obs["current"] is not None:
            rec.winner = obs["current"]["result"]
            rec.turns = obs["current"]["turn"]
            for log in obs.get("logs") or []:
                if log.get("type") == RESULT_LOG:
                    rec.reason = log.get("reason")
    finally:
        g.battle_finish()
    rec.seconds = time.perf_counter() - t0
    return rec


def decide_match(game_winners_for_a: list[str], wins_needed: int = 2) -> str:
    """Match outcome for agent A from per-game outcomes ('W', 'L', 'D', 'X' aborted).

    Any aborted game makes the series incomplete -> 'E' (error); error matches are
    excluded from win rate. Otherwise first to `wins_needed` wins, then more wins
    after all games, else draw."""
    if "X" in game_winners_for_a:
        return "E"
    wins = game_winners_for_a.count("W")
    losses = game_winners_for_a.count("L")
    if wins >= wins_needed:
        return "W"
    if losses >= wins_needed:
        return "L"
    if wins != losses:
        return "W" if wins > losses else "L"
    return "D"


def run_match(agent_a: Agent, agent_b: Agent, deck_a: list[int], deck_b: list[int],
              best_of: int = 3, a_starts_as: int = 0, recorder=None) -> dict:
    """Best-of-`best_of` match between A and B with alternating seats."""
    needed = best_of // 2 + 1
    games, outcomes = [], []
    for i in range(best_of):
        a_seat = (a_starts_as + i) % 2
        if a_seat == 0:
            rec = play_game(agent_a, agent_b, deck_a, deck_b, recorder=recorder)
        else:
            rec = play_game(agent_b, agent_a, deck_b, deck_a, recorder=recorder)
        if rec.winner in (0, 1):
            outcome = "W" if rec.winner == a_seat else "L"
        elif rec.winner == 2:
            outcome = "D"
        else:
            outcome = "X"
        outcomes.append(outcome)
        games.append({"a_seat": a_seat, "outcome_for_a": outcome, **asdict(rec)})
        if outcome == "X":  # series is incomplete; stop and report it as an error
            break
        if outcomes.count("W") >= needed or outcomes.count("L") >= needed:
            break
    return {"result_for_a": decide_match(outcomes, needed), "games": games}


def run_series(agent_a: Agent, agent_b: Agent, deck_a: list[int], deck_b: list[int],
               matches: int, seed: int = 0, best_of: int = 3, recorder=None) -> dict:
    """Run `matches` BO3 matches (alternating A's starting seat) and summarise."""
    random.seed(seed)
    results = []
    for m in range(matches):
        results.append(run_match(agent_a, agent_b, deck_a, deck_b, best_of, a_starts_as=m % 2, recorder=recorder))
    return summarise(results)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(match_results: list[dict]) -> dict:
    """Win rate and Wilson interval use only completed matches (W/L/D);
    error matches ('E', an aborted game in the series) are counted separately."""
    n_total = len(match_results)
    wins = sum(r["result_for_a"] == "W" for r in match_results)
    losses = sum(r["result_for_a"] == "L" for r in match_results)
    draws = sum(r["result_for_a"] == "D" for r in match_results)
    errors = n_total - wins - losses - draws
    n = wins + losses + draws
    games = [g for r in match_results for g in r["games"]]
    a_invalid = sum(g["invalid"][g["a_seat"]] for g in games)
    b_invalid = sum(g["invalid"][1 - g["a_seat"]] for g in games)
    a_ms = sum(g["decision_ms"][g["a_seat"]] for g in games)
    a_dec = sum(g["decisions"][g["a_seat"]] for g in games)
    lo, hi = wilson_interval(wins, n)
    return {
        "matches": n,  # completed matches used for win_rate
        "matches_played": n_total,
        "error_matches": errors,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n if n else 0.0,
        "wilson95": [round(lo, 4), round(hi, 4)],
        "games": len(games),
        "game_wins": sum(g["outcome_for_a"] == "W" for g in games),
        "aborted_games": sum(g["outcome_for_a"] == "X" for g in games),
        "invalid_responses_a": a_invalid,
        "invalid_responses_b": b_invalid,
        "mean_decision_ms_a": a_ms / a_dec if a_dec else 0.0,
        "total_seconds": sum(g["seconds"] for g in games),
        "match_results": match_results,
    }


def run_match_from_config(config: dict) -> dict:
    """Config-driven entry: {"agent_a", "agent_b", "deck_a", "deck_b", "matches", "seed"}."""
    from .adapter import read_deck
    from .baseline import POLICIES

    deck_a = read_deck(config["deck_a"])
    deck_b = read_deck(config.get("deck_b", config["deck_a"]))
    return run_series(POLICIES[config["agent_a"]], POLICIES[config["agent_b"]], deck_a, deck_b,
                      matches=int(config.get("matches", 10)), seed=int(config.get("seed", 0)))
