"""Sadaf's pipeline: game observation -> feature rows -> scores -> legal move.

Shared by B-SA1 (Decision Tree) and B-SA2 (Random Forest) so both models use
the SAME features, targets and data split (handbook Ch. 5, Sadaf step 4-6).

- Features use only information the acting player can see (handbook 7.4).
- Target = Chelsi's teacher_option_scores (heuristic-v1 preferences, NOT true values).
- Data is split by complete game, never by individual decisions (handbook 7.3/7.4).
"""

from __future__ import annotations

import random

from team_b.src import baseline as B
from team_b.src import simulator
from team_b.src.adapter import top_k
from team_b.src.card_features import card_vector

SCHEMA_VERSION = "sadaf-v1"
N_OPTION_TYPES = 17   # cg OptionType 0..16
N_CONTEXTS = 43       # cg SelectContext 0..42


# ----------------------------------------------------------------- features

def _side(player: dict) -> list[float]:
    """Public numbers about one player's board."""
    active = B._active(player)
    return [
        (active or {}).get("hp", 0) / 100.0,
        (active or {}).get("maxHp", 0) / 100.0,
        len((active or {}).get("energies") or []) / 4.0,
        len(player.get("bench") or []) / 5.0,
        sum(p.get("hp", 0) for p in player.get("bench") or []) / 500.0,
        len(player.get("prize") or []) / 6.0,
        player.get("handCount", 0) / 10.0,
        player.get("deckCount", 0) / 60.0,
        float(any(player.get(s) for s in ("poisoned", "burned", "asleep", "paralyzed", "confused"))),
    ]


def encode_state(obs: dict) -> list[float]:
    """Acting player's view of the game (no hidden opponent hand)."""
    cur = obs["current"]
    me = cur["yourIndex"]
    sel = obs["select"]
    ctx = [0.0] * N_CONTEXTS
    if 0 <= sel["context"] < N_CONTEXTS:
        ctx[sel["context"]] = 1.0
    return ctx + [
        min(cur.get("turn", 0), 30) / 30.0,
        min(cur.get("turnActionCount", 0), 50) / 50.0,
        float(cur.get("energyAttached", False)),
        float(cur.get("supporterPlayed", False)),
        float(cur.get("retreated", False)),
        float(me == cur.get("firstPlayer")),
        len(sel["option"]) / 20.0,
    ] + _side(cur["players"][me]) + _side(cur["players"][1 - me])


def encode_option(obs: dict, opt: dict) -> list[float]:
    """Numbers describing one legal option."""
    me = obs["current"]["yourIndex"]
    t = opt.get("type", -1)
    onehot = [1.0 if t == i else 0.0 for i in range(N_OPTION_TYPES)]
    # The card this option refers to (hand card to play, target to pick, ...).
    owner = opt.get("playerIndex", me)
    card = B._card_at(obs, opt.get("area"), opt.get("index"), owner)
    target = B._card_at(obs, opt.get("inPlayArea"), opt.get("inPlayIndex"), me)
    damage, knock_out = B._attack_damage(obs, opt.get("attackId")) if t == B.ATTACK else (0, False)
    return onehot + card_vector(B._info(card)) + [
        float(owner != me),
        float(opt.get("area") == B.A_ACTIVE),
        float(opt.get("area") == B.A_BENCH),
        float(opt.get("inPlayArea") == B.A_ACTIVE),
        B._energy_need(target) / 4.0,
        (card or {}).get("hp", 0) / 100.0,
        damage / 100.0,
        float(knock_out),
        (opt.get("number") or 0) / 10.0,
        (opt.get("count") or 0) / 10.0,
    ]


def option_rows(obs: dict) -> list[list[float]]:
    """One row per legal option = state features + option features."""
    state = encode_state(obs)
    return [state + encode_option(obs, opt) for opt in obs["select"]["option"]]


# ----------------------------------------------------------------- data

def collect(n_games: int, deck: list[int], seed: int = 0) -> dict:
    """Play heuristic-vs-random games and record every decision of BOTH players.

    Returns {"rows", "targets", "games", "decisions"}: one entry per legal option.
    targets = teacher scores; games/decisions = which game/decision each row came from.
    """
    rng = random.Random(seed)
    data = {"rows": [], "targets": [], "games": [], "decisions": []}
    state = {"game": 0, "decision": 0}

    def recorder(obs, _response):
        r = option_rows(obs)
        data["rows"].extend(r)
        data["targets"].extend(B.teacher_option_scores(obs))
        data["games"].extend([state["game"]] * len(r))
        data["decisions"].extend([state["decision"]] * len(r))
        state["decision"] += 1

    rand = lambda obs: B.random_response(obs, rng)  # noqa: E731
    for g in range(n_games):
        state["game"] = g
        a, b = (B.heuristic_response, rand) if g % 2 == 0 else (rand, B.heuristic_response)
        simulator.play_game(a, b, deck, deck, recorder=recorder)
    return data


def split_by_game(groups: list[int], seed: int = 0, val: float = 0.15, test: float = 0.15):
    """Row indices for train/val/test, keeping each whole game in one split."""
    games = sorted(set(groups))
    if len(games) < 3:
        raise ValueError("need at least 3 games to make train/val/test splits")
    random.Random(seed).shuffle(games)
    n_val, n_test = int(len(games) * val), int(len(games) * test)
    val_g, test_g = set(games[:n_val]), set(games[n_val:n_val + n_test])
    tr, va, te = [], [], []
    for i, g in enumerate(groups):
        (va if g in val_g else te if g in test_g else tr).append(i)
    return tr, va, te


# ----------------------------------------------------------------- agent

def top1_agreement(predicted: list[float], data: dict, idx: list[int]) -> float:
    """Share of decisions where the model's best option = the teacher's best option.
    Diagnostic only: agreeing with the teacher is not the same as winning (handbook 9.3)."""
    by_decision: dict[int, list[int]] = {}
    for pos, i in enumerate(idx):
        by_decision.setdefault(data["decisions"][i], []).append(pos)
    hits = 0
    for positions in by_decision.values():
        best_model = max(positions, key=lambda p: predicted[p])
        best_teacher = max(positions, key=lambda p: data["targets"][idx[p]])
        hits += data["targets"][idx[best_model]] == data["targets"][idx[best_teacher]]
    return hits / len(by_decision) if by_decision else 0.0


def make_policy(model):
    """Turn a trained scorer into a game policy: obs -> legal option indices.

    Uses the team contract model.score_options(state, options). The model ranks
    the options; how MANY to pick uses heuristic-v1's rule (B._selection_count),
    so only the ranking is learned.
    """
    def policy(obs: dict) -> list[int]:
        state = encode_state(obs)
        options = [encode_option(obs, opt) for opt in obs["select"]["option"]]
        return top_k(model.score_options(state, options), B._selection_count(obs))
    return policy
