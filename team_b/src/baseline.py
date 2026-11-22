"""Team B baseline policies. Owner: Chelsi.

- `random_response`: uniformly random legal selection (reference opponent).
- `first_response`: always the first legal option(s) (deterministic sanity opponent).
- `heuristic_response`: rule-based scorer (baseline_v1). Every option gets a
  hand-written score from public, acting-player-visible information; the
  selection count is chosen per decision context, then the top-k are returned.
- `teacher_option_scores`: the same heuristic scores, exposed as the documented
  regression target for the B-C1/B-C2 card scorers. They are teacher preferences,
  NOT true action values.

Enum values below mirror cg/api.py (OptionType, SelectType, SelectContext, AreaType).
"""

from __future__ import annotations

import random

from .adapter import random_valid_response, top_k
from .card_features import (
    BASIC_ENERGY,
    ITEM,
    POKEMON,
    SPECIAL_ENERGY,
    STADIUM,
    SUPPORTER,
    TOOL,
    CardInfo,
    load_card_table,
)
from .contracts import Observation

BASELINE_VERSION = "heuristic-v1"

# OptionType
NUMBER, YES, NO, CARD, TOOL_CARD, ENERGY_CARD, ENERGY = 0, 1, 2, 3, 4, 5, 6
PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END, SKILL, SPECIAL_CONDITION = range(7, 17)
# AreaType
A_DECK, A_HAND, A_DISCARD, A_ACTIVE, A_BENCH, A_PRIZE, A_STADIUM = 1, 2, 3, 4, 5, 6, 7
A_LOOKING = 12
# SelectContext groups
CTX_MAIN = 0
CTX_SETUP_ACTIVE, CTX_SETUP_BENCH, CTX_SWITCH, CTX_TO_ACTIVE, CTX_TO_BENCH, CTX_TO_FIELD, CTX_TO_HAND = 1, 2, 3, 4, 5, 6, 7
CTX_DISCARD, CTX_TO_DECK, CTX_TO_DECK_BOTTOM, CTX_TO_PRIZE = 8, 9, 10, 11
CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY, CTX_DAMAGE, CTX_REMOVE_DAMAGE_COUNTER, CTX_HEAL = 13, 14, 15, 16, 17
CTX_ATTACH_FROM, CTX_ATTACH_TO = 21, 22
CTX_DRAW_COUNT, CTX_DAMAGE_COUNTER_COUNT, CTX_REMOVE_DAMAGE_COUNTER_COUNT = 38, 39, 40
CTX_IS_FIRST, CTX_MULLIGAN = 41, 42
# Contexts where taking fewer selections is better (losing resources / hurting own side).
MINIMISE_CONTEXTS = {8, 9, 10, 11, 20, 23, 26, 27, 29, 30, 31, 32, 36}
# Contexts where a selection hurts the *selected* card's owner (target the opponent).
HARMFUL_TARGET_CONTEXTS = {CTX_DAMAGE_COUNTER, CTX_DAMAGE_COUNTER_ANY, CTX_DAMAGE, 20, 25, 36}


# --------------------------------------------------------------------------- state helpers

def _me(obs: Observation) -> int:
    return obs["current"]["yourIndex"]


def _player(obs: Observation, idx: int) -> dict:
    return obs["current"]["players"][idx]


def _active(player: dict) -> dict | None:
    active = player.get("active") or []
    return active[0] if active else None


def _card_at(obs: Observation, area: int | None, index: int | None, player_idx: int | None) -> dict | None:
    """Resolve an (area, index, player) reference to the visible card/pokemon dict."""
    if area is None or index is None:
        return None
    current = obs["current"]
    p = _player(obs, player_idx if player_idx is not None else _me(obs))
    try:
        if area == A_HAND:
            return (p.get("hand") or [])[index]
        if area == A_ACTIVE:
            return (p.get("active") or [])[index]
        if area == A_BENCH:
            return p["bench"][index]
        if area == A_DISCARD:
            return p["discard"][index]
        if area == A_PRIZE:
            return p["prize"][index]
        if area == A_DECK:
            return (obs["select"].get("deck") or [])[index]
        if area == A_LOOKING:
            return (current.get("looking") or [])[index]
        if area == A_STADIUM:
            return current["stadium"][index]
    except (IndexError, KeyError, TypeError):
        return None
    return None


def _info(card: dict | None) -> CardInfo | None:
    return load_card_table().get(card.get("id")) if card else None


def _energy_need(pokemon: dict | None) -> int:
    """How many more energies this Pokemon needs for its most expensive attack."""
    info = _info(pokemon)
    if not info or not info.attacks:
        return 0
    return max(0, info.max_attack_cost - len(pokemon.get("energies") or []))


def _attack_damage(obs: Observation, attack_id: int | None) -> tuple[int, bool]:
    """(expected base damage incl. weakness/resistance, knocks out opponent active?)"""
    atk = load_card_table().attack(attack_id)
    if atk is None:
        return 0, False
    damage = atk.damage
    me = _me(obs)
    my_active = _info(_active(_player(obs, me)))
    target = _active(_player(obs, 1 - me))
    target_info = _info(target)
    if target_info and my_active:
        if target_info.weakness is not None and target_info.weakness == my_active.energy_type:
            damage *= 2
        if target_info.resistance is not None and target_info.resistance == my_active.energy_type:
            damage = max(0, damage - 30)
    knock_out = bool(target) and damage >= target.get("hp", 10**9)
    return damage, knock_out


def _pokemon_strength(pokemon: dict | None) -> float:
    """Crude value of an in-play Pokemon: remaining HP plus attached energy."""
    if not pokemon:
        return 0.0
    return pokemon.get("hp", 0) / 10.0 + 8.0 * len(pokemon.get("energies") or [])


# --------------------------------------------------------------------------- option scoring

def _score_main_option(obs: Observation, opt: dict) -> float:
    """Turn-order priorities: develop (evolve, energy, bench, draw) before attacking;
    attack when it knocks out; ending the turn is the last resort."""
    t = opt["type"]
    me = _me(obs)
    mine = _player(obs, me)
    current = obs["current"]
    stalled = current.get("turnActionCount", 0) > 40  # avoid long action loops

    if t == ATTACK:
        damage, knock_out = _attack_damage(obs, opt.get("attackId"))
        return 95.0 + damage / 100.0 if knock_out else 40.0 + damage / 10.0
    if t == END:
        return 0.0
    if stalled:
        return 1.0
    if t == EVOLVE:
        return 85.0
    if t == ATTACH:
        card = _info(_card_at(obs, opt.get("area"), opt.get("index"), me))
        target = _card_at(obs, opt.get("inPlayArea"), opt.get("inPlayIndex"), me)
        if card and card.card_type in (BASIC_ENERGY, SPECIAL_ENERGY):
            need = _energy_need(target)
            if need > 0:
                return 75.0 if opt.get("inPlayArea") == A_ACTIVE else 70.0 - 0.1 * (opt.get("inPlayIndex") or 0)
            return 20.0
        return 50.0  # tools and other attachables
    if t == PLAY:
        card = _info(_card_at(obs, A_HAND, opt.get("index"), me))
        if card is None:
            return 30.0
        if card.card_type == POKEMON:
            bench_space = mine["benchMax"] - len(mine["bench"])
            if bench_space <= 0:
                return 5.0
            # Most local losses are "no Pokemon left to promote": an empty bench is urgent.
            return (90.0 if not mine["bench"] else 65.0) + card.hp / 100.0
        if card.card_type == SUPPORTER:
            return 60.0
        if card.card_type == ITEM:
            return 55.0
        if card.card_type == TOOL:
            return 50.0
        if card.card_type == STADIUM:
            return 35.0
        return 30.0
    if t == ABILITY:
        return 58.0
    if t == RETREAT:
        active = _active(mine)
        best_bench = max((_pokemon_strength(p) for p in mine["bench"]), default=0.0)
        if active and active.get("hp", 0) <= 30 and best_bench > _pokemon_strength(active):
            return 45.0
        return 2.0
    if t == DISCARD:
        return 3.0
    return 10.0


def _score_card_option(obs: Observation, opt: dict, context: int) -> float:
    me = _me(obs)
    owner = opt.get("playerIndex", me)
    card = _card_at(obs, opt.get("area"), opt.get("index"), owner)
    info = _info(card)
    in_play = opt.get("area") in (A_ACTIVE, A_BENCH)

    if context in HARMFUL_TARGET_CONTEXTS:
        # Prefer the opponent's Pokemon with the least HP left (closest to a knock-out).
        hp = card.get("hp", 999) if (card and in_play) else 999
        bonus = (info.prize_value * 10) if info else 0
        return (100.0 if owner != me else 0.0) - hp / 10.0 + bonus
    if context in (CTX_HEAL, CTX_REMOVE_DAMAGE_COUNTER):
        if card and in_play:
            return (100.0 if owner == me else 0.0) + (card.get("maxHp", 0) - card.get("hp", 0)) / 10.0
        return 0.0
    if context in (CTX_SWITCH, CTX_TO_ACTIVE):
        if card and in_play:
            ready = 20.0 if _energy_need(card) == 0 and _info(card) and _info(card).attacks else 0.0
            return _pokemon_strength(card) + ready
        return (info.hp / 10.0) if info else 0.0
    if context in MINIMISE_CONTEXTS:
        # Losing a card: give up the least useful one (energy surplus, then trainers).
        if info is None:
            return 0.0
        if info.card_type == BASIC_ENERGY:
            return 30.0
        if info.card_type == POKEMON:
            return 5.0 + info.hp / 100.0
        return 15.0
    if context in (CTX_ATTACH_FROM, CTX_ATTACH_TO):
        if card and in_play and owner == me:
            return 50.0 + 10.0 * _energy_need(card) + (5.0 if opt.get("area") == A_ACTIVE else 0.0)
        return 10.0 + (info.hp / 100.0 if info else 0.0)
    # Gaining cards / putting Pokemon into play (setup, to hand, to bench, search ...).
    if info is None:
        return 1.0
    if info.card_type == POKEMON:
        base = 40.0 if info.basic else 35.0
        if context == CTX_SETUP_ACTIVE:
            # Active: sturdy and cheap to attack with.
            return info.hp / 10.0 + info.max_damage / 20.0 - 3.0 * info.min_attack_cost
        return base + info.hp / 100.0
    if info.card_type == SUPPORTER:
        return 30.0
    if info.card_type in (ITEM, TOOL):
        return 25.0
    if info.card_type in (BASIC_ENERGY, SPECIAL_ENERGY):
        return 20.0
    return 10.0


def teacher_option_scores(observation: Observation) -> list[float]:
    """Heuristic score per current option (index-aligned with select['option'])."""
    select = observation["select"]
    context = select["context"]
    options = select["option"]
    scores = []
    for opt in options:
        t = opt["type"]
        if context == CTX_MAIN:
            s = _score_main_option(observation, opt)
        elif t in (CARD, TOOL_CARD, ENERGY_CARD):
            s = _score_card_option(observation, opt, context)
        elif t in (YES, NO):
            # Go first, mulligan when allowed, activate effects, pick heads: say yes.
            s = 1.0 if t == YES else 0.0
        elif t == NUMBER:
            n = opt.get("number") or 0
            s = -float(n) if context in MINIMISE_CONTEXTS else float(n)
        elif t == ENERGY:
            s = -float(opt.get("count") or 0)
        else:
            s = 0.0
        scores.append(float(s))
    return scores


def _selection_count(observation: Observation) -> int:
    """How many options to pick: take as few as allowed when selecting is a cost,
    otherwise as many as allowed (more cards to hand, more Pokemon benched)."""
    select = observation["select"]
    lo, hi = select["minCount"], select["maxCount"]
    context = select["context"]
    if context in MINIMISE_CONTEXTS:
        return lo
    if context in (CTX_HEAL, CTX_REMOVE_DAMAGE_COUNTER):
        return max(lo, min(hi, 1))
    if select["type"] == 4:  # SelectType.ENERGY: cover the remaining cost exactly when possible
        return max(lo, min(hi, select.get("remainEnergyCost") or lo))
    return hi


# --------------------------------------------------------------------------- policies

def heuristic_response(observation: Observation) -> list[int]:
    """baseline_v1: top-k options by heuristic score; always a legal selection."""
    scores = teacher_option_scores(observation)
    k = _selection_count(observation)
    return top_k(scores, k)


def random_response(observation: Observation, rng: random.Random | None = None) -> list[int]:
    return random_valid_response(observation, rng)


def first_response(observation: Observation) -> list[int]:
    select = observation["select"]
    return list(range(select["maxCount"]))


def baseline_response(observation: Observation) -> list[int]:
    """Frozen Team B baseline (heuristic-v1)."""
    return heuristic_response(observation)


POLICIES = {
    "heuristic": heuristic_response,
    "random": random_response,
    "first": first_response,
}
