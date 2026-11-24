import math
import random

from ... import features, heuristic
from ...heuristic_runner import run_battle


SHAPING_SCALE = 0.25


def terminal_reward(result_code, seat):
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError("Expected player seat 0 or 1")
    if type(result_code) is not int or result_code not in (0, 1, 2):
        raise ValueError("Expected a terminal winner seat or draw code 2")
    return 0 if result_code == 2 else 1 if result_code == seat else -1


def select_action(model, observation, rng, epsilon):
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if not features.supports_observation(observation):
        return heuristic.agent(observation), {"source": "heuristic", "probability": None,
                                              "log_probability": None, "learned_probability": None}
    encoded = features.encode_observation(observation)
    indices = encoded["option_indices"]
    learned = model.policy_probabilities(encoded["state"], encoded["options"])
    # Sample the mixture and log its marginal probability, even for uniform draws.
    mixed = [(1 - epsilon) * p + epsilon / len(indices) for p in learned]
    index = rng.choices(range(len(indices)), weights=mixed, k=1)[0]
    probability = mixed[index] / sum(mixed)
    return [indices[index]], {"source": "learned", "probability": probability,
                              "log_probability": math.log(probability), "learned_probability": learned[index]}


def prize_lead(observation):
    players = observation["current"]["players"]
    if len(players) != 2 or any(not isinstance(player.get("prize"), list) or
                                len(player["prize"]) > 6 for player in players):
        raise ValueError("Expected visible remaining prizes for both players")
    # Player entries are indexed by seat. Fewer prizes remaining means a lead.
    return (len(players[1]["prize"]) - len(players[0]["prize"])) / 6


def label_returns(decisions, result_code):
    if not decisions:
        raise ValueError("A completed rollout must contain accepted decisions")
    terminal = [terminal_reward(result_code, seat) for seat in (0, 1)]
    totals = [0.0, 0.0]
    next_potential = 0.0
    for index in reversed(range(len(decisions))):
        decision = decisions[index]
        potential = SHAPING_SCALE * decision["prize_lead_seat0"]
        delta = next_potential - potential
        base = terminal if index == len(decisions) - 1 else [0, 0]
        shaped = [base[0] + delta, base[1] - delta]
        totals = [totals[seat] + shaped[seat] for seat in (0, 1)]
        seat = decision["seat"]
        decision.update(terminal_reward=base[seat], shaped_reward=shaped[seat],
                        terminal_return=terminal[seat], shaped_return=totals[seat])
        next_potential = potential
    return terminal, totals


def play_battle(models, deck, seed, epsilon=0.1, reverse_player=False):
    rng = random.Random(seed)
    decisions = []
    pending = None

    def callback(model):
        def act(observation):
            nonlocal pending
            if observation.get("select") is None:
                return list(deck)
            action, metadata = select_action(model, observation, rng, epsilon)
            pending = metadata
            return action
        return act

    def observe(observation, action, step, seat):
        decisions.append(dict(pending, step=step, seat=seat, action=action,
                              legal_option_count=len(observation["select"]["option"]),
                              prize_lead_seat0=prize_lead(observation)))

    result = run_battle([callback(model) for model in models], seed=seed,
                        reverse_player=reverse_player, decision_observer=observe)
    if result["status"] != "passed" or not result["finished"]:
        return result, decisions
    terminal, shaped = label_returns(decisions, result["result_code"])
    result.update(terminal_returns=terminal, shaped_returns=shaped)
    return result, decisions
