"""Reusable direct-SDK individual-battle runner. SDK must be importable as cg."""

import random
import time
import traceback


def check_action(action, selection):
    if not isinstance(action, list) or not all(type(index) is int for index in action):
        raise TypeError("The agent must return a list of integer option indices.")
    if not selection["minCount"] <= len(action) <= selection["maxCount"]:
        raise ValueError("Action length violates selection counts.")
    if len(set(action)) != len(action):
        raise ValueError("Repeated option indices.")
    if any(index < 0 or index >= len(selection["option"]) for index in action):
        raise ValueError("Option index out of range.")


def run_battle(agents, seed, max_steps=10000, reverse_player=False):
    from cg.game import battle_start, battle_select, battle_finish
    from cg.sim import Battle

    random.seed(seed)
    started = time.perf_counter()
    record = {
        "python_seed": seed, "unit": "individual_battle", "status": "running",
        "finished": False, "result_code": None, "steps": 0,
        "reverse_player": reverse_player, "seconds": None,
        "max_decision_ms": 0.0, "selection_types": [],
        "invalid_responses": 0, "engine_rejections": 0,
        "error_type": "", "error_message": "", "traceback": "", "cleanup_error": "",
    }
    seen_types = set()
    try:
        initial = {"select": None, "logs": [], "current": None, "search_begin_input": None}
        decks = [callback(dict(initial)) for callback in agents]
        for deck in decks:
            if not isinstance(deck, list) or len(deck) != 60 or not all(type(card) is int for card in deck):
                raise ValueError("Initial callback did not return 60 integer card IDs.")
        observation, start = battle_start(decks[0], decks[1], reverse_player)
        if observation is None:
            raise RuntimeError(f"Battle start error: player={start.errorPlayer}, type={start.errorType}")
        for step in range(max_steps + 1):
            current = observation.get("current")
            if current is None:
                raise ValueError("Missing current state in direct battle API.")
            result_code = current["result"]
            if result_code != -1:
                record.update(status="passed", finished=True, result_code=result_code)
                break
            if step == max_steps:
                record.update(status="step_limit", error_type="StepLimit", error_message=f"No terminal result after {max_steps} selections.")
                break
            player = current["yourIndex"]
            if player not in (0, 1):
                raise ValueError("Unexpected acting player.")
            selection = observation.get("select")
            if selection is None:
                raise ValueError("Missing selection during the battle.")
            seen_types.add(selection["type"])
            t0 = time.perf_counter()
            action = agents[player](observation)
            record["max_decision_ms"] = max(record["max_decision_ms"], (time.perf_counter() - t0) * 1000)
            try:
                check_action(action, selection)
            except Exception:
                record["invalid_responses"] += 1
                raise
            try:
                observation = battle_select(action)
            except Exception:
                record["engine_rejections"] += 1
                raise
            record["steps"] = step + 1
    except Exception as error:
        record.update(status="error", error_type=type(error).__name__, error_message=str(error), traceback=traceback.format_exc())
    finally:
        try:
            if Battle.battle_ptr:
                battle_finish()
        except Exception as error:
            record["cleanup_error"] = f"{type(error).__name__}: {error}"
            record["status"] = "cleanup_error"
        finally:
            Battle.battle_ptr = None
            record["seconds"] = round(time.perf_counter() - started, 6)
            record["max_decision_ms"] = round(record["max_decision_ms"], 6)
            record["selection_types"] = sorted(seen_types)
    return record
