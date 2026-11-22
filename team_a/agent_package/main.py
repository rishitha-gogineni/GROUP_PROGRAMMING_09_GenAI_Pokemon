"""Team A random baseline. Uses the official sample deck and maxCount policy."""

import os
import random


def read_deck(configuration=None):
    candidates = []
    raw_path = (configuration or {}).get("__raw_path__")
    if raw_path:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(raw_path)), "deck.csv"))
    module_path = globals().get("__file__")
    if module_path:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(module_path)), "deck.csv"))
    candidates.extend(["/kaggle_simulations/agent/deck.csv", "deck.csv"])
    for path in candidates:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as handle:
                deck = [int(line.strip()) for line in handle if line.strip()]
            if len(deck) != 60:
                raise ValueError("deck.csv must contain exactly 60 card IDs.")
            return deck
    raise FileNotFoundError("deck.csv was not found beside the agent or in the Kaggle agent folder.")


def agent(observation, configuration=None):
    selection = observation.get("select")
    if selection is None:
        return read_deck(configuration)
    options = selection["option"]
    minimum = selection["minCount"]
    maximum = selection["maxCount"]
    if not (0 <= minimum <= maximum <= len(options)):
        raise ValueError("Invalid option counts in the observation.")
    return random.sample(range(len(options)), maximum)
