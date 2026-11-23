"""Kavya's trained Pokémon agent."""
import os
from functools import lru_cache
from pathlib import Path
import torch
from team_a.src.baseline import read_deck
from team_a.src.models.kavya.features import encode_state, encode_options
from team_a.src.models.kavya.model_1 import Model as Model1
from team_a.src.models.kavya.model_2 import Model as Model2

@lru_cache(maxsize=2)
def load_model(model_number):
    model = Model1() if model_number == 1 else Model2()

    path = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "kavya"
        / f"model_{model_number}.pt"
    )

    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model

def agent(observation, configuration=None):
    """Return legal choices using a trained MLP."""
    selection = observation.get("select")

    if selection is None:
        return read_deck(configuration)

    options = selection["option"]
    minimum = selection["minCount"]
    maximum = selection["maxCount"]

    if not 0 <= minimum <= maximum <= len(options):
        raise ValueError("Invalid option counts.")

    if maximum == 0:
        return []

    model_number = int(os.environ.get("KAVYA_MODEL", "2"))
    if model_number not in (1, 2):
        raise ValueError("KAVYA_MODEL must be 1 or 2.")

    model = load_model(model_number)

    scores = model.score_options(
        encode_state(observation),
        encode_options(observation),
    )

    ranked = sorted(
        range(len(options)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return ranked[:maximum]
