from pathlib import Path

from team_a.src.heuristic import read_deck
from team_a.src.models.jiayi.agents import choose_action
from team_a.src.models.jiayi.model_2 import Model


_model = None


def agent(observation, configuration=None):
    global _model
    if observation.get("select") is None:
        return read_deck(configuration)
    if _model is None:
        folder = Path(configuration["__raw_path__"]).resolve().parent
        _model = Model(folder / "model.json")
    return choose_action(_model, observation)
