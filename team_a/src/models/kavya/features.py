
"""State and action feature encoding for Kavya's MLP models."""

from ...contracts import FeatureVector, Observation
from ..rishitha.ar1 import features as base_features

STATE_NAMES = base_features.STATE_NAMES


def encode_state(observation: Observation) -> FeatureVector:
    """Convert the visible game state into 21 numerical features."""
    current = observation.get("current")

    if not isinstance(current, dict):
        raise ValueError("Observation must contain a current game state.")

    players = current.get("players")
    your_index = current.get("yourIndex")

    if (
        not isinstance(players, list)
        or len(players) != 2
        or type(your_index) is not int
        or your_index not in (0, 1)
        or not all(isinstance(player, dict) for player in players)
    ):
        raise ValueError("Invalid players or acting player index.")

    return base_features.state_features(observation)


SELECTION_TYPES = (0, 1, 4, 8, 9)
ACTION_TYPES = tuple(range(15))


def encode_options(observation: Observation) -> list[FeatureVector]:
    """Encode each legal option into a fixed-length numerical vector."""
    selection = observation.get("select")

    if not isinstance(selection, dict):
        raise ValueError("Observation must contain a selection.")

    selection_type = selection.get("type")
    options = selection.get("option")

    if not isinstance(options, list):
        raise ValueError("Selection must contain legal options.")

    if selection_type not in SELECTION_TYPES:
        raise ValueError(f"Unsupported selection type: {selection_type}")

    encoded = []

    for option in options:
        if not isinstance(option, dict):
            raise ValueError("Each option must be a dictionary.")

        selection_vector = [
            float(selection_type == t) for t in SELECTION_TYPES
        ]

        action_type = option.get("type")
        action_vector = [
            float(action_type == t) for t in ACTION_TYPES
        ] + [float(action_type not in ACTION_TYPES)]

        
        def numeric(key, scale):
            value = option.get(key)
            return float(value) / scale if type(value) in (int, float) else 0.0

        numerical_vector = [
            numeric("index", 60),
            numeric("count", 10),
            numeric("number", 60),
            numeric("area", 7),
            numeric("inPlayArea", 7),
            numeric("inPlayIndex", 6),
            numeric("playerIndex", 2),
            numeric("energyIndex", 10),
            numeric("attackId", 2000),
        ]


        encoded.append(
            selection_vector + action_vector + numerical_vector
        )

    return encoded
