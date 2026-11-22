"""Kaggle entry point for Sadaf's agents (B-SA1 Decision Tree / B-SA2 Random Forest).

Built by `python3 -m team_b.src.models.sadaf.package --model tree|forest`, which
reuses Chelsi's packager (deck.csv, cg/, teamb/) and adds sadaf_model.json.

Safety: if the model file cannot be loaded, the agent plays Team B's frozen
heuristic-v1 baseline instead; safe_agent() also replaces any illegal move
with a legal one, so the agent can never submit an invalid action.
"""

import os
import sys


def _agent_dirs():
    dirs = []
    if "__file__" in globals():  # the simulator may exec this file without __file__
        dirs.append(os.path.dirname(os.path.abspath(__file__)))
    dirs += [os.getcwd(), "/kaggle_simulations/agent"]
    return [d for d in dirs if os.path.exists(os.path.join(d, "deck.csv"))]


_AGENT_DIR = _agent_dirs()[0]
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

from teamb.adapter import read_deck, safe_agent  # noqa: E402  (created by the packager)
from teamb.baseline import baseline_response  # noqa: E402

try:
    from teamb.models.sadaf.pipeline import make_policy  # noqa: E402
    from teamb.models.sadaf.portable import PortableModel  # noqa: E402
    _inner = make_policy(PortableModel.load(os.path.join(_AGENT_DIR, "sadaf_model.json")))
except Exception:  # never fail to start: fall back to the frozen baseline
    _inner = baseline_response

_POLICY = safe_agent(_inner, read_deck(os.path.join(_AGENT_DIR, "deck.csv")))


def agent(obs_dict: dict) -> list[int]:
    """Entry point; kept as the last function defined in this file."""
    return _POLICY(obs_dict)
