"""Team B Kaggle submission entry point (GROUP_PROGRAMMING_09_b).

Built into a submission by `python3 -m team_b.scripts.package_agent`, which puts
next to this file: deck.csv, the official `cg/` engine package, and `teamb/`
(a copy of team_b/src). Do not upload this folder directly.

Policy: Team B baseline heuristic-v1 with a random-legal fallback.
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

from teamb.adapter import read_deck, safe_agent  # noqa: E402  (created by package_agent)
from teamb.baseline import baseline_response  # noqa: E402

_POLICY = safe_agent(baseline_response, read_deck(os.path.join(_AGENT_DIR, "deck.csv")))


def agent(obs_dict: dict) -> list[int]:
    """Entry point; kept as the last function defined in this file."""
    return _POLICY(obs_dict)
