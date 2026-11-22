"""Locate and import the official cabt engine wrapper (`cg` package). Owner: Chelsi.

The engine binaries and the `cg` Python wrapper come from the Kaggle data page
(sample_submission/cg). They are competition material and must NOT be committed.

Lookup order:
1. `cg` already importable (true inside the Kaggle submission package).
2. Directory given by the PTCG_CG_DIR environment variable (the folder that
   contains `cg/`, or the `cg/` folder itself).
3. team_b/data/raw/cg (git-ignored local copy).

macOS has no official binary; build one from the engine source, see team_b/README.md.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

TEAM_B_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CG_PARENT = TEAM_B_DIR / "data" / "raw"


def _candidate_parents() -> list[Path]:
    parents = []
    env = os.environ.get("PTCG_CG_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        parents.append(p.parent if p.name == "cg" else p)
    parents.append(DEFAULT_CG_PARENT)
    return parents


def load_cg():
    """Import and return the `cg` package (game + api modules loaded)."""
    try:
        cg = importlib.import_module("cg")
    except ImportError:
        cg = None
        for parent in _candidate_parents():
            if (parent / "cg" / "api.py").exists():
                sys.path.insert(0, str(parent))
                cg = importlib.import_module("cg")
                break
        if cg is None:
            raise ImportError(
                "Cannot find the official `cg` engine package. Copy "
                "sample_submission/cg into team_b/data/raw/cg or set PTCG_CG_DIR."
            )
    importlib.import_module("cg.game")
    importlib.import_module("cg.api")
    return cg


def game():
    """Return the `cg.game` module (battle_start / battle_select / battle_finish)."""
    return load_cg().game


def api():
    """Return the `cg.api` module (enums, all_card_data, all_attack)."""
    return load_cg().api
