"""B-SA1: Decision Tree Action Scorer (owner: Sadaf).

Idea: for each legal move, join the game-state numbers with that move's
numbers into one row. The tree predicts a score for the row. The agent
then plays the move with the highest score.
"""

import math
from pathlib import Path
from typing import Sequence

import joblib
from sklearn.tree import DecisionTreeRegressor

from ...contracts import FeatureVector

MODEL_ID = "B-SA1"
ARCHITECTURE = "Decision Tree regressor"


def make_row(state: FeatureVector, option: FeatureVector) -> list[float]:
    """One training/scoring row = state features followed by option features."""
    return list(state) + list(option)


class Model:
    def __init__(self, max_depth: int = 8, min_samples_leaf: int = 5, seed: int = 0):
        self.tree = DecisionTreeRegressor(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed
        )
        self.is_trained = False

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "Model":
        """Train on rows made with make_row() and their teacher scores."""
        self.tree.fit(rows, targets)
        self.is_trained = True
        return self

    def score_options(
        self, state: FeatureVector, options: Sequence[FeatureVector]
    ) -> list[float]:
        """Return one finite score per legal option, in the same order."""
        if not options:
            return []
        if not self.is_trained:
            return [0.0] * len(options)  # safe fallback: all moves equal
        rows = [make_row(state, opt) for opt in options]
        scores = [float(s) for s in self.tree.predict(rows)]
        return [s if math.isfinite(s) else 0.0 for s in scores]

    def save(self, path: Path) -> None:
        joblib.dump(self.tree, path)

    @classmethod
    def load(cls, path: Path) -> "Model":
        m = cls()
        m.tree = joblib.load(path)
        m.is_trained = True
        return m