"""B-SA2: Random Forest Action Scorer (owner: Sadaf).

Same idea as B-SA1 (Decision Tree), but uses many trees and averages
their scores. Uses the SAME rows and targets as B-SA1 so the two models
can be compared fairly (handbook Ch. 5).
"""

import math
from pathlib import Path
from typing import Sequence

import joblib
from sklearn.ensemble import RandomForestRegressor

from ...contracts import FeatureVector

MODEL_ID = "B-SA2"
ARCHITECTURE = "Random Forest regressor"


def make_row(state: FeatureVector, option: FeatureVector) -> list[float]:
    """One training/scoring row = state features followed by option features.
    Must stay identical to make_row() in model_1.py."""
    return list(state) + list(option)


class Model:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 8,
        min_samples_leaf: int = 5,
        seed: int = 0,
    ):
        # n_jobs=1: Kaggle gives only 2 CPUs, so keep inference light.
        self.forest = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=1,
        )
        self.is_trained = False

    def fit(self, rows: Sequence[Sequence[float]], targets: Sequence[float]) -> "Model":
        """Train on rows made with make_row() and their teacher scores."""
        self.forest.fit(rows, targets)
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
        scores = [float(s) for s in self.forest.predict(rows)]
        return [s if math.isfinite(s) else 0.0 for s in scores]

    def describe(self) -> dict:
        """Ensemble size, depth and feature importance (handbook Ch. 5, step 5)."""
        return {
            "n_estimators": self.forest.n_estimators,
            "max_depth": self.forest.max_depth,
            "feature_importances": (
                [round(float(x), 4) for x in self.forest.feature_importances_]
                if self.is_trained else None
            ),
        }

    def save(self, path: Path) -> None:
        joblib.dump(self.forest, path)

    @classmethod
    def load(cls, path: Path) -> "Model":
        m = cls()
        m.forest = joblib.load(path)
        m.is_trained = True
        return m
