"""Plain-Python versions of Sadaf's trained trees (no scikit-learn needed).

Kaggle's agent machine may not have scikit-learn, so for submission the
trained tree / forest is exported to JSON (the split rules and leaf values)
and replayed here with ordinary Python (handbook 3.1: "exported estimator").

    export_json(model, path)          # on your Mac, after training (needs sklearn)
    PortableModel.load(path)          # inside the Kaggle agent (standard library only)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


def _tree_to_dict(tree) -> dict:
    """Copy one fitted sklearn tree's arrays into plain lists."""
    t = tree.tree_
    return {
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": [float(v) for v in t.value[:, 0, 0]],
    }


def export_json(model, path: Path) -> None:
    """Save a trained B-SA1 (model.tree) or B-SA2 (model.forest) as JSON."""
    if hasattr(model, "forest"):
        trees = [_tree_to_dict(est) for est in model.forest.estimators_]
        kind = "random_forest"
    else:
        trees = [_tree_to_dict(model.tree)]
        kind = "decision_tree"
    Path(path).write_text(json.dumps({"kind": kind, "trees": trees}))


class PortableModel:
    """Same score_options() contract as the sklearn models, standard library only."""

    def __init__(self, trees: list[dict]):
        self.trees = trees

    @classmethod
    def load(cls, path: Path) -> "PortableModel":
        return cls(json.loads(Path(path).read_text())["trees"])

    @staticmethod
    def _predict_one(tree: dict, row: Sequence[float]) -> float:
        node = 0
        left, right = tree["left"], tree["right"]
        while left[node] != -1:                       # -1 marks a leaf
            go_left = row[tree["feature"][node]] <= tree["threshold"][node]
            node = left[node] if go_left else right[node]
        return tree["value"][node]

    def score_rows(self, rows: Sequence[Sequence[float]]) -> list[float]:
        scores = []
        for row in rows:
            s = sum(self._predict_one(t, row) for t in self.trees) / len(self.trees)
            scores.append(s if math.isfinite(s) else 0.0)
        return scores

    def score_options(self, state: Sequence[float], options: Sequence[Sequence[float]]) -> list[float]:
        return self.score_rows([list(state) + list(opt) for opt in options])
