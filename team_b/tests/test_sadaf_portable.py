"""The plain-Python export must score EXACTLY like the scikit-learn model."""

import random

from team_b.src.models.sadaf.model_1 import Model as DecisionTree
from team_b.src.models.sadaf.portable import PortableModel, export_json


def _data(n=400, seed=0):
    rng = random.Random(seed)
    rows = [[rng.random() for _ in range(6)] for _ in range(n)]
    return rows, [r[2] * 10 - r[4] for r in rows]


def test_exported_tree_matches_sklearn(tmp_path):
    rows, y = _data()
    tree = DecisionTree().fit(rows, y)
    path = tmp_path / "tree.json"
    export_json(tree, path)
    portable = PortableModel.load(path)
    test_rows, _ = _data(seed=1)
    assert portable.score_rows(test_rows) == tree.score_rows(test_rows)
    state, options = test_rows[0][:3], [r[3:] for r in test_rows[:5]]
    assert portable.score_options(state, options) == tree.score_options(state, options)
