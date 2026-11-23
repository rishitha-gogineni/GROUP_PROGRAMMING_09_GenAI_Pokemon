"""The plain-Python export must score EXACTLY like the scikit-learn model.

Standard-library unittest. Skipped if scikit-learn is not installed.
"""

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _data(n=400, seed=0):
    rng = random.Random(seed)
    rows = [[rng.random() for _ in range(6)] for _ in range(n)]
    return rows, [r[2] * 10 - r[4] for r in rows]


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class PortableExportTests(unittest.TestCase):
    def test_exported_tree_matches_sklearn(self):
        from team_b.src.models.sadaf.model_1 import Model as DecisionTree
        from team_b.src.models.sadaf.portable import PortableModel, export_json
        rows, y = _data()
        tree = DecisionTree().fit(rows, y)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.json"
            export_json(tree, path)
            portable = PortableModel.load(path)
        test_rows, _ = _data(seed=1)
        self.assertEqual(portable.score_rows(test_rows), tree.score_rows(test_rows))
        state, options = test_rows[0][:3], [r[3:] for r in test_rows[:5]]
        self.assertEqual(portable.score_options(state, options), tree.score_options(state, options))


if __name__ == "__main__":
    unittest.main()
