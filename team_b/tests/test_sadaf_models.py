"""Tests for B-SA1 Decision Tree using PRACTICE data (not real games).

Standard-library unittest. Skipped if scikit-learn is not installed
(training needs it; the Kaggle agent does not, see portable.py).
"""

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def practice_data(n_states=200, n_options=4, seed=0):
    """Fake game: the 'best' move is simply the one with the biggest first number."""
    from team_b.src.models.sadaf.model_1 import make_row
    rng = random.Random(seed)
    rows, targets = [], []
    for _ in range(n_states):
        state = [rng.random(), rng.random()]
        for _ in range(n_options):
            option = [rng.random(), rng.random()]
            rows.append(make_row(state, option))
            targets.append(option[0])  # teacher score
    return rows, targets


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
class DecisionTreeTests(unittest.TestCase):
    def setUp(self):
        from team_b.src.models.sadaf.model_1 import Model
        self.Model = Model

    def test_untrained_model_is_safe(self):
        m = self.Model()
        self.assertEqual(m.score_options([0.1, 0.2], [[1, 0], [0, 1]]), [0.0, 0.0])
        self.assertEqual(m.score_options([0.1, 0.2], []), [])

    def test_tree_learns_to_pick_best_move(self):
        rows, targets = practice_data()
        m = self.Model().fit(rows, targets)
        scores = m.score_options([0.5, 0.5], [[0.1, 0.9], [0.9, 0.1], [0.4, 0.4]])
        self.assertEqual(len(scores), 3)                  # one score per legal move
        self.assertEqual(scores.index(max(scores)), 1)    # picks first number 0.9

    def test_save_and_load(self):
        rows, targets = practice_data()
        m = self.Model().fit(rows, targets)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tree.joblib"
            m.save(path)
            same = self.Model.load(path)
        opts = [[0.2, 0.3], [0.8, 0.1]]
        self.assertEqual(same.score_options([0.5, 0.5], opts), m.score_options([0.5, 0.5], opts))


if __name__ == "__main__":
    unittest.main()
