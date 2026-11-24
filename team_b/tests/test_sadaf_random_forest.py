"""Tests for B-SA2 Random Forest using PRACTICE data (not real games).

Standard-library unittest. Skipped if scikit-learn is not installed.
"""

import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def practice_data(n_states=200, n_options=4, seed=0):
    """Same fake game as the Decision Tree tests: best move = biggest first number."""
    from team_b.src.models.sadaf.model_2 import make_row
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
class RandomForestTests(unittest.TestCase):
    def setUp(self):
        from team_b.src.models.sadaf.model_2 import Model
        self.Model = Model

    def test_untrained_model_is_safe(self):
        m = self.Model()
        self.assertEqual(m.score_options([0.1, 0.2], [[1, 0], [0, 1]]), [0.0, 0.0])
        self.assertEqual(m.score_options([0.1, 0.2], []), [])
        self.assertIsNone(m.describe()["feature_importances"])

    def test_forest_learns_to_pick_best_move(self):
        rows, targets = practice_data()
        m = self.Model(n_estimators=20).fit(rows, targets)
        scores = m.score_options([0.5, 0.5], [[0.1, 0.9], [0.9, 0.1], [0.4, 0.4]])
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores.index(max(scores)), 1)

    def test_feature_importance_finds_the_right_number(self):
        rows, targets = practice_data()
        m = self.Model(n_estimators=20).fit(rows, targets)
        imp = m.describe()["feature_importances"]
        # row = [state0, state1, option0, option1]; only option0 matters here
        self.assertEqual(imp.index(max(imp)), 2)

    def test_save_and_load(self):
        rows, targets = practice_data()
        m = self.Model(n_estimators=20).fit(rows, targets)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forest.joblib"
            m.save(path)
            same = self.Model.load(path)
        opts = [[0.2, 0.3], [0.8, 0.1]]
        self.assertEqual(same.score_options([0.5, 0.5], opts), m.score_options([0.5, 0.5], opts))


if __name__ == "__main__":
    unittest.main()
