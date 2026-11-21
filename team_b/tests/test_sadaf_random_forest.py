"""Tests for B-SA2 Random Forest using PRACTICE data (not real games)."""

import random

from team_b.src.models.sadaf.model_2 import Model as RandomForest, make_row


def practice_data(n_states=200, n_options=4, seed=0):
    """Same fake game as the Decision Tree tests: best move = biggest first number."""
    rng = random.Random(seed)
    rows, targets = [], []
    for _ in range(n_states):
        state = [rng.random(), rng.random()]
        for _ in range(n_options):
            option = [rng.random(), rng.random()]
            rows.append(make_row(state, option))
            targets.append(option[0])          # teacher score
    return rows, targets


def test_untrained_model_is_safe():
    m = RandomForest()
    assert m.score_options([0.1, 0.2], [[1, 0], [0, 1]]) == [0.0, 0.0]
    assert m.score_options([0.1, 0.2], []) == []
    assert m.describe()["feature_importances"] is None


def test_forest_learns_to_pick_best_move():
    rows, targets = practice_data()
    m = RandomForest(n_estimators=20).fit(rows, targets)
    scores = m.score_options([0.5, 0.5], [[0.1, 0.9], [0.9, 0.1], [0.4, 0.4]])
    assert len(scores) == 3                    # one score per legal move
    assert scores.index(max(scores)) == 1      # picks the move with first number 0.9


def test_feature_importance_finds_the_right_number():
    rows, targets = practice_data()
    m = RandomForest(n_estimators=20).fit(rows, targets)
    imp = m.describe()["feature_importances"]
    # row = [state0, state1, option0, option1]; only option0 matters in this fake game
    assert imp.index(max(imp)) == 2


def test_save_and_load(tmp_path):
    rows, targets = practice_data()
    m = RandomForest(n_estimators=20).fit(rows, targets)
    path = tmp_path / "forest.joblib"
    m.save(path)
    same = RandomForest.load(path)
    opts = [[0.2, 0.3], [0.8, 0.1]]
    assert same.score_options([0.5, 0.5], opts) == m.score_options([0.5, 0.5], opts)