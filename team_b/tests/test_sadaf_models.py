"""Tests for Sadaf's models using PRACTICE data (not real games)."""

import random

from team_b.src.models.sadaf.model_1 import Model as DecisionTree, make_row


def practice_data(n_states=200, n_options=4, seed=0):
    """Fake data: the 'best' move is simply the one with the biggest first number."""
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
    m = DecisionTree()
    assert m.score_options([0.1, 0.2], [[1, 0], [0, 1]]) == [0.0, 0.0]
    assert m.score_options([0.1, 0.2], []) == []


def test_tree_learns_to_pick_best_move():
    rows, targets = practice_data()
    m = DecisionTree().fit(rows, targets)
    scores = m.score_options([0.5, 0.5], [[0.1, 0.9], [0.9, 0.1], [0.4, 0.4]])
    assert len(scores) == 3                    # one score per legal move
    assert scores.index(max(scores)) == 1      # picks the move with first number 0.9


def test_save_and_load(tmp_path):
    rows, targets = practice_data()
    m = DecisionTree().fit(rows, targets)
    path = tmp_path / "tree.joblib"
    m.save(path)
    same = DecisionTree.load(path)
    opts = [[0.2, 0.3], [0.8, 0.1]]
    assert same.score_options([0.5, 0.5], opts) == m.score_options([0.5, 0.5], opts)