"""Checks for Sadaf's pipeline against the REAL game engine (needs team_b/data/raw)."""

from pathlib import Path

import pytest

DECK = Path(__file__).resolve().parents[1] / "data" / "raw" / "deck_sample.csv"
pytestmark = pytest.mark.skipif(not DECK.exists(), reason="engine data not installed")


@pytest.fixture(scope="module")
def setup():
    from team_b.src.adapter import read_deck
    from team_b.src.models.sadaf import pipeline as P
    deck = read_deck(DECK)
    return P, deck, P.collect(12, deck, seed=0)


def test_rows_line_up_with_targets(setup):
    P, _, data = setup
    n = len(data["rows"])
    assert n > 0
    assert len(data["targets"]) == len(data["games"]) == len(data["decisions"]) == n
    assert len({len(r) for r in data["rows"]}) == 1          # fixed row length


def test_split_keeps_whole_games_together(setup):
    P, _, data = setup
    tr, va, te = P.split_by_game(data["games"])
    games = lambda idx: {data["games"][i] for i in idx}       # noqa: E731
    assert not (games(tr) & games(va) or games(tr) & games(te) or games(va) & games(te))
    assert len(tr) + len(va) + len(te) == len(data["rows"])


def test_trained_tree_plays_only_legal_moves(setup):
    P, deck, data = setup
    from team_b.src import baseline as B
    from team_b.src import simulator
    from team_b.src.adapter import safe_agent
    from team_b.src.models.sadaf.model_1 import Model
    model = Model().fit(data["rows"], data["targets"])
    agent = safe_agent(P.make_policy(model), deck)
    rec = simulator.play_game(agent, B.random_response, deck, deck)
    assert rec.winner in (0, 1, 2)                           # game finished
    assert agent.stats["fallbacks"] == 0                     # never needed the safety net
