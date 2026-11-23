"""Checks for Sadaf's pipeline against the REAL game engine.

Standard-library unittest. Skipped if the engine data (team_b/data/raw) or
scikit-learn is missing.
"""

import importlib.util
import unittest
from pathlib import Path

DECK = Path(__file__).resolve().parents[1] / "data" / "raw" / "deck_sample.csv"
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


@unittest.skipUnless(DECK.exists(), "engine data not installed")
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from team_b.src.adapter import read_deck
        from team_b.src.models.sadaf import pipeline as P
        cls.P = P
        cls.deck = read_deck(DECK)
        cls.data = P.collect(12, cls.deck, seed=0)

    def test_rows_line_up_with_targets(self):
        d = self.data
        n = len(d["rows"])
        self.assertGreater(n, 0)
        self.assertTrue(len(d["targets"]) == len(d["games"]) == len(d["decisions"]) == n)
        self.assertEqual(len({len(r) for r in d["rows"]}), 1)  # fixed row length

    def test_split_keeps_whole_games_together(self):
        d = self.data
        tr, va, te = self.P.split_by_game(d["games"])
        games = lambda idx: {d["games"][i] for i in idx}  # noqa: E731
        self.assertFalse(games(tr) & games(va) or games(tr) & games(te) or games(va) & games(te))
        self.assertEqual(len(tr) + len(va) + len(te), len(d["rows"]))

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn not installed")
    def test_trained_tree_plays_only_legal_moves(self):
        from team_b.src import baseline as B
        from team_b.src import simulator
        from team_b.src.adapter import safe_agent
        from team_b.src.models.sadaf.model_1 import Model
        model = Model().fit(self.data["rows"], self.data["targets"])
        agent = safe_agent(self.P.make_policy(model), self.deck)
        rec = simulator.play_game(agent, B.random_response, self.deck, self.deck)
        self.assertIn(rec.winner, (0, 1, 2))              # game finished
        self.assertEqual(agent.stats["fallbacks"], 0)     # never needed the safety net


if __name__ == "__main__":
    unittest.main()
