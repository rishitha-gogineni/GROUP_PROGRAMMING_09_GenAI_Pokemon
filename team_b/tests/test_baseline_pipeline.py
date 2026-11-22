"""Team B acceptance checks for the simulator, adapter, card data and baseline.

Run from the repository root (standard library only):

    python3 -m unittest discover -s team_b/tests -t . -v

Tests that need the engine are skipped when team_b/data/raw/cg is missing.
"""

import random
import unittest
from pathlib import Path

from team_b.src import adapter, simulator
from team_b.src.adapter import random_valid_response, top_k, validation_error

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
HAVE_ENGINE = (RAW / "cg" / "api.py").exists() and (RAW / "deck_sample.csv").exists()


def fake_obs(n_options=4, lo=1, hi=1):
    return {"select": {"type": 0, "context": 0, "minCount": lo, "maxCount": hi,
                       "option": [{"type": 14}] * n_options}, "current": {}, "logs": []}


class AdapterTests(unittest.TestCase):
    def test_validation_rules(self):
        obs = fake_obs(4, 1, 2)
        self.assertIsNone(validation_error(obs, [0]))
        self.assertIsNone(validation_error(obs, [3, 1]))
        self.assertIsNotNone(validation_error(obs, []))          # below minCount
        self.assertIsNotNone(validation_error(obs, [0, 1, 2]))   # above maxCount
        self.assertIsNotNone(validation_error(obs, [1, 1]))      # duplicate
        self.assertIsNotNone(validation_error(obs, [4]))         # out of range
        self.assertIsNotNone(validation_error(obs, [-1]))
        self.assertIsNotNone(validation_error(obs, (0,)))        # not a list

    def test_optional_selection_allows_empty(self):
        self.assertIsNone(validation_error(fake_obs(3, 0, 1), []))

    def test_random_response_always_valid(self):
        rng = random.Random(0)
        for n in range(1, 8):
            for hi in range(0, n + 1):
                obs = fake_obs(n, 0, hi)
                self.assertIsNone(validation_error(obs, random_valid_response(obs, rng)))

    def test_top_k_is_deterministic(self):
        self.assertEqual(top_k([1.0, 3.0, 3.0, 0.5], 2), [1, 2])
        self.assertEqual(top_k([1.0], 0), [])

    def test_safe_agent_falls_back(self):
        deck = list(range(60))

        def broken(_obs):
            raise RuntimeError("model failed")

        wrapped = adapter.safe_agent(broken, deck)
        self.assertEqual(wrapped({"select": None}), deck)          # deck phase
        obs = fake_obs(5, 1, 1)
        self.assertIsNone(validation_error(obs, wrapped(obs)))
        self.assertEqual(wrapped.stats["fallbacks"], 1)

        wrapped = adapter.safe_agent(lambda o: [7], deck)            # illegal index
        self.assertIsNone(validation_error(obs, wrapped(obs)))
        self.assertEqual(wrapped.stats["fallbacks"], 1)


class MatchLogicTests(unittest.TestCase):
    def test_best_of_three_decisions(self):
        d = simulator.decide_match
        self.assertEqual(d(["W", "W"]), "W")
        self.assertEqual(d(["L", "W", "L"]), "L")
        self.assertEqual(d(["W", "D", "D"]), "W")
        self.assertEqual(d(["D", "D", "D"]), "D")
        self.assertEqual(d(["W", "L", "D"]), "D")
        self.assertEqual(d(["W", "X", "L"]), "D")

    def test_wilson_interval(self):
        lo, hi = simulator.wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(simulator.wilson_interval(0, 0), (0.0, 0.0))


@unittest.skipUnless(HAVE_ENGINE, "engine not installed in team_b/data/raw")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from team_b.src import baseline, card_features
        cls.baseline = baseline
        cls.cf = card_features
        cls.deck = adapter.read_deck(RAW / "deck_sample.csv")

    def test_sample_deck_is_valid(self):
        self.assertEqual(len(self.deck), 60)
        self.assertEqual(self.cf.deck_problems(self.deck), [])

    def test_deck_checks_catch_problems(self):
        self.assertTrue(self.cf.deck_problems(self.deck[:59]))
        self.assertTrue(self.cf.deck_problems([3] * 60))              # no Basic Pokemon
        self.assertTrue(self.cf.deck_problems([722] * 5 + [3] * 55))  # 5 copies

    def test_card_csv_matches_engine(self):
        csv_path = RAW / "EN_Card_Data_R2_full.csv"
        if not csv_path.exists():
            self.skipTest("card CSV not in team_b/data/raw")
        report = self.cf.validate_card_csv(csv_path)
        self.assertEqual(report["in_csv_not_engine"], [])
        self.assertEqual(report["in_engine_not_csv"], [])

    def test_card_vector_shape(self):
        table = self.cf.load_card_table()
        for cid in set(self.deck):
            self.assertEqual(len(self.cf.card_vector(table.get(cid))), self.cf.CARD_VECTOR_SIZE)
        self.assertEqual(len(self.cf.card_vector(None)), self.cf.CARD_VECTOR_SIZE)

    def test_complete_random_game(self):
        rec = simulator.play_game(self.baseline.random_response, self.baseline.random_response, self.deck, self.deck)
        self.assertIn(rec.winner, (0, 1, 2))
        self.assertEqual(rec.invalid, [0, 0])
        self.assertIsNone(rec.error)

    def test_heuristic_is_always_legal_and_scores_align(self):
        seen_contexts = set()

        def checking_policy(obs):
            scores = self.baseline.teacher_option_scores(obs)
            self.assertEqual(len(scores), len(obs["select"]["option"]))
            seen_contexts.add(obs["select"]["context"])
            response = self.baseline.heuristic_response(obs)
            self.assertIsNone(validation_error(obs, response))
            return response

        for g in range(20):
            rec = simulator.play_game(checking_policy, self.baseline.random_response, self.deck, self.deck)
            self.assertEqual(rec.invalid[0], 0)
            self.assertIsNone(rec.error)
        self.assertIn(0, seen_contexts)  # main phase exercised

    def test_heuristic_beats_random(self):
        summary = simulator.run_series(self.baseline.heuristic_response, self.baseline.random_response,
                                       self.deck, self.deck, matches=30, seed=3)
        self.assertGreaterEqual(summary["win_rate"], 0.8)
        self.assertEqual(summary["invalid_responses_a"], 0)


if __name__ == "__main__":
    unittest.main()
