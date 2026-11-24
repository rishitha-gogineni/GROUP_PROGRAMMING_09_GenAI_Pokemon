import math
import json
from pathlib import Path
import random
import unittest

from .model_1 import Model as Linear
from .model_2 import Model as MLP
from .self_play import label_returns, prize_lead, select_action, terminal_reward


class SelfPlayTests(unittest.TestCase):
    def test_terminal_perspective(self):
        for result, expected in [(0, [1, -1]), (1, [-1, 1]), (2, [0, 0])]:
            self.assertEqual([terminal_reward(result, seat) for seat in (0, 1)], expected)
        for result, seat in [(-1, 0), (None, 1), (3, 0), (1, 2), (True, 0)]:
            with self.assertRaises(ValueError):
                terminal_reward(result, seat)

    def test_behavior_probability_includes_exploration(self):
        class Policy:
            def policy_probabilities(self, state, options):
                return [0.0, 1.0]

        observation = {"current": {"players": [{}, {}], "yourIndex": 0},
                       "select": {"type": 0, "minCount": 1, "maxCount": 1,
                                  "option": [{"type": 7}, {"type": 14}]}}
        rng = random.Random(42)
        for epsilon in (0, 0.2, 1):
            explored = False
            for _ in range(50):
                action, log = select_action(Policy(), observation, rng, epsilon)
                self.assertIn(action, ([0], [1]))
                expected = epsilon / 2 if action == [0] else 1 - epsilon / 2
                self.assertAlmostEqual(log["probability"], expected)
                self.assertAlmostEqual(log["log_probability"], math.log(expected))
                explored |= action == [0]
            self.assertEqual(explored, epsilon > 0)

    def test_fallback_has_no_policy_probabilities(self):
        for count, options in [(0, []), (2, [{}, {}, {}])]:
            observation = {"select": {"type": 1, "minCount": count, "maxCount": count,
                                      "option": options}}
            action, log = select_action(None, observation, random.Random(0), 0.1)
            self.assertEqual(action, list(range(count)))
            self.assertEqual(log, {"source": "heuristic", "probability": None,
                                   "log_probability": None, "learned_probability": None})

    def test_fitted_models_sample_saved_legal_candidates(self):
        folder = Path(__file__).resolve().parent
        with (folder.parents[2] / "experiments/day4/training/train.jsonl").open() as handle:
            observation = json.loads(next(handle))["observation"]
        for loader, artifact in [(Linear, "aj1_bc.json"), (MLP, "aj2_bc.json")]:
            model = loader(folder / artifact)
            action, log = select_action(model, observation, random.Random(42), 0.1)
            [index] = action
            probabilities = model.probabilities_observation(observation)
            self.assertIn(index, range(len(probabilities)))
            self.assertAlmostEqual(log["probability"], 0.9 * probabilities[index] + 0.1 / len(probabilities))
            self.assertAlmostEqual(log["log_probability"], math.log(log["probability"]))

    def test_prize_shaping_and_returns_use_fixed_seats(self):
        observation = {"current": {"players": [{"prize": [None] * 3}, {"prize": [None] * 6}],
                                   "yourIndex": 1}}
        self.assertEqual(prize_lead(observation), 0.5)
        decisions = [{"seat": 0, "prize_lead_seat0": 0.0},
                     {"seat": 1, "prize_lead_seat0": 0.5},
                     {"seat": 1, "prize_lead_seat0": 0.5}]
        terminal, shaped = label_returns(decisions, 0)
        self.assertEqual((terminal, shaped), ([1, -1], [1, -1]))
        self.assertEqual([d["terminal_reward"] for d in decisions], [0, 0, -1])
        self.assertEqual([d["shaped_reward"] for d in decisions], [0.125, 0, -0.875])
        self.assertEqual([d["shaped_return"] for d in decisions], [1, -0.875, -0.875])
        for empty, result in [([], 0), (decisions, -1)]:
            with self.assertRaises(ValueError):
                label_returns(empty, result)
