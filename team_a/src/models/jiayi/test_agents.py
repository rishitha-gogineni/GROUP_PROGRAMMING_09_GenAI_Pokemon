import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from ... import features, heuristic
from ...common_runtime import load_agent
from ...simulator import check_action
from .agents import choose_action, stage_policy
from .model_1 import Model as Linear
from .model_2 import Model as MLP


ROOT = Path(__file__).resolve().parents[4]


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "team_a/experiments/day4/training/train.jsonl").open() as handle:
            cls.observation = json.loads(next(handle))["observation"]

    def test_staged_entries_load_real_weights_without_file_or_working_directory(self):
        for model_id, loader in (("A-J1", Linear), ("A-J2", MLP)):
            with self.subTest(model=model_id), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "agent"
                deck = Path(temporary) / "deck.csv"
                deck.write_text("1\n" * 60)  # Callback fixture, not a gameplay-valid deck.
                stage_policy(model_id, destination, deck)
                self.assertEqual({p.name for p in destination.iterdir()},
                                 {"main.py", "model.json", "deck.csv"})
                callback, namespace = load_agent(destination)
                self.assertNotIn("__file__", namespace)
                self.assertEqual(callback({"select": None}),
                                 [int(line) for line in deck.read_text().splitlines() if line.strip()])
                model = loader(destination / "model.json")
                for observation in (self.observation, self.reversed_observation()):
                    scores = model.score_observation(observation)
                    action = callback(observation)
                    self.assertEqual(action, [max(range(len(scores)), key=scores.__getitem__)])
                    check_action(action, observation["select"])
                # The actual worker starts outside the repository with repo imports enabled.
                script = '''import json, sys
sys.path.insert(0, sys.argv[1])
from team_a.src.common_runtime import load_agent
callback, namespace = load_agent(sys.argv[2])
assert "__file__" not in namespace
assert len(callback({"select": None})) == 60
assert callback(json.loads(sys.stdin.read())) == json.loads(sys.argv[3])
'''
                result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT),
                                         str(destination), json.dumps(callback(self.observation))],
                                        input=json.dumps(self.observation), text=True,
                                        cwd=temporary, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def reversed_observation(self):
        observation = copy.deepcopy(self.observation)
        observation["select"]["option"].reverse()
        return observation

    def test_mapping_ties_and_inference_errors(self):
        encoded = features.encode_observation(self.observation)
        count = len(encoded["options"])
        encoded["option_indices"] = list(reversed(range(count)))
        with patch("team_a.src.models.jiayi.agents.features.encode_observation", return_value=encoded):
            model = Mock()
            model.score_options.return_value = [1.0] * count
            self.assertEqual(choose_action(model, self.observation), [count - 1])
            model.score_options.side_effect = ValueError("Invalid parameters")
            with self.assertRaisesRegex(ValueError, "Invalid parameters"):
                choose_action(model, self.observation)

    def test_fallback_single_multiple_and_empty_selections(self):
        for options, minimum, maximum in (([{}], 1, 1), ([{}, {}], 2, 2), ([], 0, 0)):
            observation = {"current": {}, "select": {"type": 1, "option": options,
                                                       "minCount": minimum, "maxCount": maximum}}
            model = Mock()
            action = choose_action(model, observation)
            self.assertEqual(action, heuristic.agent(observation))
            check_action(action, observation["select"])
            model.score_options.assert_not_called()

    def test_missing_staged_weight_is_not_silently_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "agent"
            deck = Path(temporary) / "deck.csv"
            deck.write_text("1\n" * 60)
            stage_policy("A-J1", destination, deck)
            (destination / "model.json").unlink()
            callback, _ = load_agent(destination)
            with self.assertRaises(FileNotFoundError):
                callback(self.observation)


if __name__ == "__main__":
    unittest.main()
