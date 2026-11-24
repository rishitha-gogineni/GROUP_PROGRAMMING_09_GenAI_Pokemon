import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from ... import features
from .model_1 import Model as Linear
from .model_2 import Model as MLP
from .policy import ARTIFACT_VERSION, DIMENSION, softmax


def fixture(model_id):
    parameters = {
        "artifact_version": ARTIFACT_VERSION,
        "model_id": model_id,
        "schema_version": features.SCHEMA_VERSION,
        "feature_names": list(features.FEATURE_NAMES),
        "mean": [0.0] * DIMENSION,
        "scale": [1.0] * DIMENSION,
    }
    if model_id == "A-J1":
        parameters["weights"] = [0.0] * DIMENSION
        parameters["weights"][:2] = [2.0, -3.0]
    else:
        parameters.update(activation="tanh", hidden_units=2,
                          W1=[[0.0, 0.0] for _ in range(DIMENSION)],
                          b1=[0.5, -0.5], w2=[2.0, -1.0])
        parameters["W1"][0] = [1.0, -1.0]
        parameters["W1"][1] = [0.5, 2.0]
    return parameters


def candidate(first=0.0, second=0.0):
    return [first, second] + [0.0] * (DIMENSION - 2)


class InferenceTests(unittest.TestCase):
    def test_linear_hand_calculation_with_normalization(self):
        parameters = fixture("A-J1")
        parameters["mean"][:2] = [1.0, 2.0]
        parameters["scale"][:2] = [2.0, 4.0]
        model = Linear(parameters=parameters)
        # (5-1)/2 * 2 + (6-2)/4 * -3 = 1.
        self.assertEqual(model.score_options([], [candidate(5, 6)]), [1.0])

    def test_mlp_hand_calculation_with_normalization(self):
        parameters = fixture("A-J2")
        parameters["mean"][0], parameters["scale"][0] = 1.0, 2.0
        model = MLP(parameters=parameters)
        # Raw (5, 1) becomes (2, 1); hidden preactivations are (3, -0.5).
        expected = 2 * math.tanh(3) - math.tanh(-0.5)
        self.assertAlmostEqual(model.score_options([], [candidate(5, 1)])[0], expected)

    def test_softmax_known_values_and_shift_invariance(self):
        expected = [0.25, 0.75]
        for scores in ([0, math.log(3)], [1000, 1000 + math.log(3)]):
            for actual, target in zip(softmax(scores), expected):
                self.assertAlmostEqual(actual, target)
        self.assertEqual(softmax([1e308, -1e308]), [1.0, 0.0])
        self.assertEqual(softmax([7, 7]), [0.5, 0.5])
        self.assertEqual(softmax([7]), [1.0])

    def test_candidate_permutation_preserves_mapping(self):
        options = [candidate(1, 0), candidate(0, 1), candidate(2, 1)]
        order = [2, 0, 1]
        for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
            model = cls(parameters=fixture(name))
            scores = model.score_options([], options)
            probabilities = model.policy_probabilities([], options)
            self.assertEqual(model.score_options([], [options[i] for i in order]), [scores[i] for i in order])
            reordered = model.policy_probabilities([], [options[i] for i in order])
            for actual, expected in zip(reordered, [probabilities[i] for i in order]):
                self.assertAlmostEqual(actual, expected)
            self.assertAlmostEqual(sum(probabilities), 1.0)
            self.assertTrue(all(0 <= value <= 1 for value in probabilities))

    def test_json_loading_matches_explicit_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.json"
            for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
                parameters = fixture(name)
                path.write_text(json.dumps(parameters))
                self.assertEqual(cls(path).score_options([], [candidate(2, 3)]),
                                 cls(parameters=parameters).score_options([], [candidate(2, 3)]))
            path.write_text("{invalid")
            with self.assertRaises(ValueError):
                Linear(path)
            with self.assertRaises(FileNotFoundError):
                Linear(Path(folder) / "missing.json")

    def test_explicit_parameter_source_required(self):
        for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
            with self.assertRaises(ValueError):
                cls()
            with self.assertRaises(ValueError):
                cls("unused.json", parameters=fixture(name))

    def test_artifact_contract_rejects_invalid_metadata_and_normalization(self):
        mutations = [
            ("artifact_version", 99), ("artifact_version", True),
            ("model_id", "A-R1"), ("schema_version", "draft-v0"),
            ("feature_names", list(reversed(features.FEATURE_NAMES))),
            ("mean", [0.0]), ("scale", [0.0] * DIMENSION),
            ("scale", [-1.0] * DIMENSION), ("mean", [math.nan] * DIMENSION),
            ("scale", [math.inf] * DIMENSION), ("mean", [True] * DIMENSION),
        ]
        for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
            for key, value in mutations:
                with self.subTest(model=name, field=key, value=str(value)[:30]):
                    parameters = fixture(name)
                    parameters[key] = value
                    with self.assertRaises(ValueError):
                        cls(parameters=parameters)
            parameters = fixture(name)
            del parameters["mean"]
            with self.assertRaises(ValueError):
                cls(parameters=parameters)

    def test_architecture_parameters_rejected(self):
        for key, value in [("weights", []), ("weights", [math.inf] * DIMENSION)]:
            parameters = fixture("A-J1")
            parameters[key] = value
            with self.assertRaises(ValueError):
                Linear(parameters=parameters)
        for key, value in [("hidden_units", 0), ("hidden_units", True),
                           ("activation", "relu"), ("W1", []),
                           ("W1", [[1.0]] * DIMENSION),
                           ("W1", [[math.nan, 0.0]] * DIMENSION),
                           ("b1", [0]), ("w2", [0, math.inf])]:
            with self.subTest(field=key):
                parameters = fixture("A-J2")
                parameters[key] = value
                with self.assertRaises(ValueError):
                    MLP(parameters=parameters)

    def test_input_and_empty_candidate_contract(self):
        for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
            model = cls(parameters=fixture(name))
            self.assertEqual(model.score_options([], []), [])
            with self.assertRaises(ValueError):
                model.policy_probabilities([], [])
            with self.assertRaises(ValueError):
                model.score_options([1], [candidate()])
            for bad in ([], [math.nan] * DIMENSION, [math.inf] * DIMENSION):
                with self.assertRaises(ValueError):
                    model.score_options([], [bad])
            for observation in ({}, {"select": None}):
                with self.assertRaises(ValueError):
                    model.score_observation(observation)

    def test_parameter_inputs_are_copied(self):
        for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
            parameters = fixture(name)
            model = cls(parameters=parameters)
            before = model.score_options([], [candidate(1, 2)])
            parameters["mean"][0] = 100
            if name == "A-J1":
                parameters["weights"][0] = 999
            else:
                parameters["W1"][0][0] = 999
            self.assertEqual(before, model.score_options([], [candidate(1, 2)]))

    def test_overflow_is_rejected(self):
        for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
            parameters = fixture(name)
            parameters["scale"][0] = 1e-308
            with self.assertRaises(ValueError):
                cls(parameters=parameters).score_options([], [candidate(1e308)])
            parameters = fixture(name)
            if name == "A-J1":
                parameters["weights"][0] = 1e308
            else:
                parameters["W1"][0][0] = 1e308
            with self.assertRaises(ValueError):
                cls(parameters=parameters).score_options([], [candidate(1e308)])

    def test_saved_observation_uses_shared_encoder(self):
        root = Path(__file__).resolve().parents[3]
        path = root / "experiments/day4/training/train.jsonl"
        with path.open() as handle:
            observation = json.loads(next(handle))["observation"]
        options = features.encode_options(observation)
        for cls, name in ((Linear, "A-J1"), (MLP, "A-J2")):
            model = cls(parameters=fixture(name))
            self.assertEqual(model.score_observation(observation), model.score_options([], options))
            self.assertEqual(model.probabilities_observation(observation), model.policy_probabilities([], options))
            changed = copy.deepcopy(observation)
            changed["select"]["maxCount"] = 2
            with self.assertRaises(ValueError):
                model.probabilities_observation(changed)

    def test_softmax_rejects_invalid_scores(self):
        for scores in ([], [math.nan], [math.inf], [True]):
            with self.assertRaises(ValueError):
                softmax(scores)


if __name__ == "__main__":
    unittest.main()
