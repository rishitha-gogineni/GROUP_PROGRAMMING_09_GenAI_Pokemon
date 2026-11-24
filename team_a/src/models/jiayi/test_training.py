import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .train import (audit_collection, load_split, fit_normalization, initialize, joined_reward,
                    loss_and_gradients, metrics, normalize, outcome_weights)


def examples():
    return [
        {"x": np.array([[1., 0.], [0., 2.]]), "chosen": 0, "reward": 1., "key": ("a", 0)},
        {"x": np.array([[0., 1.], [2., -1.], [-1., 1.]]), "chosen": 1, "reward": -1., "key": ("b", 0)},
    ]


class TrainingTests(unittest.TestCase):
    def test_chosen_action_loss_and_gradient(self):
        data = examples()
        parameters = {"weights": np.zeros(2)}
        loss, gradient = loss_and_gradients(data, parameters, np.ones(2))
        self.assertAlmostEqual(loss, (np.log(2) + np.log(3)) / 2)
        expected = ((data[0]["x"].mean(axis=0) - data[0]["x"][0]) +
                    (data[1]["x"].mean(axis=0) - data[1]["x"][1])) / 2
        np.testing.assert_allclose(gradient["weights"], expected)
        # A single observed action is the target, not a uniform teacher tie set.
        changed = copy.deepcopy(data)
        changed[0]["chosen"] = 1
        _, alternative = loss_and_gradients(changed, parameters, np.ones(2))
        self.assertFalse(np.allclose(gradient["weights"], alternative["weights"]))

    def test_weighted_gradients_by_finite_difference(self):
        data = examples()
        weights = outcome_weights(data, .5)
        parameter_sets = [
            {"weights": np.array([.2, -.3])},
            {"W1": np.array([[.1, -.2], [.3, .4]]), "b1": np.array([.1, -.1]), "w2": np.array([.5, -.4])},
        ]
        for parameters in parameter_sets:
            for weighting in (np.ones(2), weights):
                _, gradients = loss_and_gradients(data, parameters, weighting)
                for key, values in parameters.items():
                    for index in np.ndindex(values.shape):
                        original = values[index]
                        values[index] = original + 1e-5
                        above = loss_and_gradients(data, parameters, weighting)[0]
                        values[index] = original - 1e-5
                        below = loss_and_gradients(data, parameters, weighting)[0]
                        values[index] = original
                        self.assertAlmostEqual((above - below) / 2e-5, gradients[key][index], places=7)

    def test_outcome_weights_are_positive_and_mean_one(self):
        data = examples()
        np.testing.assert_array_equal(outcome_weights(data, 0), np.ones(2))
        weights = outcome_weights(data, .5)
        self.assertTrue((weights > 0).all())
        self.assertAlmostEqual(weights.mean(), 1.)
        self.assertAlmostEqual(weights[0] / weights[1], np.e)
        loss0 = loss_and_gradients([data[0]], {"weights": np.zeros(2)}, np.array([weights[0]]))[0]
        self.assertAlmostEqual(loss0, weights[0] * np.log(2))

    def test_normalization_uses_only_supplied_training_rows(self):
        train = examples()
        mean, scale, active = fit_normalization(train)
        np.testing.assert_allclose(mean, np.concatenate([e["x"] for e in train]).mean(axis=0))
        validation = [{"x": np.array([[1000., 2000.], [3000., 4000.]])}]
        original = mean.copy()
        normalized = normalize(validation, mean, scale)
        np.testing.assert_array_equal(mean, original)
        np.testing.assert_allclose(normalized[0]["x"], (validation[0]["x"] - mean) / scale)
        constant = [{"x": np.array([[1., 2.], [1., 4.]])}]
        _, constant_scale, active = fit_normalization(constant)
        self.assertEqual(constant_scale[0], 1.)
        self.assertFalse(active[0])

    def test_metrics_entropy_and_exact_choice(self):
        result = metrics(examples(), {"weights": np.zeros(2)})
        self.assertAlmostEqual(result["policy_entropy_nats"], (np.log(2) + np.log(3)) / 2)
        self.assertAlmostEqual(result["normalized_policy_entropy"], 1.)
        self.assertEqual(result["exact_choice_accuracy"], .5)

    def test_terminal_join_both_seats_and_draws(self):
        for seat in (0, 1):
            row = {"game_id": "g", "acting_player": seat, "observation": {"current": {"yourIndex": seat}}}
            for code, outcome, expected in ((seat, "win", 1.), (1-seat, "loss", -1.), (2, "draw", 0.)):
                battle = {"game_id": "g", "candidate_seat": str(seat), "result_code": str(code),
                          "outcome": outcome, "status": "passed", "finished": "True",
                          "invalid_responses": "0", "engine_rejections": "0"}
                self.assertEqual(joined_reward(row, battle), expected)
                for field, bad in (("game_id", "other"), ("result_code", "-1"), ("outcome", "error"),
                                   ("finished", "False"), ("status", "error"), ("engine_rejections", "1")):
                    changed = dict(battle, **{field: bad})
                    with self.assertRaises(ValueError):
                        joined_reward(row, changed)
                changed = copy.deepcopy(row)
                changed["acting_player"] = 1-seat
                with self.assertRaises(ValueError):
                    joined_reward(changed, battle)

    def test_initialization_reproducible_and_constant_features_zero(self):
        active = np.array([True, False])
        for model_id in ("A-J1", "A-J2"):
            first, second = initialize(model_id, active), initialize(model_id, active)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertTrue(np.all(first["weights" if model_id == "A-J1" else "W1"][1] == 0))

    def test_collection_audit_and_prepared_row_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "collection/records").mkdir(parents=True)
            (source / "training").mkdir()
            manifest, battles = {}, []
            for index, split in enumerate(("train", "validation", "held_out")):
                game_id = f"fixture:collection:{index}"
                row = {"game_id": game_id, "run_id": "fixture", "step": 1,
                       "acting_player": 0, "selection_accepted": True,
                       "chosen_option_index": 0, "chosen_option_indices": [0], "legal_option_indices": [0, 1],
                       "observation": {"current": {"yourIndex": 0, "players": [{}, {}]},
                                       "select": {"type": 0, "minCount": 1, "maxCount": 1,
                                                  "option": [{"type": 13}, {"type": 14}]}}}
                text = json.dumps(row) + "\n"
                (source / f"collection/records/{index}.jsonl").write_text(text)
                (source / f"training/{split}.jsonl").write_text(text)
                manifest[split] = {"game_ids": [game_id], "games": 1, "rows": 1}
                battles.append({"game_id": game_id, "candidate_seat": 0, "result_code": 0,
                                "outcome": "win", "status": "passed", "finished": True,
                                "invalid_responses": 0, "engine_rejections": 0,
                                "demo_records": 1, "demo_cap_reached": False, "demo_file": f"records/{index}.jsonl"})
            manifest_path = source / "training/dataset_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            (source / "collection/report.json").write_text(json.dumps(
                {"all_passed": True, "phase": "collection", "unit": "individual_battle",
                 "run_id": "fixture", "battles": battles}))
            with (source / "collection/battle_results.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(battles[0]))
                writer.writeheader()
                writer.writerows(battles)
            audited = audit_collection(source)
            self.assertEqual(len(load_split(source, "train", *audited)), 1)
            path = source / "training/train.jsonl"
            altered = json.loads(path.read_text())
            altered["chosen_option_index"] = 1
            path.write_text(json.dumps(altered) + "\n")
            with self.assertRaises(ValueError):
                load_split(source, "train", *audited)
            manifest["validation"]["game_ids"] = manifest["train"]["game_ids"]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                audit_collection(source)


if __name__ == "__main__":
    unittest.main()
