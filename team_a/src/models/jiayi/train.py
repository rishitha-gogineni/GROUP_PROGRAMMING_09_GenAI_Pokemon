import csv
import json
from pathlib import Path
import time

import numpy as np

from ... import features
from .model_1 import Model as Linear
from .model_2 import Model as MLP
from .policy import ARTIFACT_VERSION, DIMENSION

CONFIGURATIONS = (
    {"name": "aj1_bc", "model_id": "A-J1", "beta": 0.0, "learning_rate": 0.03},
    {"name": "aj2_bc", "model_id": "A-J2", "beta": 0.0, "learning_rate": 0.003},
    {"name": "aj1_weighted_bc", "model_id": "A-J1", "beta": 0.5, "learning_rate": 0.03},
)
SEED, EPOCHS, BATCH_SIZE, HIDDEN = 42, 60, 64, 32


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def joined_reward(row, battle):
    require(row["game_id"] == battle["game_id"], "Game ID mismatch")
    seat, code = int(battle["candidate_seat"]), int(battle["result_code"])
    require(seat in (0, 1) and code in (0, 1, 2), "Invalid seat or terminal result")
    require(battle["status"] == "passed" and str(battle["finished"]).lower() == "true",
            "Incomplete or failed collection game")
    require(int(battle["invalid_responses"]) == int(battle["engine_rejections"]) == 0,
            "Collection game has action errors")
    require(row["acting_player"] == seat == row["observation"]["current"]["yourIndex"],
            "Demonstration is not from the candidate perspective")
    expected = "draw" if code == 2 else "win" if code == seat else "loss"
    require(battle["outcome"] == expected, "Outcome disagrees with terminal result and seat")
    return {"win": 1.0, "loss": -1.0, "draw": 0.0}[expected]


def audit_collection(source):
    collection, training = source / "collection", source / "training"
    manifest = json.loads((training / "dataset_manifest.json").read_text())
    report = json.loads((collection / "report.json").read_text())
    with (collection / "battle_results.csv").open() as handle:
        battles = list(csv.DictReader(handle))
    require(report.get("all_passed") is True, "Collection report did not pass")
    require(report.get("phase") == "collection" and report.get("unit") == "individual_battle",
            "Only the audited individual-battle collection can supply outcomes")
    by_id = {battle["game_id"]: battle for battle in battles}
    reported = {battle["game_id"]: battle for battle in report["battles"]}
    require(len(by_id) == len(battles) == len(reported) == len(report["battles"]), "Duplicate battle ID")
    require(set(by_id) == set(reported), "CSV/report game IDs differ")
    raw = {}
    for game_id, battle in by_id.items():
        require(game_id.startswith(report["run_id"] + ":collection:"), "Non-collection game ID")
        detail = reported[game_id]
        for field in ("candidate_seat", "result_code", "demo_records"):
            require(int(battle[field]) == detail[field], "CSV/report mismatch: " + field)
        require(battle["outcome"] == detail["outcome"], "CSV/report outcome mismatch")
        require(detail["status"] == "passed" and detail["finished"] is True,
                "Report contains incomplete game")
        require(not detail["demo_cap_reached"], "Truncated demonstration game")
        require(not detail.get("cleanup_error") and not detail.get("error_message"), "Collection reported an error")
        relative = Path(detail["demo_file"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid demo path")
        path = collection / relative
        rows = read_rows(path)
        require(len(rows) == detail["demo_records"], "Raw record count mismatch")
        for row in rows:
            require(row["run_id"] == report["run_id"], "Demonstration run ID mismatch")
            joined_reward(row, battle)
            require(row["selection_accepted"] is True, "Rejected demonstration action")
            key = (row["game_id"], row["step"])
            require(key not in raw, "Duplicate decision")
            raw[key] = row
    groups = [set(manifest[split]["game_ids"]) for split in ("train", "validation", "held_out")]
    require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]),
            "Game split leakage")
    require(set.union(*groups) == set(by_id), "Split groups differ from collection games")
    for split, group in zip(("train", "validation", "held_out"), groups):
        require(len(group) == manifest[split]["games"] == len(manifest[split]["game_ids"]),
                "Manifest group count mismatch")
    return manifest, by_id, raw


def load_split(source, split, manifest, battles, raw):
    path = source / "training" / (split + ".jsonl")
    rows = read_rows(path)
    entry = manifest[split]
    require(len(rows) == entry["rows"], "Prepared record count mismatch")
    require({row["game_id"] for row in rows} == set(entry["game_ids"]), "Prepared game groups mismatch")
    examples, seen = [], set()
    for row in rows:
        key = (row["game_id"], row["step"])
        require(key not in seen and raw.get(key) == row, "Duplicate or altered prepared row")
        seen.add(key)
        encoded = features.encode_observation(row["observation"])
        matrix = np.asarray(encoded["options"], dtype=np.float64)
        chosen = row["chosen_option_index"]
        require(type(chosen) is int and 0 <= chosen < len(matrix), "Invalid chosen index")
        require(row["chosen_option_indices"] == [chosen], "Chosen action mismatch")
        require(encoded["option_indices"] == row["legal_option_indices"], "Candidate order mismatch")
        require(np.isfinite(matrix).all() and matrix.shape[1] == DIMENSION, "Invalid candidate features")
        examples.append({"x": matrix, "chosen": chosen,
                         "reward": joined_reward(row, battles[row["game_id"]]), "key": key})
    return examples


def fit_normalization(train):
    matrix = np.concatenate([example["x"] for example in train])
    mean, scale = matrix.mean(axis=0), matrix.std(axis=0)
    active = scale >= 1e-6
    scale[~active] = 1.0
    return mean, scale, active


def normalize(examples, mean, scale):
    return [dict(example, x=(example["x"] - mean) / scale) for example in examples]


def outcome_weights(examples, beta):
    weights = np.exp(beta * np.asarray([example["reward"] for example in examples]))
    return weights / weights.mean()


def pack(examples):
    sizes = np.asarray([len(example["x"]) for example in examples])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    chosen = starts + np.asarray([example["chosen"] for example in examples])
    return np.concatenate([example["x"] for example in examples]), sizes, starts, chosen


def distribution(scores, sizes, starts):
    shifted = scores - np.repeat(np.maximum.reduceat(scores, starts), sizes)
    exponentials = np.exp(shifted)
    totals = np.add.reduceat(exponentials, starts)
    logp = shifted - np.repeat(np.log(totals), sizes)
    return exponentials / np.repeat(totals, sizes), logp


def initialize(model_id, active):
    if model_id == "A-J1":
        return {"weights": np.zeros(len(active))}
    rng = np.random.default_rng(SEED)
    limit = np.sqrt(6 / (int(active.sum()) + HIDDEN))
    W1 = rng.uniform(-limit, limit, (len(active), HIDDEN))
    W1[~active] = 0
    return {"W1": W1, "b1": np.zeros(HIDDEN), "w2": rng.normal(0, 1 / np.sqrt(HIDDEN), HIDDEN)}


def forward(x, parameters):
    if "weights" in parameters:
        return x @ parameters["weights"], None
    hidden = np.tanh(x @ parameters["W1"] + parameters["b1"])
    return hidden @ parameters["w2"], hidden


def loss_and_gradients(examples, parameters, weights):
    x, sizes, starts, chosen = pack(examples)
    scores, hidden = forward(x, parameters)
    probability, logp = distribution(scores, sizes, starts)
    derivative = probability.copy()
    derivative[chosen] -= 1
    # Fixed training-global mean-one weights, not batch-specific renormalization.
    derivative *= np.repeat(weights / len(examples), sizes)
    if hidden is None:
        gradients = {"weights": x.T @ derivative}
    else:
        backprop = derivative[:, None] * parameters["w2"] * (1 - hidden * hidden)
        gradients = {"W1": x.T @ backprop, "b1": backprop.sum(axis=0), "w2": hidden.T @ derivative}
    return float(-weights @ logp[chosen] / len(examples)), gradients


def metrics(examples, parameters):
    x, sizes, starts, chosen = pack(examples)
    scores, _ = forward(x, parameters)
    probability, logp = distribution(scores, sizes, starts)
    predictions = np.asarray([start + int(np.argmax(scores[start:start + size]))
                              for start, size in zip(starts, sizes)])
    entropies = -np.add.reduceat(probability * logp, starts)
    return {"decisions": len(examples), "games": len({e["key"][0] for e in examples}),
            "chosen_action_cross_entropy": float(-logp[chosen].mean()),
            "exact_choice_accuracy": float((predictions == chosen).mean()),
            "policy_entropy_nats": float(entropies.mean()),
            "normalized_policy_entropy": float((entropies / np.log(sizes)).mean())}


def train_one(config, train, validation, active):
    parameters = initialize(config["model_id"], active)
    first = {k: np.zeros_like(v) for k, v in parameters.items()}
    second = {k: np.zeros_like(v) for k, v in parameters.items()}
    weights = outcome_weights(train, config["beta"])
    rng = np.random.default_rng(SEED)
    best_loss, best, best_epoch = float("inf"), None, None
    histories, updates = [], 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train))
        for offset in range(0, len(order), BATCH_SIZE):
            indices = order[offset:offset + BATCH_SIZE]
            batch = [train[int(i)] for i in indices]
            _, gradients = loss_and_gradients(batch, parameters, weights[indices])
            updates += 1
            for key in parameters:
                first[key] = .9 * first[key] + .1 * gradients[key]
                second[key] = .999 * second[key] + .001 * gradients[key] ** 2
                parameters[key] -= config["learning_rate"] * (first[key] / (1 - .9 ** updates)) / (np.sqrt(second[key] / (1 - .999 ** updates)) + 1e-8)
                require(np.isfinite(parameters[key]).all(), "Nonfinite training parameters")
            parameters["weights" if config["model_id"] == "A-J1" else "W1"][~active] = 0
        loss = metrics(validation, parameters)["chosen_action_cross_entropy"]
        histories.append({"epoch": epoch, "validation_cross_entropy": loss})
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best = {key: value.copy() for key, value in parameters.items()}
        if epoch == 1 or epoch % 10 == 0:
            print(config["name"], "epoch", epoch, "validation CE", round(loss, 6), flush=True)
    result = {"selected_epoch": best_epoch, "training_seconds": time.perf_counter() - started,
              "train": metrics(train, best), "validation": metrics(validation, best)}
    return best, result, histories


def export(parameters, config, mean, scale, result, path):
    artifact = {"artifact_version": ARTIFACT_VERSION, "model_id": config["model_id"],
                "schema_version": features.SCHEMA_VERSION, "feature_names": list(features.FEATURE_NAMES),
                "mean": mean.tolist(), "scale": scale.tolist(),
                **{key: value.tolist() for key, value in parameters.items()},
                "training": {"objective": "chosen_action_bc" if config["beta"] == 0 else "outcome_weighted_chosen_action_bc",
                             "beta": config["beta"], "seed": SEED, "selected_epoch": result["selected_epoch"]}}
    if config["model_id"] == "A-J2":
        artifact.update(hidden_units=HIDDEN, activation="tanh")
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def check_reload(path, model_id, parameters, raw_examples, mean, scale):
    model = (Linear if model_id == "A-J1" else MLP)(path)
    maximum = 0.0
    for example in raw_examples:
        matrix = example["x"]
        expected, _ = forward((matrix - mean) / scale, parameters)
        actual = np.asarray(model.score_options([], matrix.tolist()))
        require(np.allclose(actual, expected, atol=1e-9, rtol=1e-9), "Exported score parity failed")
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        expected_probability, _ = distribution(expected, np.array([len(matrix)]), np.array([0]))
        actual_probability = model.policy_probabilities([], matrix.tolist())
        require(np.allclose(actual_probability, expected_probability, atol=1e-10, rtol=1e-9), "Probability parity failed")
        require(np.allclose(model.score_options([], matrix[::-1].tolist()), actual[::-1], atol=1e-9, rtol=1e-9),
                "Candidate order parity failed")
    return {"passed": True, "decisions_checked": len(raw_examples), "max_absolute_logit_error": maximum,
            "checks": ["logits", "probabilities", "reversed_candidate_order"]}

