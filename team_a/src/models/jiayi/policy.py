import json
import math
from pathlib import Path

from ... import features

ARTIFACT_VERSION = 1
DIMENSION = len(features.FEATURE_NAMES)


def finite_vector(values, size, name):
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{name} must contain {size} numbers.")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain finite numbers.")
    return tuple(float(value) for value in values)


def softmax(scores):
    scores = finite_vector(scores, len(scores), "scores")
    if not scores:
        raise ValueError("A policy distribution requires at least one legal candidate.")
    maximum = max(scores)
    exponentials = [math.exp(score - maximum) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


class CandidatePolicy:
    def __init__(self, model_path=None, *, parameters=None):
        if (model_path is None) == (parameters is None):
            raise ValueError("Supply exactly one of model_path or parameters.")
        if model_path is not None:
            parameters = json.loads(Path(model_path).read_text(encoding="utf-8"))
        if not isinstance(parameters, dict):
            raise ValueError("Model artifact must be a JSON object.")
        if type(parameters.get("artifact_version")) is not int or parameters["artifact_version"] != ARTIFACT_VERSION:
            raise ValueError("Unsupported artifact_version.")
        if parameters.get("model_id") != self.MODEL_ID:
            raise ValueError("Model ID mismatch.")
        if parameters.get("schema_version") != features.SCHEMA_VERSION:
            raise ValueError("Feature schema mismatch.")
        if parameters.get("feature_names") != list(features.FEATURE_NAMES):
            raise ValueError("Feature names or order mismatch.")
        self.mean = finite_vector(parameters.get("mean"), DIMENSION, "mean")
        self.scale = finite_vector(parameters.get("scale"), DIMENSION, "scale")
        if any(value <= 0 for value in self.scale):
            raise ValueError("Normalization scales must be positive.")
        self._load_weights(parameters)

    def _load_weights(self, parameters):
        raise NotImplementedError

    def _score(self, candidate):
        raise NotImplementedError

    def score_options(self, state, options):
        if len(state) != 0:
            raise ValueError("Pass an empty state; candidates include state interactions.")
        result = []
        for option in options:
            raw = finite_vector(option, DIMENSION, "candidate")
            normalized = tuple((value - mean) / scale for value, mean, scale
                               in zip(raw, self.mean, self.scale))
            if not all(math.isfinite(value) for value in normalized):
                raise ValueError("Normalization produced nonfinite features.")
            score = self._score(normalized)
            if not math.isfinite(score):
                raise ValueError("Model produced a nonfinite score.")
            result.append(score)
        return result

    def policy_probabilities(self, state, options):
        return softmax(self.score_options(state, options))

    def score_observation(self, observation):
        encoded = features.encode_observation(observation)
        return self.score_options(encoded["state"], encoded["options"])

    def probabilities_observation(self, observation):
        return softmax(self.score_observation(observation))
