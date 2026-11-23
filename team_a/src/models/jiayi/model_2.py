import math

from .policy import CandidatePolicy, DIMENSION, finite_vector

MODEL_ID = "A-J2"
ARCHITECTURE = "Small MLP masked policy"


class Model(CandidatePolicy):
    MODEL_ID = MODEL_ID

    def _load_weights(self, parameters):
        hidden = parameters.get("hidden_units")
        if type(hidden) is not int or hidden <= 0:
            raise ValueError("hidden_units must be a positive integer.")
        if parameters.get("activation") != "tanh":
            raise ValueError("A-J2 requires tanh activation.")
        rows = parameters.get("W1")
        if not isinstance(rows, (list, tuple)) or len(rows) != DIMENSION:
            raise ValueError(f"W1 must have {DIMENSION} input rows.")
        self.W1 = tuple(finite_vector(row, hidden, "W1 row") for row in rows)
        self.b1 = finite_vector(parameters.get("b1"), hidden, "b1")
        self.w2 = finite_vector(parameters.get("w2"), hidden, "w2")
        self.hidden_units = hidden

    def _score(self, candidate):
        preactivation = [bias + sum(value * row[j] for value, row in zip(candidate, self.W1))
                         for j, bias in enumerate(self.b1)]
        if not all(math.isfinite(value) for value in preactivation):
            raise ValueError("MLP produced nonfinite hidden activations.")
        return sum(math.tanh(value) * weight for value, weight in zip(preactivation, self.w2))
