from .policy import CandidatePolicy, DIMENSION, finite_vector

MODEL_ID = "A-J1"
ARCHITECTURE = "Linear contextual masked policy"


class Model(CandidatePolicy):
    MODEL_ID = MODEL_ID

    def _load_weights(self, parameters):
        self.weights = finite_vector(parameters.get("weights"), DIMENSION, "weights")

    def _score(self, candidate):
        return sum(value * weight for value, weight in zip(candidate, self.weights))
