"""A-S1: Ordinary candidate-value network. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "A-S1"
ARCHITECTURE = "Ordinary candidate-value network"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("A-S1 requires implementation and actual training.")
