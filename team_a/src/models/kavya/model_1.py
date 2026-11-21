"""A-K1: Concatenated state-option MLP. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "A-K1"
ARCHITECTURE = "Concatenated state-option MLP"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("A-K1 requires implementation and actual training.")
