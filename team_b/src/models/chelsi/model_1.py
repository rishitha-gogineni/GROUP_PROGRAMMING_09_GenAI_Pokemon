"""B-C1: Linear contextual card-action scorer. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "B-C1"
ARCHITECTURE = "Linear contextual card-action scorer"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("B-C1 requires implementation and actual training.")
