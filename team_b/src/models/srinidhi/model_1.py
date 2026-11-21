"""B-SR1: Linear contextual behavior-cloning policy. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "B-SR1"
ARCHITECTURE = "Linear contextual behavior-cloning policy"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("B-SR1 requires implementation and actual training.")
