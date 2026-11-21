"""B-SR2: Small MLP behavior-cloning policy. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "B-SR2"
ARCHITECTURE = "Small MLP behavior-cloning policy"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("B-SR2 requires implementation and actual training.")
