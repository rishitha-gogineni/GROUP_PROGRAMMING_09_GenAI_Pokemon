"""B-SA1: Decision Tree regressor. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "B-SA1"
ARCHITECTURE = "Decision Tree regressor"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("B-SA1 requires implementation and actual training.")
