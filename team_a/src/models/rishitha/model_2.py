"""A-R2: Small MLP imitation scorer. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "A-R2"
ARCHITECTURE = "Small MLP imitation scorer"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("A-R2 requires implementation and actual training.")
