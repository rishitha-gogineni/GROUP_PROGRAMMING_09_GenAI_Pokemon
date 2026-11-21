"""A-K2: Separate state and option encoders with joint score head. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "A-K2"
ARCHITECTURE = "Separate state and option encoders with joint score head"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("A-K2 requires implementation and actual training.")
