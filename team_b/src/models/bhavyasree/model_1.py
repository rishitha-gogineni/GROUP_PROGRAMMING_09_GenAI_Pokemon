"""B-B1: Shared actor-critic backbone. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "B-B1"
ARCHITECTURE = "Shared actor-critic backbone"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("B-B1 requires implementation and actual training.")
