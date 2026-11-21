"""B-C2: Small card-action MLP. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "B-C2"
ARCHITECTURE = "Small card-action MLP"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("B-C2 requires implementation and actual training.")
