"""B-B2: Separate actor and critic networks. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "B-B2"
ARCHITECTURE = "Separate actor and critic networks"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("B-B2 requires implementation and actual training.")
