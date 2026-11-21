"""A-J1: Linear contextual masked policy. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "A-J1"
ARCHITECTURE = "Linear contextual masked policy"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("A-J1 requires implementation and actual training.")
