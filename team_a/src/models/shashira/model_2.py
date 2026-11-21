"""A-S2: Dueling-style state value and legal-candidate advantage network. NOT IMPLEMENTED."""

from ...contracts import FeatureVector

MODEL_ID = "A-S2"
ARCHITECTURE = "Dueling-style state value and legal-candidate advantage network"


class Model:
    """Implement training/artifact loading separately; do not fabricate fitted weights."""

    def score_options(
        self, state: FeatureVector, options: list[FeatureVector]
    ) -> list[float]:
        """Return finite scores mapped to the current option list."""
        raise NotImplementedError("A-S2 requires implementation and actual training.")
