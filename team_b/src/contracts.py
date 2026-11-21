"""Minimal internal scoring contract, not the official simulator API."""

from typing import Mapping, Protocol, Sequence

Observation = Mapping[str, object]
FeatureVector = Sequence[float]


class OptionScorer(Protocol):
    """Score current candidates; output position maps to current options only."""

    def score_options(
        self, state: FeatureVector, options: Sequence[FeatureVector]
    ) -> Sequence[float]:
        """Return one finite score per option; selection is a separate responsibility."""
        ...
