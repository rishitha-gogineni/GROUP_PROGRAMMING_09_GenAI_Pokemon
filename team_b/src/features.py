"""Versioned acting-player state and candidate encoders. NOT IMPLEMENTED."""

from .contracts import FeatureVector, Observation

SCHEMA_VERSION = "draft-v0"


def encode_state(observation: Observation) -> FeatureVector:
    """Use only information available to the acting player."""
    raise NotImplementedError("Define schema after inspecting real observations.")


def encode_options(observation: Observation) -> list[FeatureVector]:
    """Preserve a mapping to current legal option indices and selection context."""
    raise NotImplementedError("Handle variable candidate sets and explicit unknown values.")
