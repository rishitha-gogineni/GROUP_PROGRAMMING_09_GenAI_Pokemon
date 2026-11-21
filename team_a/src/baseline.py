"""Baseline teacher/full-response policy. NOT IMPLEMENTED."""

from .contracts import Observation


def baseline_response(observation: Observation) -> list[int]:
    """Return a complete valid response, not a universal argmax/top-k."""
    raise NotImplementedError("Implement and verify every supported SDK decision phase.")


def teacher_option_scores(observation: Observation) -> list[float]:
    """Document heuristic score targets; do not call them true action values."""
    raise NotImplementedError("Define scores for the explicitly supported single-choice contexts.")
