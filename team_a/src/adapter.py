"""Full SDK-response adapter. Owner: Rishitha. NOT IMPLEMENTED."""

from .contracts import Observation


def agent(observation: Observation) -> list[int]:
    """Confirm actual SDK callback/response. Handle initial, single and multi-choice phases."""
    raise NotImplementedError(
        "Integrate the official example, count/context constraints and verified fallback."
    )
