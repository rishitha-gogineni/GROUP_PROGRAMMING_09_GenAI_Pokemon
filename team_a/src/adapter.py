"""Repository callback adapter for Team A's submitted random baseline."""

from .baseline import baseline_response


def agent(observation, configuration=None):
    """Delegate deck selection and legal-option choices to the baseline."""
    return baseline_response(observation, configuration)
