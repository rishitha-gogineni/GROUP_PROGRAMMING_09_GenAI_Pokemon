"""Shared entry point for the frozen ar1-visible-v1 candidate features.

Supported scope: MAIN, one required choice, at least two current legal options.
Candidate vectors already contain state interactions. Pass an empty state to
the A-R1/A-R2 score_options interfaces. Normalization belongs to each model.
"""

from collections.abc import Mapping

from .contracts import FeatureVector, Observation
from .models.rishitha.ar1 import features as _encoder

SCHEMA_VERSION = _encoder.SCHEMA_VERSION
FEATURE_NAMES = _encoder.FEATURE_NAMES
STATE_NAMES = _encoder.STATE_NAMES
OPTION_NAMES = _encoder.OPTION_NAMES


def supports_observation(observation: Observation) -> bool:
    if not isinstance(observation, Mapping):
        return False
    current, selection = observation.get('current'), observation.get('select')
    if not isinstance(current, Mapping) or not isinstance(selection, Mapping):
        return False
    players, options = current.get('players'), selection.get('option')
    if not isinstance(players, list) or len(players) != 2 or not all(isinstance(p, Mapping) for p in players):
        return False
    if not isinstance(options, list) or not all(isinstance(option, Mapping) for option in options):
        return False
    return (type(current.get('yourIndex')) is int and type(selection.get('type')) is int
            and type(selection.get('minCount')) is int and type(selection.get('maxCount')) is int
            and _encoder.supported_main(observation))


def _require_supported(observation):
    if not supports_observation(observation):
        raise ValueError('ar1-visible-v1 supports MAIN with one required choice and at least two legal options. Use the agent fallback for other decisions.')


def encode_state(observation: Observation) -> FeatureVector:
    """Return an empty state because the candidate vectors include interactions."""
    _require_supported(observation)
    return []


def encode_options(observation: Observation) -> list[FeatureVector]:
    """Return raw 375-element candidate vectors in current legal option order."""
    _require_supported(observation)
    return _encoder.candidate_features(observation)


def encode_observation(observation: Observation) -> dict:
    """Include the explicit mapping from candidate rows to legal option indices."""
    options = encode_options(observation)
    return {'schema_version': SCHEMA_VERSION, 'state': [], 'options': options,
            'option_indices': list(range(len(options)))}


def describe_state(observation: Observation) -> dict[str, float]:
    """Inspect the 21 visible state values; these are not a separate model input."""
    _require_supported(observation)
    return dict(zip(STATE_NAMES, _encoder.state_features(observation)))
