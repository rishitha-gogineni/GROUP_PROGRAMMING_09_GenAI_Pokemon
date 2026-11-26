
"""Bhavya's PPO observation and option encoder."""

import math
from numbers import Real


def _number(value, default=0.0):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, Real):
        value = float(value)
        return value if math.isfinite(value) else default
    return default


def _length(value):
    if value is None:
        return 0.0
    try:
        return float(len(value))
    except TypeError:
        return 0.0


def _status(player):
    return [
        _number(player.get("asleep")),
        _number(player.get("burned")),
        _number(player.get("confused")),
        _number(player.get("paralyzed")),
        _number(player.get("poisoned")),
    ]


def encode_state(observation):
    """Convert visible game information into a fixed-length vector."""
    current = observation.get("current") or {}
    players = current.get("players") or []
    select = observation.get("select") or {}

    features = [
        _number(current.get("turn")) / 100.0,
        _number(current.get("round")) / 10.0,
        _number(current.get("yourIndex")),
        _number(current.get("firstPlayer")),
        _number(current.get("energyAttached")),
        _number(current.get("stadiumPlayed")),
        _number(current.get("supporterPlayed")),
        _number(current.get("retreated")),
        _number(current.get("result")) / 2.0,
        _number(current.get("turnActionCount")) / 20.0,
    ]

    for player in players[:2]:
        features.extend([
            _number(player.get("deckCount")) / 60.0,
            _number(player.get("handCount")) / 20.0,
            _length(player.get("hand")) / 20.0,
            _length(player.get("active")) / 5.0,
            _length(player.get("bench")) / 5.0,
            _length(player.get("discard")) / 60.0,
            _length(player.get("prize")) / 6.0,
            _number(player.get("benchMax")) / 5.0,
            _number(player.get("win")),
        ])
        features.extend(_status(player))

    features.extend([
        _number(select.get("type")) / 20.0,
        _number(select.get("context")) / 100.0,
        _number(select.get("minCount")) / 10.0,
        _number(select.get("maxCount")) / 10.0,
        _length(select.get("option")) / 20.0,
    ])

    return features


def encode_options(observation):
    """Encode each current legal option separately."""
    select = observation.get("select") or {}
    options = select.get("option") or []
    count = max(len(options), 1)

    encoded = []

    for index, option in enumerate(options):
        option = option if isinstance(option, dict) else {}

        values = [
            _number(option.get("type")) / 20.0,
            _number(option.get("context")) / 100.0,
            _number(option.get("contextCard")) / 100000.0,
            _number(option.get("deck")) / 60.0,
            _number(option.get("effect")) / 100.0,
            _number(option.get("minCount")) / 10.0,
            _number(option.get("maxCount")) / 10.0,
            _number(option.get("remainDamageCounter")) / 100.0,
            _number(option.get("remainEnergyCost")) / 10.0,
            float(index) / float(count),
        ]

        encoded.append(values)

    return encoded
