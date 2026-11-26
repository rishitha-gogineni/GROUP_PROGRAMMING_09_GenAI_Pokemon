
"""Legal-action selection for Bhavya's PPO agents."""

import torch

from .state_encoder import encode_state, encode_options


def choose_action(model, observation, deterministic=False):
    """
    Select legal option indices.

    Returns:
        response: list[int]
        log_probability: tensor
        value: tensor
        logits: tensor
    """
    select = observation.get("select")

    if select is None:
        raise ValueError("Deck phase has no selectable options.")

    options = select.get("option") or []

    if not options:
        raise ValueError("No legal options were provided.")

    state = torch.tensor(
        [encode_state(observation)],
        dtype=torch.float32,
    )

    option_vectors = torch.tensor(
        [encode_options(observation)],
        dtype=torch.float32,
    )

    logits, value = model(state, option_vectors)

    logits = logits.squeeze(0)
    value = value.squeeze(0)

    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = torch.softmax(logits, dim=-1)

    minimum = int(select.get("minCount", 1))
    maximum = int(select.get("maxCount", 1))

    number_to_select = min(maximum, len(options))

    if number_to_select < minimum:
        raise ValueError(
            f"Cannot select {minimum} options from {len(options)} candidates."
        )

    if deterministic:
        selected = torch.topk(
            logits,
            k=number_to_select,
        ).indices
    else:
        selected = torch.multinomial(
            probabilities,
            num_samples=number_to_select,
            replacement=False,
        )

    selected_indices = sorted(int(index) for index in selected.tolist())

    selected_log_probability = log_probabilities[selected].sum()

    return (
        selected_indices,
        selected_log_probability,
        value,
        logits,
    )
