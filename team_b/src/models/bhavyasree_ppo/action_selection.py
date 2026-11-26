"""Legal-action selection for Bhavya's PPO agents."""

from itertools import permutations

import torch

from .state_encoder import encode_state, encode_options


def _unordered_response_log_probability(logits, response):
    """
    Calculate the probability of an unordered response selected without
    replacement.

    torch.multinomial samples an ordered sequence, but the game response
    is treated as an unordered sorted list of option indices.
    """
    selected = [int(index) for index in response]
    number_of_options = logits.numel()

    # Selecting every available option has probability exactly 1.
    if len(selected) == 0 or len(selected) == number_of_options:
        return logits.new_tensor(0.0)

    log_probabilities = torch.log_softmax(logits, dim=-1)
    ordered_log_probabilities = []

    for order in permutations(selected):
        available = torch.ones(
            number_of_options,
            dtype=torch.bool,
            device=logits.device,
        )

        order_log_probability = logits.new_tensor(0.0)

        for index in order:
            denominator = torch.logsumexp(
                log_probabilities[available],
                dim=0,
            )

            order_log_probability += (
                log_probabilities[index] - denominator
            )

            available = available.clone()
            available[index] = False

        ordered_log_probabilities.append(order_log_probability)

    # Add probabilities of all possible sampling orders.
    return torch.logsumexp(
        torch.stack(ordered_log_probabilities),
        dim=0,
    )


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

    selected_indices = sorted(
        int(index) for index in selected.tolist()
    )

    selected_log_probability = _unordered_response_log_probability(
        logits,
        selected_indices,
    )

    return (
        selected_indices,
        selected_log_probability,
        value,
        logits,
    )
