
"""PPO clipped objective for Bhavya's agents."""

import torch
from torch import nn


def ppo_loss(
    new_log_probability,
    old_log_probability,
    advantages,
    predicted_values,
    returns,
    logits,
    clip_epsilon=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
):
    """Calculate PPO policy, value, entropy, and total losses."""

    ratio = torch.exp(new_log_probability - old_log_probability)

    unclipped_objective = ratio * advantages

    clipped_ratio = torch.clamp(
        ratio,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )

    clipped_objective = clipped_ratio * advantages

    policy_loss = -torch.min(
        unclipped_objective,
        clipped_objective,
    ).mean()

    value_loss = 0.5 * (predicted_values - returns).pow(2).mean()

    distribution = torch.distributions.Categorical(logits=logits)
    entropy = distribution.entropy().mean()

    total_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )

    return {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "ratio_mean": ratio.mean(),
    }
