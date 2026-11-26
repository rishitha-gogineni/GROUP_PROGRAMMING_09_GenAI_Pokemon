
"""Rollout storage and generalized advantage estimation for PPO."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Transition:
    state: Any
    options: Any
    action: list[int]
    old_log_probability: float
    value: float
    reward: float
    done: bool


class RolloutBuffer:
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.transitions = []

    def add(
        self,
        state,
        options,
        action,
        old_log_probability,
        value,
        reward,
        done,
    ):
        self.transitions.append(
            Transition(
                state=state,
                options=options,
                action=list(action),
                old_log_probability=float(
                    old_log_probability.detach().item()
                    if isinstance(old_log_probability, torch.Tensor)
                    else old_log_probability
                ),
                value=float(
                    value.detach().item()
                    if isinstance(value, torch.Tensor)
                    else value
                ),
                reward=float(reward),
                done=bool(done),
            )
        )

    def compute_advantages_and_returns(self, last_value=0.0):
        if not self.transitions:
            raise ValueError("The rollout buffer is empty.")

        advantages = [0.0] * len(self.transitions)
        returns = [0.0] * len(self.transitions)

        gae = 0.0
        next_value = float(last_value)

        for index in reversed(range(len(self.transitions))):
            transition = self.transitions[index]

            non_terminal = 0.0 if transition.done else 1.0

            delta = (
                transition.reward
                + self.gamma * next_value * non_terminal
                - transition.value
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * non_terminal
                * gae
            )

            advantages[index] = gae
            returns[index] = gae + transition.value
            next_value = transition.value

        advantages_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
        )

        returns_tensor = torch.tensor(
            returns,
            dtype=torch.float32,
        )

        normalized_advantages = (
            advantages_tensor - advantages_tensor.mean()
        ) / (advantages_tensor.std(unbiased=False) + 1e-8)

        return normalized_advantages, returns_tensor

    def clear(self):
        self.transitions.clear()

    def __len__(self):
        return len(self.transitions)
