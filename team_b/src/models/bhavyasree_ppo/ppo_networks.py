
"""Bhavya's PPO actor-critic network architectures."""

import torch
from torch import nn


class SharedActorCritic(nn.Module):
    """Actor and critic share the same state representation."""

    def __init__(self, state_dim=43, option_dim=10, hidden_dim=128):
        super().__init__()

        self.shared_state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.option_encoder = nn.Sequential(
            nn.Linear(option_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, option_vectors):
        """
        state shape: [batch, state_dim]
        option_vectors shape: [batch, number_of_options, option_dim]
        """
        batch_size, option_count, _ = option_vectors.shape

        state_features = self.shared_state(state)
        option_features = self.option_encoder(
            option_vectors.reshape(-1, option_vectors.shape[-1])
        )
        option_features = option_features.reshape(
            batch_size, option_count, -1
        )

        repeated_state = state_features.unsqueeze(1).expand(
            -1, option_count, -1
        )

        combined = torch.cat(
            [repeated_state, option_features],
            dim=-1,
        )

        logits = self.actor_head(combined).squeeze(-1)
        value = self.critic_head(state_features).squeeze(-1)

        return logits, value


class SeparateActorCritic(nn.Module):
    """Actor and critic use separate state representations."""

    def __init__(self, state_dim=43, option_dim=10, hidden_dim=128):
        super().__init__()

        self.actor_state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_option = nn.Sequential(
            nn.Linear(option_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, option_vectors):
        """
        state shape: [batch, state_dim]
        option_vectors shape: [batch, number_of_options, option_dim]
        """
        batch_size, option_count, _ = option_vectors.shape

        actor_state_features = self.actor_state(state)

        actor_option_features = self.actor_option(
            option_vectors.reshape(-1, option_vectors.shape[-1])
        )
        actor_option_features = actor_option_features.reshape(
            batch_size, option_count, -1
        )

        repeated_state = actor_state_features.unsqueeze(1).expand(
            -1, option_count, -1
        )

        combined = torch.cat(
            [repeated_state, actor_option_features],
            dim=-1,
        )

        logits = self.actor_head(combined).squeeze(-1)
        value = self.critic(state).squeeze(-1)

        return logits, value
