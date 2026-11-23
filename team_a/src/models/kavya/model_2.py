"""A-K2: Separate state and option encoders with joint score head."""

import math
import torch
from torch import nn
from ...contracts import FeatureVector

MODEL_ID = "A-K2"
ARCHITECTURE = "Separate state and option encoders with joint score head"

STATE_DIM = 21
OPTION_DIM = 30

class Model(nn.Module):
    """Score legal actions using separate state and option encoders."""

    def __init__(self):
        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.option_encoder = nn.Sequential(
            nn.Linear(OPTION_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.score_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        options: torch.Tensor,
    ) -> torch.Tensor:
        """Return one score per legal option."""
        if state.ndim != 1 or state.shape[0] != STATE_DIM:
            raise ValueError("Expected 21 state features.")

        if options.ndim != 2 or options.shape[1] != OPTION_DIM:
            raise ValueError("Expected 30 option features.")

        state_embedding = self.state_encoder(state)
        option_embeddings = self.option_encoder(options)

        repeated_state = state_embedding.unsqueeze(0).expand(
            options.shape[0], -1
        )

        combined = torch.cat(
            [repeated_state, option_embeddings], dim=1
        )

        return self.score_head(combined).squeeze(-1)

    def score_options(
        self,
        state: FeatureVector,
        options: list[FeatureVector],
    ) -> list[float]:
        """Return finite scores in legal-option order."""
        if not options:
            return []

        device = next(self.parameters()).device

        state_tensor = torch.as_tensor(
            list(state), dtype=torch.float32, device=device
        )
        option_tensor = torch.as_tensor(
            options, dtype=torch.float32, device=device
        )

        self.eval()

        with torch.no_grad():
            scores = self.forward(
                state_tensor, option_tensor
            )

        result = scores.cpu().tolist()

        if not all(math.isfinite(score) for score in result):
            raise ValueError("Model produced non-finite scores.")

        return result