"""A-K1: Concatenated state-option MLP."""

import math
import torch
from torch import nn
from ...contracts import FeatureVector

MODEL_ID = "A-K1"
ARCHITECTURE = "Concatenated state-option MLP"

STATE_DIM = 21
OPTION_DIM = 30
INPUT_DIM = STATE_DIM + OPTION_DIM


class Model(nn.Module):
    """MLP that scores each legal action."""

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
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

        repeated_state = state.unsqueeze(0).expand(
            options.shape[0], -1
        )

        combined = torch.cat(
            [repeated_state, options], dim=1
        )

        return self.network(combined).squeeze(-1)

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
