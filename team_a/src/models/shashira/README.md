# Shashira Guntuka

**Primary responsibility:** Value-style architecture pair and conditional DQN experiment.

| ID | Planned architecture |
| --- | --- |
| A-S1 | Ordinary candidate-value network |
| A-S2 | Dueling-style state value and legal-candidate advantage network |

**Objective:** First regress own-team heuristic scores; verified TD/DQN extension only if feasible.

## Tasks

- **Days 1-2:** Define candidate tensors and verified reward/transition perspective with Jiayi.
- **Days 3-4:** Fit warm starts; A-S1 upload Day 4; cut unstable optional RL by Day 4.
- **Days 5-7:** A-S2 upload Day 5; compare matched models; label supervised warm starts accurately; report/present.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
