# Bhavyasree Kondi

**Primary responsibility:** Actor-critic architectures and conditional PPO experiment.

| ID | Planned architecture |
| --- | --- |
| B-B1 | Shared actor-critic backbone |
| B-B2 | Separate actor and critic networks |

**Objective:** Actor imitation warm start plus documented critic fitting; genuine on-policy PPO only if feasible.

## Tasks

- **Days 1-2:** Define masked candidate actor and critic targets; collect actual rollout returns for critic fitting where used.
- **Days 3-4:** Train both heads; B-B1 upload Day 4; cap optional PPO debugging and stop if loop is not correct.
- **Days 5-7:** B-B2 upload Day 5; distinguish warm-start from PPO results; compare deployed policy behavior; report/present.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
