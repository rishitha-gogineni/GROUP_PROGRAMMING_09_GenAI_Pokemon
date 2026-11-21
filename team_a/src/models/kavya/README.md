# Kavya Ayyappan

**Primary responsibility:** State/action features and Team A evaluation.

| ID | Planned architecture |
| --- | --- |
| A-K1 | Concatenated state-option MLP |
| A-K2 | Separate state and option encoders with joint score head |

**Objective:** Matched imitation objective/data for both candidate scorers.

## Tasks

- **Days 1-2:** Schema v1, encoders, grouped split and evaluation harness with Rishitha.
- **Days 3-4:** Fit both models; A-K1 upload Day 3; second model ready Day 4.
- **Days 5-7:** A-K2 upload Day 5; finalist comparison, figures, feature methods and presentation.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
