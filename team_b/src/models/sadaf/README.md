# Sadaf Fatima Syeda

**Primary responsibility:** Team B evaluation, experiment tracking and report coordination.

| ID | Planned architecture |
| --- | --- |
| B-SA1 | Decision Tree regressor |
| B-SA2 | Random Forest regressor |

**Objective:** Regress same teacher-score targets on same state-option feature data.

## Tasks

- **Days 1-2:** Evaluation harness, grouped split, tracking, report outline and section owners.
- **Days 3-4:** Fit pair; B-SA1 upload Day 3; measure fit error, latency and game performance separately.
- **Days 5-7:** B-SA2 upload Day 5; finalists, figures, combined report, evidence checks and presentation.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
