# Jiayi Liang

**Primary responsibility:** Policies, rewards, outcome logging and combined slide assembly.

| ID | Planned architecture |
| --- | --- |
| A-J1 | Linear contextual masked policy |
| A-J2 | Small MLP masked policy |

**Objective:** Matched behavior-cloning training first; optional correctly implemented policy-gradient refinement.

## Tasks

- **Days 1-2:** Own outcome/reward logs, fallback fixtures and demonstration collection.
- **Days 3-4:** Fit both policies; A-J1 upload Day 4; optional small reward pilot after both work.
- **Days 5-7:** A-J2 upload Day 5; analyze coverage/reward results; assemble slides; present.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
