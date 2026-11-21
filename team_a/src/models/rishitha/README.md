# Rishitha Gogineni

**Primary responsibility:** Simulator, baseline, response adapter, packages and Team A uploads.

| ID | Planned architecture |
| --- | --- |
| A-R1 | Linear contextual imitation scorer |
| A-R2 | Small MLP imitation scorer |

**Objective:** Masked candidate-set imitation cross-entropy on own-team demonstrations.

## Tasks

- **Days 1-2:** Run SDK example; full response handler; baseline upload; hand real observations to Kavya.
- **Days 3-4:** Fit matched scorer pair; submit A-R1 Day 3 and A-R2 Day 4.
- **Days 5-7:** Repair packages; preserve evidence; write setup/methods and present.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
