# Chelsi Shulamit Elthuri

**Primary responsibility:** Team B simulator, card data, baseline, packages and uploads.

| ID | Planned architecture |
| --- | --- |
| B-C1 | Linear contextual card-action scorer |
| B-C2 | Small card-action MLP |

**Objective:** Regress documented own-team heuristic scores with matched data and targets.

## Tasks

- **Days 1-2:** Runtime/baseline/response handler; validate card grain and joins; baseline upload.
- **Days 3-4:** Fit scorer pair; B-C1 upload Day 3 and B-C2 Day 4.
- **Days 5-7:** Repair packages; evidence; card/baseline methods, visuals and presentation.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
