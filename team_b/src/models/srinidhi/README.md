# Srinidhi Jaya Revanth Srirangarajapally

**Primary responsibility:** Replay/demonstration provenance, imitation, splits and citations.

| ID | Planned architecture |
| --- | --- |
| B-SR1 | Linear contextual behavior-cloning policy |
| B-SR2 | Small MLP behavior-cloning policy |

**Objective:** Masked cross-entropy on demonstrator selections, held out by complete game/match.

## Tasks

- **Days 1-2:** Verify replay access Day 1; use own-team demonstrations if unavailable by Day 2 morning; visibility audit and split.
- **Days 3-4:** Fit pair; B-SR1 upload Day 4; evaluate held-out imitation and game results separately.
- **Days 5-7:** B-SR2 upload Day 5; coverage/teacher limits; methods, citation checks and presentation.

## Handoff and evidence

Implement both modules behind the team's candidate-scoring contract. Specify supported contexts, targets, loss, train/validation/held-out match groups, normalization and compute budget. Keep a valid full-response adapter for other decisions.

For linear models include state-option interaction features when needed; an additive state-only constant cannot change the option ranking. For actor-critic/value models name the actual training objective: supervised warm starts are not automatically PPO/DQN. Compare the deployed policies as well as fitting metrics.

Update the model registry, measured experiment log, official submission register and actual contributions. Supply architecture diagrams/configs, artifacts and reproducible commands. These files are placeholders; no implementation, training or performance is claimed.
