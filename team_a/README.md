# Team A - GROUP_PROGRAMMING_09_a

**Starter scaffold; no trained or submission-ready agent yet.**

| Member | Responsibility | Planned models |
| --- | --- | --- |
| Rishitha Gogineni | Simulator, baseline, response adapter, packages and Team A uploads | A-R1, A-R2 |
| Kavya Ayyappan | State/action features and Team A evaluation | A-K1, A-K2 |
| Shashira Guntuka | Value-style architecture pair and conditional DQN experiment | A-S1, A-S2 |
| Jiayi Liang | Policies, rewards, outcome logging and combined slide assembly | A-J1, A-J2 |

## Day 1 handoffs

Rishitha owns simulator/integration and uploads; Kavya owns evaluation. Reproduce the official example, complete a game and supply real observations. Record verified versions in `configs/environment.json`; then fill `requirements.txt` with compatible tested versions. Do not assume SDK keys or return types from these team-defined interfaces.

## Data, models and evaluation

Use `data/README.md`, `configs/data_contract.template.json` and each model-folder README. Begin with supported single-choice contexts and a verified full-response fallback elsewhere. Every trained artifact needs a schema/data version and reproducible config.

`experiments/results.csv` is empty by design. `experiments/model_registry.csv` lists planned candidates, not completed models. `submissions/register.csv` reserves attempts but contains no invented scores or submission IDs.

## Reproduction commands

Pending SDK integration. The owner adds exact setup, baseline, train, evaluate and package commands after they actually work. Scripts and source placeholders currently fail explicitly; do not upload them as an agent.

## Folder purposes

- `src/`: adapter, features, baseline, simulator and member models.
- `configs/`: verified environment, data contract and evaluation settings.
- `scripts/`: future train/evaluate/package entry points.
- `tests/`: acceptance checks to implement against real observations.
- `data/`: local data and provenance instructions.
- `artifacts/`, `outputs/`: ignored local checkpoints/results.
- `agent_package/`: future self-contained runtime package; not upload-ready.
- `submissions/`: official attempt metadata; archives ignored.
