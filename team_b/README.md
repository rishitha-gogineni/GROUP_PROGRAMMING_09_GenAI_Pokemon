# Team B - GROUP_PROGRAMMING_09_b

**Status:** local simulator, best-of-three runner, card data, heuristic baseline (`heuristic-v1`), evaluation script, Kaggle packager and tests work locally. No trained models yet; no Kaggle submission recorded yet.

| Member | Responsibility | Planned models |
| --- | --- | --- |
| Chelsi Shulamit Elthuri | Team B simulator, card data, baseline, packages and uploads | B-C1, B-C2 |
| Sadaf Fatima Syeda | Team B evaluation, experiment tracking and report coordination | B-SA1, B-SA2 |
| Bhavyasree Kondi | Actor-critic architectures and conditional PPO experiment | B-B1, B-B2 |
| Srinidhi Jaya Revanth Srirangarajapally | Replay/demonstration provenance, imitation, splits and citations | B-SR1, B-SR2 |

## Day 1 handoffs

Chelsi owns simulator/integration and uploads; Sadaf owns evaluation. Reproduce the official example, complete a game and supply real observations. Record verified versions in `configs/environment.json`; then fill `requirements.txt` with compatible tested versions. Do not assume SDK keys or return types from these team-defined interfaces.

## Data, models and evaluation

Use `data/README.md`, `configs/data_contract.template.json` and each model-folder README. Begin with supported single-choice contexts and a verified full-response fallback elsewhere. Every trained artifact needs a schema/data version and reproducible config.

`experiments/results.csv` is empty by design. `experiments/model_registry.csv` lists planned candidates, not completed models. `submissions/register.csv` reserves attempts but contains no invented scores or submission IDs.

## Setup (once per machine)

The engine is competition material: keep it in `data/raw/` (git-ignored) and never commit it.

1. From the Kaggle data page, copy `sample_submission/cg/` to `team_b/data/raw/cg/`,
   `sample_submission/deck.csv` to `team_b/data/raw/deck_sample.csv`, and
   `EN_Card_Data_R2_full.csv` to `team_b/data/raw/`.
2. Linux/Colab: nothing else; `cg/sim.py` loads the official `libcg.so`.
   macOS: there is no official `libcg.dylib`; compile it once from the engine source
   on the data page (takes about 3-4 minutes):

   ```
   cd "<data>/ptcg_engine_playground/ptcgProgram 22"
   clang++ -std=c++20 -O2 -shared -fPIC -fvisibility=hidden -o <repo>/team_b/data/raw/cg/libcg.dylib Export.cpp
   ```
3. Python 3.10+ standard library only; no `pip install` needed. Alternatively set
   `PTCG_CG_DIR` to a folder containing `cg/`.

## Reproduction commands (run from the repository root)

```
python3 -m unittest discover -s team_b/tests -t . -v                      # acceptance tests
python3 -m team_b.scripts.evaluate --agent heuristic --opponent random --matches 200 --seed 7
python3 -m team_b.scripts.evaluate --agent heuristic --opponent first  --matches 200 --seed 7 --record
python3 -m team_b.scripts.package_agent --games 10                      # builds team_b/outputs/submission.tar.gz
```

Policies available to `evaluate`: `heuristic` (baseline), `random`, `first`.

## Verified engine facts (local)

- `cg.game.battle_start(deck0, deck1)` plays **one game**; `current.round` stays 1.
  `simulator.run_match` therefore plays best-of-three itself (first to 2 game wins, seats alternate).
- `current.result`: -1 running, 0/1 winner index, 2 draw. The decision maker is `current.yourIndex`.
- Engine shuffles/coin flips are not seedable from Python; `--seed` only fixes agent randomness.
- The card CSV has one row per (card, attack): 2,273 rows, 1,431 unique IDs, identical to the
  engine's 1,431 cards. Three names differ only in spelling (e.g. "Lugia ex" vs "LugiaEX").
- With the sample deck (35 energy, 6 Basic Pokemon) most games end by "no Pokemon in the
  Active Spot", so bench development dominates; deck choice is a separate Day 6 experiment.

## Team B code map (for teammates)

| Module | Use it for |
| --- | --- |
| `src/engine.py` | `engine.game()` / `engine.api()`: import the official engine |
| `src/adapter.py` | `validation_error`, `safe_agent(policy, deck)`, `read_deck`, `top_k` |
| `src/card_features.py` | `load_card_table()`, `card_vector(info)`, `deck_problems(deck)` |
| `src/baseline.py` | `heuristic_response`, `teacher_option_scores` (teacher targets), `random_response` |
| `src/simulator.py` | `play_game`, `run_match` (BO3), `run_series`; `recorder=` hook to log decisions |

A policy is any function `obs_dict -> list[int]`. Pass `recorder=fn(obs, response)` to
`play_game`/`run_series` to collect demonstrations or transitions for training.

## Folder purposes

- `src/`: adapter, features, baseline, simulator and member models.
- `configs/`: verified environment, data contract and evaluation settings.
- `scripts/`: `evaluate.py` and `package_agent.py` work; `train.py` is still a placeholder.
- `tests/`: acceptance checks against the real engine (`unittest`).
- `data/`: local data and provenance instructions.
- `artifacts/`, `outputs/`: ignored local checkpoints/results.
- `agent_package/`: submission `main.py`; build the upload with `scripts/package_agent.py`.
- `submissions/`: official attempt metadata; archives ignored.
