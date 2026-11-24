# Team B - GROUP_PROGRAMMING_09_b

**Status:** local simulator, best-of-three runner, card data, heuristic baseline (`heuristic-v1`), evaluation script, Kaggle packager and tests work locally. No trained models yet; no Kaggle submission recorded yet.

## Common Colab evaluation

Use [06_Common_Agent_Evaluation.ipynb](notebooks/06_Common_Agent_Evaluation.ipynb) for the common comparison between both teams. This copy is preset to `GROUP_PROGRAMMING_09_b`. The agent label comes from your uploaded archive filename. No code edits are needed.

1. Download the notebook from GitHub and open it in Google Colab.
2. Use a fresh CPU session and select **Runtime > Run all**.
3. At the first upload prompt, upload `the-pokemon-company-ptcg-ai-battle-challenge-playground.zip` from the official competition download used for this project.
4. At the second prompt, upload your own agent submission `.tar.gz`. It must contain `main.py`, `deck.csv` and all required assets at the archive root. Use the official sample deck for this benchmark.
5. Let all comparisons finish. Download `comparison_summary.csv`, `comparison_summary.json` and the private results ZIP. Also save the executed notebook using **File > Download > Download .ipynb**.

Keep all evaluation settings unchanged. Each agent faces the same three simple reference policies for 40 matches each, with at most three games per match. Compare only completed runs with the same protocol fingerprint, and review runtime differences. Earlier individual game results and other evaluation commands use different protocols.

Share the comparison CSV and JSON as permitted by the course and competition rules. Keep your agent archive, model files and detailed private logs within your Kaggle team. The notebook does not submit to Kaggle. Incomplete runs are flagged and receive no overall comparison score. Native engine randomness is not fully controlled, and these simple controls do not establish an overall best agent.

Preparation was checked with standard library tests and a synthetic simulator. The notebook is unexecuted; run it in Colab to obtain real evaluation results.

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
