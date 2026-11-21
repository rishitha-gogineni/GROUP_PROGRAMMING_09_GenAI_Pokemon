# Team A - GROUP_PROGRAMMING_09_a

Day 1 simulator setup has been run successfully in Colab. The reusable source modules and model files remain starter placeholders; a Kaggle-ready baseline has not been validated or submitted yet.

| Member | Responsibility | Planned models |
| --- | --- | --- |
| Rishitha Gogineni | Simulator, baseline, response adapter, packages and Team A uploads | A-R1, A-R2 |
| Kavya Ayyappan | State/action features and Team A evaluation | A-K1, A-K2 |
| Shashira Guntuka | Value-style architecture pair and conditional DQN experiment | A-S1, A-S2 |
| Jiayi Liang | Policies, rewards, outcome logging and combined slide assembly | A-J1, A-J2 |

## Day 1 results - 21 November 2026

Rishitha ran the official sample SDK in a Colab CPU session. The inspection, initial random-versus-random battle and all three smoke tests passed without recorded Python or cleanup errors.

| Battle | Python action-sampling seed | Selections | Recorded time (s) | Saved observation examples |
| --- | ---: | ---: | ---: | ---: |
| Initial battle | 42 | 34 | 0.017 | 17 |
| Smoke test | 10 | 31 | 0.017 | 14 |
| Smoke test | 20 | 119 | 0.069 | 26 |
| Smoke test | 30 | 74 | 0.042 | 22 |

The four battles saved 79 distinct observation examples across setup, early, middle and later turn bands. These selected examples and their handoff notes are available in [Kavya's handoff folder](experiments/day1/handoff_for_kavya/). The top-level `representative_sample_count` in `day1_status.json` is 17 because that field refers to the initial battle only. Receipt of the handoff by Kavya is not yet confirmed.

These are individual battles with the sample deck, not best-of-three results or official Kaggle scores. Timings include observation collection and file writes; four short tests do not establish training throughput or playing strength. Only Python action sampling is seeded; engine randomness is not verified as seeded. Full decision-type coverage and submission-loader compatibility remain to be checked.

## Notebook and evidence

- [Executed Day 1 notebook](notebooks/01_simulator_Day1.ipynb)
- [Runtime, source archive and SDK file hashes](experiments/day1/environment.json)
- [Three smoke-test results](experiments/day1/smoke_test_results.csv)
- [All four battle results](experiments/day1/all_battle_results.json)
- [Day 1 completion status](experiments/day1/day1_status.json)
- [Observation examples and instructions for Kavya](experiments/day1/handoff_for_kavya/)

The notebook text was shortened after execution. Executable code, execution counts and computational outputs were preserved; transient upload/download widgets were removed.

## Reproduce the Day 1 checks

1. Open the notebook in a fresh Colab CPU session.
2. Download the competition ZIP using an account that has accepted the competition rules; keep it local.
3. Run cells in order and upload the ZIP when prompted. The recorded input filename and SHA-256 are in the environment record above.
4. Download the results ZIP and the executed notebook before closing the session.

The recorded run used Python 3.13.18 on Linux x86_64 with the bundled `libcg.so`. It called the SDK directly; `kaggle-environments` was not installed in that run. No extra packages were installed by the notebook. Different engine randomness can produce different outcomes, even with the same Python seeds.

## Next tasks

- Rishitha: confirm the observation handoff, implement `src/simulator.py`, build the baseline in `src/baseline.py`, then verify the actual callback, package and official baseline submission.
- Kavya: inspect the observation examples and define the state/action schema and controlled evaluation protocol.
- Shashira and Jiayi: use the agreed interfaces to develop their assigned architecture pairs and training objectives.
- Record model experiments and official submissions only when they actually occur. The model registry still lists planned models, and the submission register has no official IDs or ratings yet.

## Data, configuration and scripts

Use `data/README.md` and `configs/data_contract.template.json` to define sources, visibility, targets and grouped splits. `configs/environment.json` remains an unverified integration template; the actual Day 1 runtime is recorded in `experiments/day1/environment.json`. Pin dependencies only after the reusable runtime is verified.

The `scripts/` train/evaluate/package entry points and `agent_package/main.py` are placeholders and must not be uploaded as a working agent. Exact commands for the reusable pipeline will be added when implemented.

## Folder purposes

- `notebooks/`: executed notebooks and their saved outputs.
- `src/`: adapters, encoders, baseline, simulator and member models.
- `configs/`: integration templates and planned data/evaluation settings.
- `scripts/`: future training, evaluation and packaging entry points.
- `tests/`: acceptance checks to implement against actual observations.
- `data/`: local data and provenance instructions.
- `artifacts/`, `outputs/`: ignored local checkpoints/results.
- `experiments/`: model registry, measured records and Day 1 evidence.
- `submissions/`: official-attempt metadata; archives remain local.
- `agent_package/`: future self-contained runtime package.

The 79 selected observation examples are included in the Day 1 handoff. Original competition downloads, SDK binaries, the deck file, complete local observation logs and archives remain outside Git.
