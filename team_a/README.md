# Team A: GROUP_PROGRAMMING_09_a

The simulator, random baseline, heuristic and first learned model are integrated. A-R1 training is complete. The latest supplied Kaggle screenshot marks all three submitted agents Complete, displaying A-R1 at 828.8, heuristic at 257.6 and random at 11.5. These are rating snapshots. A-R2 remains pending.

| Member | Responsibility | Assigned models |
| --- | --- | --- |
| Rishitha Gogineni | Simulator, baseline, response adapter, packages and Team A uploads | A-R1, A-R2 |
| Kavya Ayyappan | State/action features and Team A evaluation | A-K1, A-K2 |
| Shashira Guntuka | Value architecture pair and conditional DQN experiment | A-S1, A-S2 |
| Jiayi Liang | Policies, rewards, outcome logging and combined slide assembly | A-J1, A-J2 |

## Executed notebooks

1. [Day 1 simulator](notebooks/01_simulator_Day1.ipynb): four completed battles and 79 observation examples.
2. [Day 2 random baseline](notebooks/02_baseline_Day2.ipynb): ten local battles and two successful archive tests.
3. [Day 3 heuristic](notebooks/03_heuristic_baseline.ipynb): 24 wins in 30 evaluation battles and 176 separate demonstration decisions.
4. [A-R1 linear scorer](notebooks/04_AR1_Linear_Scorer.ipynb): fresh collection, fitting, model export, evaluation and package checks.

Notebook explanations use simple lab notes. Code, execution counts, plot images and computational outputs are preserved. Colab upload and download display controls were removed. No training or simulator code was rerun during integration.

## A-R1 results

Run: 20261122T002608Z_bd91a79d. The UTC run ID falls on 22 November; the local project date is 21 November 2026 in America/Los_Angeles.

| Check | Recorded result |
| --- | --- |
| Fresh demonstration games | 300 completed |
| Collected decisions | 3,346 |
| Forced single option decisions excluded | 256 |
| Training data | 210 games, 2,161 decisions |
| Validation data | 45 games, 539 decisions |
| Held out test data | 45 games, 390 decisions |
| Feature schema | ar1-visible-v1, 375 candidate features |
| Selected model | L2 0.0, checkpoint at epoch 14 |
| Test agreement with any teacher maximum | 98.72 percent |
| Test agreement with exact teacher choice | 96.15 percent |
| Heuristic against random in this run | 35 wins, 5 losses in 40 battles |
| A-R1 against random | 40 wins in 40 battles |
| A-R1 against heuristic | 19 wins, 21 losses in 40 battles |
| Fresh archive checks | All four passed |
| Invalid responses and engine rejections | Zero across all phases |

A-R1 is a shared linear dot product over legal candidate features, with action type and visible state interactions. Normalization is fitted on training candidate rows only. The objective is cross entropy over the current legal options, with target probability divided equally among actions sharing the highest teacher score. Validation loss selects the checkpoint and regularization value. Test metrics are computed afterward.

The learned scorer handles MAIN selections requiring one option and offering at least two. Other decisions use the frozen heuristic. The battle results therefore describe the deployed combination. Policy usage counters include callback contract checks and are preserved without interpreting them as an exact gameplay fallback fraction.

The 19 of 40 result does not establish that A-R1 is stronger than its teacher. Higher imitation accuracy does not establish stronger play. Local results use one sample deck and individual battles. Approximate intervals do not adjust for dependence between games, and native engine randomness is not verified as seeded. The higher displayed Kaggle rating is evidence of that snapshot, not a fixed final score or a controlled comparison.

## Source, weights and data

* [Fitted A-R1 model folder](src/models/rishitha/ar1/) contains the exact tested entry point, model.json and feature definitions.
* [Repository scorer](src/models/rishitha/model_1.py) loads those fitted weights. Its complete candidate vectors already include state interactions; its usage is documented in [Rishitha's model README](src/models/rishitha/README.md).
* [Day 4 source and results](experiments/day4/) preserve the exported scripts, original reports, checkpoints, figures, collection records and split files.
* [Feature contract](experiments/day4/training/feature_contract.json), [dataset manifest](experiments/day4/training/dataset_manifest.json) and [training metrics](experiments/day4/training/metrics.json) define the reproducible experiment.
* [A-R1 Kaggle evidence](submissions/A-R1/) contains the screenshot, submission record and rating observation.

The A-R1 workflow is self contained. It does not require another member's files or review. The general team feature stub is unchanged. Other models can use the exported schema deliberately or define and record their own contract.

For A-R2, reuse the A-R1 feature definitions and the exact train, validation and test game groups for a meaningful comparison. Fit preprocessing on training data only. Preserve the final test groups and do not select checkpoints using test results. The earlier Day 3 dataset is separate and was not included in this run.

## Reproduce and package

Open the executed A-R1 notebook in a fresh Colab CPU session, upload the same official competition ZIP and run all cells. It uses preinstalled NumPy and Matplotlib for fitting and plots; the deployed entry point uses the Python standard library. The notebook checks the input hash and records the Python 3.13.18 Linux x86_64 runtime and source hashes. No package installation is performed.

The notebook collects 300 new games, fits three regularization settings, checks NumPy versus standalone scores, then plays 40 battles per comparison. It packages main.py, model.json and deck.csv only after all required checks pass and tests a fresh extraction against both opponents. The tested archive is 20,803 bytes. The competition ZIP, deck, SDK binaries and generated archive remain local. Selected results and fitted JSON weights are deliberately included in Git.

The exact deployed source and fitted artifact match Colab. The additional repository Model wrapper was checked statically only; no native runtime was executed locally. Generic team training and packaging CLI templates remain incomplete; use the notebook workflow.

## Submission history and next work

[Random history](submissions/A-BASE/rating_observations.csv): 600.0 at initialization, then 490.5, 103.0 and 11.5. [Heuristic history](submissions/A-HEUR-01/rating_observations.csv): 489.7, then 257.6. [A-R1 observation](submissions/A-R1/rating_observations.csv): 828.8. Historical screenshots are retained. Do not treat the initial rating as measured strength.

The latest image does not show submission IDs, absolute observation times, rating uncertainty or the actual team name. Missing fields remain blank. GROUP_PROGRAMMING_09_a is the intended team. Local package hashes are verified; screenshots cannot establish uploaded archive hashes. Original Colab outputs retain their not_submitted flags because manual uploads happened afterward.

Rishitha's next model is A-R2. Other members' actual progress is unchanged by this integration. See the [model registry](experiments/model_registry.csv), [experiment log](experiments/results.csv), [submission register](submissions/register.csv) and [contribution table](../docs/contributions.csv).

Earlier setup evidence remains under [Day 1](experiments/day1/), [Day 2](experiments/day2/) and [Day 3](experiments/day3/). The [79 Day 1 observation examples](experiments/day1/handoff_for_kavya/) are debugging material, not expert demonstrations or complete reinforcement learning trajectories.
