# Team A: GROUP_PROGRAMMING_09_a

The simulator, random baseline, heuristic, A-R1 linear scorer and A-R2 MLP are integrated. Both of Rishitha's assigned model architectures are trained and locally evaluated. A-R2's manual Kaggle upload is reported by the submitter; its hosted status and rating are not yet evidenced. The last supplied screenshot shows A-R1 Complete at 828.8, heuristic at 257.6 and random at 11.5. These remain rating snapshots, not live scores.

## Common Colab evaluation

Use [06_Common_Agent_Evaluation.ipynb](notebooks/06_Common_Agent_Evaluation.ipynb) for the common comparison between both teams. This copy is preset to `GROUP_PROGRAMMING_09_a`. The agent label comes from your uploaded archive filename. No code edits are needed.

1. Download the notebook from GitHub and open it in Google Colab.
2. Use a fresh CPU session and select **Runtime > Run all**.
3. At the first upload prompt, upload `the-pokemon-company-ptcg-ai-battle-challenge-playground.zip` from the official competition download used for this project.
4. At the second prompt, upload your own agent submission `.tar.gz`. It must contain `main.py`, `deck.csv` and all required assets at the archive root. Use the official sample deck for this benchmark.
5. Let all comparisons finish. Download `comparison_summary.csv`, `comparison_summary.json` and the private results ZIP. Also save the executed notebook using **File > Download > Download .ipynb**.

Keep all evaluation settings unchanged. Each agent faces the same three simple reference policies for 40 matches each, with at most three games per match. Compare only completed runs with the same protocol fingerprint, and review runtime differences. Earlier individual game results and other evaluation commands use different protocols.

Share the comparison CSV and JSON as permitted by the course and competition rules. Keep your agent archive, model files and detailed private logs within your Kaggle team. The notebook does not submit to Kaggle. Incomplete runs are flagged and receive no overall comparison score. Native engine randomness is not fully controlled, and these simple controls do not establish an overall best agent.

Preparation was checked with standard library tests and a synthetic simulator. The notebook is unexecuted; run it in Colab to obtain real evaluation results.

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
4. [A-R1 linear scorer](notebooks/04_AR1_Linear_Scorer.ipynb): fresh collection, linear fitting and submission checks.
5. [A-R2 small MLP](notebooks/05_AR2_MLP_Scorer.ipynb): matched data, MLP fitting, paired model analysis and archive checks.

Notebook prose uses simple lab notes. Code, execution counts, figures and computational outputs are preserved. Colab browser controls were removed from the saved display. The saved notebook computations were not rerun. Shared-tool tests exercised standard-library inference on three saved observations and temporary synthetic SDK fixtures; no native engine or training was run.

## Matched model comparison

The A-R2 run is 20261122T052210Z_0c2ec591 and reuses parent A-R1 run 20261122T002608Z_bd91a79d. Both were executed on local project date 21 November 2026 in America/Los_Angeles; UTC run IDs fall on 22 November.

Both models use the same 375 ar1-visible-v1 candidate features, acting player perspective, teacher labels and training only normalization. The game groups are unchanged: 210 training games with 2,161 decisions, 45 validation games with 539 decisions and 45 test games with 390 decisions. A-R1 originally collected 3,346 decisions from 300 games and excluded 256 forced choices. The earlier Day 3 games were not reused.

| Item | A-R1 | A-R2 |
| --- | --- | --- |
| Architecture | Shared linear candidate scorer | One hidden layer, 32 tanh units |
| Parameter count, excluding normalization | 375 | 12,064 |
| Selected L2 | 0.0 | 0.0001 |
| Selected checkpoint | Epoch 14 | Epoch 60 |
| Learning rate | 0.03 | 0.003 |
| Test cross entropy | 0.71579 | 0.67886 |
| Test agreement with any teacher maximum | 98.72 percent | 99.49 percent |
| Exact teacher choice agreement | 96.15 percent | 95.13 percent |

The objective is cross entropy over each current legal candidate set, distributing the target equally among teacher maximum score options. Both runs use Adam, batch size 64, seed 42, a maximum of 60 epochs, patience 10 and the same three L2 values. Learning rates differ as recorded. Validation selects checkpoints; test results are calculated afterward. The MLP selected the last checkpoint in its fixed budget, not a demonstrated convergence point. Input weight rows with no training variation stay zero.

The test groups were already inspected for A-R1. They support the planned architecture comparison, not a new unseen benchmark. Ties in the teacher scores allow top-set agreement to improve while exact choice agreement falls. These are descriptive decision metrics; decisions within each game are dependent.

## New battle results from the A-R2 run

| Candidate | Opponent | Wins | Losses |
| --- | --- | ---: | ---: |
| Heuristic | Random | 37 | 3 |
| A-R1 | Random | 37 | 3 |
| A-R1 | Heuristic | 25 | 15 |
| A-R2 | Random | 36 | 4 |
| A-R2 | Heuristic | 15 | 25 |
| A-R2 | A-R1 | 17 | 23 |

All 240 evaluation battles and six archive tests passed without invalid responses or engine rejections. Gradient checks and NumPy versus exported policy checks also passed. The archive tests are execution checks and are not added to the evaluation win rates. The earlier A-R1 run recorded 40/40 against random and 19/40 against heuristic; those original results remain separate under day4/.

The MLP reduced test cross entropy and improved teacher top-set agreement, but did not establish stronger gameplay in these samples. This is an ineffective improvement attempt to discuss honestly in the report. It is not evidence that the training or package failed. The small battle samples and approximate intervals do not establish a conclusive ranking. Native randomness is not verified as seeded, and matching Python seeds does not produce controlled identical native games.

Both deployed agents use the learned scorer only for MAIN selections requiring one option and offering at least two. Other selections use the frozen heuristic. Battle results therefore describe the combined deployed policies. Usage counters include callback contract checks; no exact gameplay fallback fraction is inferred.

## Source, weights and evidence

* [A-R1 model](src/models/rishitha/ar1/) and [A-R2 model](src/models/rishitha/ar2/) contain the exact tested entry points, fitted JSON weights and feature definitions.
* [Repository scorers](src/models/rishitha/README.md) document the model_1.py and model_2.py interfaces. The wrappers passed feature-interface checks on three saved observations; the deployed entry points still match Colab byte for byte.
* [Day 4 records](experiments/day4/) preserve A-R1 collection, source, fitting and original evaluation.
* [Day 5 records](experiments/day5/) preserve the MLP source, checkpoints, gradient checks, comparison CSVs, figures, battle logs and original parent data.
* [Shared game manifest](experiments/day5/training/dataset_manifest.json), [feature contract](experiments/day5/training/feature_contract.json) and [model metrics](experiments/day5/training/metrics.json) define the matched comparison.
* [A-R1 Kaggle evidence](submissions/A-R1/) has the supplied screenshot. [A-R2 upload record](submissions/A-R2/) records the submitter's report, with hosted status and rating pending evidence.

These workflows are self contained and require no other member's files or review. The shared feature entry point delegates to the frozen encoder. The accepted random entry point, baseline adapter and existing model code and weights remain unchanged. Other members' actual progress is not changed by this integration.

## Reproduce and package

Run the executed notebooks in fresh Colab CPU sessions. A-R1 needs the official competition ZIP. A-R2 needs that ZIP and AR1_20261122T002608Z_bd91a79d_outputs.zip; it reuses the saved data rather than collecting or splitting again. Input hashes and source hashes are recorded. The saved runtime is Python 3.13.18, NumPy 2.4.6 and Matplotlib 3.10.0 on Linux x86_64. No package installation is performed.

A-R2 checks gradients using training examples, trains three regularization settings, verifies the exported policy and runs the six comparisons above. It packages main.py, model.json and deck.csv only after all required checks pass, then tests a fresh extraction against three opponents. The tested A-R2 archive is 75,537 bytes. Local checks establish execution, not hosted acceptance or strength across other decks.

Official competition inputs, SDK binaries, decks and generated archives remain local. Selected experiment outputs and fitted JSON weights are deliberately included in Git. Shared evaluation and packaging commands are implemented; only the generic train.py remains a placeholder. The new commands were checked with synthetic fixtures, not a new native SDK run.

## Submission history and remaining work

[Random history](submissions/A-BASE/rating_observations.csv): initialization 600.0, then 490.5, 103.0 and 11.5. [Heuristic history](submissions/A-HEUR-01/rating_observations.csv): 489.7, then 257.6. [A-R1 observation](submissions/A-R1/rating_observations.csv): 828.8. No A-R2 rating is recorded yet. These histories have not been updated from an unobserved live leaderboard.

Missing IDs, absolute timestamps, uncertainty and actual Kaggle team name remain blank. GROUP_PROGRAMMING_09_a is the intended team. Original Colab not_submitted flags were saved before manual uploads and remain unchanged. Local hashes identify tested files; a user report or screenshot cannot verify uploaded archive bytes.

Rishitha's two planned models are trained and evaluated. Remaining work includes A-R2 hosted result evidence, the final paired analysis, assigned report sections and presentation preparation. See the [model registry](experiments/model_registry.csv), [experiment log](experiments/results.csv), [submission register](submissions/register.csv) and [contribution table](../docs/contributions.csv).

Earlier records remain under [Day 1](experiments/day1/), [Day 2](experiments/day2/) and [Day 3](experiments/day3/). The 79 Day 1 observations are debugging examples, not expert demonstrations or complete reinforcement learning trajectories.

## Shared Team A tools

[src/features.py](src/features.py) provides the tested encoder's public entry point and explicit supported context. [scripts/evaluate.py](scripts/evaluate.py) compares configured or custom standalone agents. [scripts/package_agent.py](scripts/package_agent.py) checks a fresh archive before publishing a ready submission. [configs/evaluation.json](configs/evaluation.json) records individual battles, opponents, deck/source hashes, seed schedules, draw handling and native RNG limits. [requirements.txt](requirements.txt) pins the recorded NumPy and Matplotlib training versions.

Use the [shared command instructions](scripts/README.md) for exact examples. Eleven standard-library tests passed, including both fitted model interfaces and temporary packaging/evaluation fixtures. No package was installed and no native engine or training was run. These tests validate shared-tool plumbing, not new Pokemon playing strength. Earlier models, executed notebooks and historical results are unchanged.
