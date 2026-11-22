# Team A: GROUP_PROGRAMMING_09_a

The Day 1 simulator setup, Day 2 random baseline and Day 3 heuristic workflow are integrated. All three notebooks contain the executed Colab results. The heuristic won 24 of 30 local evaluation battles against random. The latest supplied Kaggle screenshot marks both submissions Complete and shows the heuristic at 489.7 and random at 103.0. These are rating snapshots, not fixed scores.

| Member | Responsibility | Planned models |
| --- | --- | --- |
| Rishitha Gogineni | Simulator, baseline, response adapter, packages and Team A uploads | A-R1, A-R2 |
| Kavya Ayyappan | State/action features and Team A evaluation | A-K1, A-K2 |
| Shashira Guntuka | Value architecture pair and conditional DQN experiment | A-S1, A-S2 |
| Jiayi Liang | Policies, rewards, outcome logging and combined slide assembly | A-J1, A-J2 |

## Completed notebooks

1. [Day 1 simulator](notebooks/01_simulator_Day1.ipynb): four completed battles and 79 observation examples for debugging and feature design.
2. [Day 2 random baseline](notebooks/02_baseline_Day2.ipynb): ten callback battles and two tests from the extracted archive; zero invalid responses or engine rejections.
3. [Day 3 heuristic](notebooks/03_heuristic_baseline.ipynb): separate heuristic comparison, demonstration collection and archive checks.

Notebook prose uses simple lab notes without decorative symbols or emojis. Functional code, filenames, identifiers, execution counts and computational outputs are preserved. Colab upload and download controls were removed from the saved display. The underlying calculations were not rerun during this presentation edit.

## Day 3 results

| Check | Recorded result |
| --- | --- |
| Heuristic evaluation against frozen random | 24 wins, 6 losses, 0 draws in 30 individual battles |
| Observed local win rate | 80 percent |
| Approximate Wilson 95 percent interval | 62.7 to 90.5 percent |
| Invalid responses and engine rejections | 0 and 0 across all three phases |
| Separate demonstration collection | 20 completed games and 176 decisions |
| Training split | 14 games, 127 decisions |
| Validation split | 3 games, 28 decisions |
| Held out split | 3 games, 21 decisions |
| Fresh archive checks | 2 of 2 passed |
| Archive contents | main.py and deck.csv; 12,504 bytes |
| Kaggle status in latest screenshot | Complete for both agents |
| Displayed heuristic and random ratings | 489.7 and 103.0 |

The interval is approximate and does not account for shared decks, opponent dependence or uncontrolled native randomness. Only one random opponent and the sample deck were used locally. Individual battles do not reproduce the full hosted Kaggle match harness. Collection and archive games are separate from the 30 evaluation games and must not be added to its reported win rate.

The heuristic is a handcrafted policy, not a learned architecture. A-R1 and A-R2 remain pending. The 176 examples are enough to test a feature and training pipeline; collect more games before drawing conclusions about a learned model.

## Source and results

The accepted random entry point, baseline wrapper, adapter and Day 2 simulator are unchanged.

* [src/baseline.py](src/baseline.py) retains the tested random policy and baseline_response interface. Random teacher scores remain undefined.
* [src/simulator.py](src/simulator.py) is the Day 2 individual battle runner.
* [src/heuristic.py](src/heuristic.py) is the exact Day 3 policy, including the card metadata loaded during the run.
* [src/heuristic_runner.py](src/heuristic_runner.py) is the exact Day 3 runner with an optional observation hook. It remains separate from the Day 2 runner.
* [Day 3 source exports and results](experiments/day3/) preserve the Colab reports, per battle CSV files, logs, manifests and source bytes.
* [Demonstration manifest](experiments/day3/dataset_manifest.json) defines the game groups. [Decision records](experiments/day3/demonstrations/records/) contain only the teacher's accepted MAIN decisions requiring one option.
* [Kaggle evidence](submissions/A-HEUR-01/) contains the supplied screenshot, submission record and rating observation.

## Use the demonstration data

Kavya should start from the manifest and define the versioned state and legal option feature contract. Use current.yourIndex for the acting player. Preserve the groups in the manifest and fit preprocessing on training games only. Use validation games for tuning and reserve the held out games for the final check. The evaluation games are excluded from these groups.

Only the heuristic player's supported decisions are labelled. The scores express handcrafted preferences, not optimal action values. These rows do not contain full reinforcement learning transitions. Shashira and Jiayi should agree with Kavya on features, rewards, actor perspective and required trajectory fields before implementing their models. More collection is needed for stronger training and evaluation claims.

Rishitha can build the linear scorer and small MLP once the feature contract is agreed. Preserve a legal response policy for unsupported contexts and evaluate the deployed agent, not just imitation accuracy.

## Reproduce the Colab workflows

Open the appropriate notebook in a fresh Colab CPU session and upload the official competition ZIP when prompted. Run all cells, download the output ZIP, and save the executed notebook. No GPU, Kaggle credentials or package installation is required by these notebooks. The recorded runtime was Python 3.13.18 on Linux x86_64 with the SDK loaded directly.

Day 3 runs evaluation, then separate demonstrations, then two tests of a freshly extracted submission archive. It includes submission.tar.gz in the download only after all required checks pass. The archive and official SDK binaries stay local; create a package with the notebook when needed. The Git source exports are sufficient to inspect the policy, but the official input ZIP is still required to reproduce it. Generic training and packaging scripts remain unfinished; use the notebook workflow.

## Kaggle evidence and rating history

[Random baseline evidence](submissions/A-BASE/) records the sequence 600.0 at initialization, then 490.5, then 103.0. [Heuristic evidence](submissions/A-HEUR-01/) records 489.7 in the latest supplied image, alongside random at 103.0. Preserve these observations rather than replacing historical images. Do not treat the initial rating as measured playing strength.

The screenshots do not expose submission IDs, absolute observation times, uncertainty or the actual Kaggle team name. Missing fields remain blank. GROUP_PROGRAMMING_09_a is the intended team name. Local package hashes were verified; the images cannot establish the uploaded archive hashes. Original Colab JSON and CSV outputs keep their preupload not_submitted flags because the manual uploads happened afterward.

See the [submission register](submissions/register.csv), [measured experiment log](experiments/results.csv) and [contribution table](../docs/contributions.csv). Keep Team B results and other members' actual work separate from Rishitha's recorded completion.

## Day 1 and Day 2 handoff

[Day 1 records](experiments/day1/) include 79 saved observation examples: 17 initial, 14 from seed 10, 26 from seed 20 and 22 from seed 30. The [handoff folder](experiments/day1/handoff_for_kavya/) contains these debugging examples. They are random actions, not expert demonstrations or complete training trajectories. Kavya's receipt has not been independently confirmed.

[Day 2 records](experiments/day2/) contain callback tests, ten completed local battles, two successful archive checks and the original package manifest. Repository aliases and import paths were checked statically after integration. No native SDK execution was performed locally during either integration.
