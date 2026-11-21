# Team A - GROUP_PROGRAMMING_09_a

The Day 2 random baseline and individual-battle runner are integrated. The saved Colab run passed all 12 battles and callback checks. Kaggle screenshots show **Complete**, initially **600.0**, then **490.5** in the later snapshot. Learned models, feature encoders and training remain in progress.

| Member | Responsibility | Planned models |
| --- | --- | --- |
| Rishitha Gogineni | Simulator, baseline, response adapter, packages and Team A uploads | A-R1, A-R2 |
| Kavya Ayyappan | State/action features and Team A evaluation | A-K1, A-K2 |
| Shashira Guntuka | Value-style architecture pair and conditional DQN experiment | A-S1, A-S2 |
| Jiayi Liang | Policies, rewards, outcome logging and combined slide assembly | A-J1, A-J2 |

## Day 2 baseline - 21 November 2026

| Check | Recorded result |
| --- | --- |
| Local callback-based battles | 10/10 passed |
| Battles from a fresh archive extraction | 2/2 passed |
| Callback contracts | Initial deck, single, multiple, optional and zero choices passed |
| Invalid responses / engine rejections | 0 / 0 |
| Tested archive | 892 bytes; main.py and deck.csv at the root |
| Kaggle status | Complete in both screenshots |
| Initialization rating | 600.0 |
| Latest observed rating | 490.5 |

The 600.0 value is the initialization snapshot; it is not evidence of competitive improvement. Both images show the same named submission. The screenshots do not expose its ID, exact timestamp, uncertainty or actual team name. Missing fields remain blank. The later score is a normal ladder update, not a validation error.

- [Executed Day 2 notebook](notebooks/02_baseline_Day2.ipynb)
- [Local checks and package manifest](experiments/day2/)
- [Kaggle screenshots, record and rating history](submissions/A-BASE/)
- [Submission register](submissions/register.csv)

Original notebook JSON/CSV status files were saved before the manual upload and retain their not_submitted flags. The later screenshot record provides the subsequent Kaggle result. Code, execution counts and computational notebook outputs were preserved when the explanatory text was shortened.

## Integrated code

- `src/baseline.py`: the tested random agent and deck loader, plus a repository-facing baseline_response alias. The teacher_option_scores function remains explicitly undefined because random actions do not provide heuristic-value targets.
- `src/adapter.py`: a thin callback wrapper delegating to the baseline.
- `src/simulator.py`: the tested run_battle/check_action implementation for individual battles, including cleanup and error recording.
- `agent_package/main.py`: byte-identical standalone entry point from the tested package. It needs the local official deck.csv; it has no model/SDK runtime dependency.
- `scripts/check_baseline.py`: portable checker with repository import paths. See [script instructions](scripts/README.md).

The core runner and submitted policy match the Colab-tested source. Repository aliases and checker import-path changes were statically checked; no native simulator was rerun during integration. The runner does not implement BO3 control, complete replay/transition collection or reinforcement learning.

## Reproduce the baseline

1. Open the executed Day 2 notebook in a fresh Colab CPU session.
2. Upload the official competition ZIP when prompted; the exact input hash is recorded in experiments/day2/environment.json.
3. Run all cells. The notebook checks the callback, plays ten battles, verifies the archive and plays two more battles from its extraction.
4. Download the results ZIP and executed notebook. Use its submission.tar.gz only after all local checks pass; coordinate any new upload with Team A.

The recorded runtime was Python 3.13.18 on Linux x86_64. The SDK was loaded directly; kaggle-environments was not installed. The submission uses standard-library imports only. Python action sampling is seeded; native-engine randomness is not verified as seeded. These tests establish execution checks, not playing strength, full selection coverage or a BO3 benchmark.

For a local Linux check, use the command in scripts/README.md with an unpacked agent folder and the official SDK root. The deck file and competition binaries stay local. Generic train.py, evaluate.py and package_agent.py still require implementation; packaging currently uses the notebook.

## Day 1 setup and handoff

The initial battle and three smoke tests passed. The four battles saved 79 examples: 17 initial, 14 with seed 10, 26 with seed 20 and 22 with seed 30. These cover setup, early, middle and later turn bands. Only the initial 17 appear in the top-level Day 1 sample count.

- [Executed Day 1 notebook](notebooks/01_simulator_Day1.ipynb)
- [Day 1 runtime and results](experiments/day1/)
- [Kavya's 79 observation examples and reading instructions](experiments/day1/handoff_for_kavya/)

The examples are random-action debugging/feature material, not expert demonstrations or complete training trajectories. Use current.yourIndex for the acting player. Kavya's receipt has not been independently confirmed.

## Next work

- Rishitha: retain official submission evidence, capture the missing ID/time if available, and begin A-R1/A-R2 after agreeing on features and training targets. The accepted random baseline does not count as either learned architecture.
- Kavya: define the versioned state/option encoding interface and controlled evaluation protocol.
- Shashira and Jiayi: build against that interface; agree on rewards, transition fields and acting-player perspective before RL experiments. The current runner reports outcomes but does not yet collect a training trajectory.

Use data/README.md and configs/data_contract.template.json for data preparation. configs/environment.json records the actual Day 2 runtime; detailed manifests live under experiments/day2/. Record later experiments and submissions only when performed.

Competition archives, simulator binaries, the deck file and generated package remain local. The 79 selected Day 1 observations are included in their handoff folder. Team B code and other members' contribution entries are unchanged by this integration.
