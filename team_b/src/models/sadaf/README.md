# Sadaf Fatima Syeda (Team B, GROUP_PROGRAMMING_09_b)

**Primary responsibility:** Team B evaluation, experiment tracking and submission evidence.

**Status (25 Nov 2026):** both models trained, evaluated locally and passed the Group 9 common
evaluation; Kaggle packages built and smoke-tested; official uploads pending the team's decision
on active slots (Team B's two active slots hold B-SR2 and the heuristic-v1 baseline).

| ID | Architecture | vs heuristic-v1 (100 BO3) | vs random (100 BO3) | ms / decision |
| --- | --- | --- | --- | --- |
| B-SA1 | Decision Tree regressor (depth 8, 102 leaves) | 56–44, 56.0% (95% CI 46.2–65.3%) | 98–2 | 0.06 |
| B-SA2 | Random Forest regressor (100 trees, depth 8) | 45–55, 45.0% (95% CI 35.6–54.8%) | 99–1 | 1.06 |

Neither model beats the baseline (both intervals include 50%), and the forest is no stronger
than the single tree while ~19× slower. Full analysis: [FINDINGS.md](../../../experiments/sadaf/FINDINGS.md).

### Group 9 common evaluation (24 Nov 2026)

Team notebook `team_b/notebooks/06_Common_Agent_Evaluation.ipynb`, fresh Colab CPU session per
agent, 40 BO3 matches against each of three reference policies. Both runs are complete and valid
(0 invalid responses, 0 engine errors) under the same protocol
(`b5071f21…`) and runtime fingerprint.

| ID | sample_random | first_options | last_options | Mean match score | ms / decision |
| --- | --- | --- | --- | --- | --- |
| B-SA1 | 38–2 (95% CI 83.5–98.6%) | 23–17 (42.2–71.5%) | 40–0 (91.2–100%) | 0.842 | 0.20–0.26 |
| B-SA2 | 39–1 (87.1–99.6%) | 24–16 (44.6–73.7%) | 40–0 (91.2–100%) | 0.858 | 0.92–1.57 |

The two models differ by two matches out of 120 and every interval overlaps, so they are tied;
`first_options` is the hardest reference for both. These are three simple controls, not the hosted
Kaggle ladder.

## Approach

- **Features** (`pipeline.py`, schema `sadaf-v1`): 68 state + 59 option = 127 numbers per legal
  option, using only information visible to the acting player.
- **Target:** Chelsi's `teacher_option_scores` (heuristic-v1 preferences), regressed with MSE.
  These are teacher preferences, not true action values.
- **Data:** 300 heuristic-vs-random games, every decision of both players, split **by whole game**
  70/15/15 (train/validation/test).
- **Agent:** the model ranks legal options; the number of selections follows heuristic-v1's rule;
  `safe_agent` replaces any illegal output (never triggered in evaluation).

## Files

| File | Purpose |
| --- | --- |
| `model_1.py` | B-SA1 Decision Tree (`score_options`, `score_rows`, `describe`, save/load) |
| `model_2.py` | B-SA2 Random Forest (same interface + feature importances) |
| `pipeline.py` | Features, data collection, split by game, top-1 agreement, game policy |
| `train.py` | Collect → fit → offline metrics → BO3 matches → evidence JSON (+ `--record`) |
| `portable.py` | Export trees to JSON and score them without scikit-learn |
| `kaggle_main.py` | Kaggle entry point; falls back to heuristic-v1 if the model can't load |
| `package.py` | Builds `team_b/outputs/B-SA1.tar.gz` / `B-SA2.tar.gz` via Chelsi's packager |
| `plot_results.py` | Comparison chart and table in `team_b/experiments/sadaf/figures/` |
| `plan.json` | Model status, experiment IDs and evidence paths |

Shared evaluation helpers live in `team_b/src/evaluation.py` (`write_experiment`,
`summarize_matches`, `outcomes_from_series`).

## Reproduce (from the repository root)

Needs the engine in `team_b/data/raw/` (see [Team B README](../../../README.md)) and
scikit-learn for training only.

```bash
python3 -m team_b.src.models.sadaf.train --model tree --games 300 --matches 100 --record
python3 -m team_b.src.models.sadaf.train --model forest --games 300 --matches 100 --record
python3 -m team_b.src.models.sadaf.plot_results
python3 -m team_b.src.models.sadaf.package --model tree      # or --model forest
python3 -m unittest discover -s team_b/tests -t .
```

The engine's shuffles are not seedable, so reruns differ by a few matches.

## Evidence

- Experiment records: `team_b/experiments/sadaf/B-20261122-132818-B-SA1.json`,
  `B-20261122-132923-B-SA2.json`; rows in `team_b/experiments/results.csv`.
- Chart and table: `team_b/experiments/sadaf/figures/`.
- Common evaluation summaries: `team_b/experiments/sadaf/common_eval/` (CSV and JSON per model;
  detailed private logs kept outside the repository).
- Pull requests: #3 (evaluation), #4 (B-SA1), #5 (B-SA2), #6 (records and findings).

## Remaining

- Upload B-SA1 and B-SA2 to Kaggle if the team frees active slots; record IDs and ratings in
  `team_b/submissions/register.csv`.
- Report section (evaluation and results), slides and peer review.
