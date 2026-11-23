# Sadaf (Team B): B-SA1 Decision Tree vs B-SA2 Random Forest: findings

Owner: Sadaf Fatima Syeda · Kaggle team GROUP_PROGRAMMING_09_b · measured 22 Nov 2026 (local engine)

## 1. What was built

| Piece | File | Notes |
|---|---|---|
| Evaluation helpers | `team_b/src/evaluation.py` | `write_experiment`, `summarize_matches` (Wilson 95% CI), `outcomes_from_series` (W/L/D/E) |
| Features | `team_b/src/models/sadaf/pipeline.py` | schema `sadaf-v1`: 68 state + 59 option = 127 numbers per legal option; acting-player-visible information only |
| B-SA1 | `model_1.py` | `DecisionTreeRegressor(max_depth=8, min_samples_leaf=5)` |
| B-SA2 | `model_2.py` | `RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=5)` |
| Training / evaluation | `train.py` | collect → split by game → fit → offline metrics → BO3 matches → evidence |
| Kaggle export | `portable.py`, `kaggle_main.py`, `package.py` | sklearn-free JSON trees; falls back to heuristic-v1 if the model fails to load |

**Target.** Chelsi's `teacher_option_scores` (heuristic-v1 preferences), regressed per legal option (MSE).
These are teacher preferences, **not** true action values.

**Data.** 300 heuristic-vs-random games; every decision of both players; split **by whole game**
70/15/15 (train/val/test) so turns from one game never appear in two splits.

**Agent.** The model ranks the legal options; how many to pick follows heuristic-v1's rule.
`safe_agent` replaces any illegal output with a legal one (never needed: 0 fallbacks).

## 2. Results (same protocol for both models)

Protocol: sample deck on both sides, alternating seats, 100 best-of-three matches per opponent,
engine shuffles not seedable (results vary slightly between runs). Win rate excludes aborted series.

| | B-SA1 Decision Tree | B-SA2 Random Forest |
|---|---|---|
| vs heuristic-v1 baseline | **56–44** → 56.0% (95% CI 46.2–65.3%) | **45–55** → 45.0% (95% CI 35.6–54.8%) |
| vs random | 98–2 → 98.0% (93.0–99.5%) | 99–1 → 99.0% (94.6–99.8%) |
| Illegal moves / fallbacks | 0 | 0 |
| Mean decision time | **0.06 ms** | 1.06 ms (~19× slower) |
| Fit time | 0.1 s | 4.0 s |
| Test MSE / top-1 agreement with teacher | 14.94 / 93.2% | 12.32 / 92.3% |
| Size | depth 8, 102 leaves | 100 trees, depth 8 |

Evidence: `B-20261122-132818-B-SA1.json`, `B-20261122-132923-B-SA2.json`, chart and table in `figures/`.

## 3. Failed / ineffective approaches and why

**(a) Neither model beats the baseline.** Both 95% intervals include 50%, so the honest claim is
"no significant difference from heuristic-v1", not "better". This is expected: the models were
trained to **copy** heuristic-v1's scores, so the teacher is a ceiling. Imitating a teacher with
93% agreement reproduces its play, including its mistakes. Improving on it needs a target that
contains information the teacher lacks (e.g. game outcomes / returns), not a better copy.

**(b) The Random Forest was not better than the single tree.** 100 trees gave a slightly
lower test MSE but no improvement in games (45% vs 56%, overlapping intervals) while being
~19× slower per decision and ~30× slower to train. Averaging many trees reduces variance in the
**score**, but what matters is the **ranking** of a handful of options; the single depth-8 tree
already ranks them like the teacher 93% of the time, so the extra capacity had little to fix.

**(c) Lower offline error ≠ more wins.** The forest had the lower test MSE yet the lower win rate.
This confirms handbook 9.3: offline error is diagnostic only; full-game win rate is the measure.

## 4. What the models learned (Random Forest feature importance)

Top features (share of total importance): **energy still needed by the target Pokémon** (0.52),
**option is "play a card"** (0.21), **option is "attack"** (0.10), selection context "attach from"
(0.03). The model mainly learned heuristic-v1's turn-order logic: develop the board and attach
energy where it is needed, then attack. Board details (HP, prizes) mattered much less.

## 5. Limitations

- One deck (the official sample deck) on both sides; deck effects not tested here.
- Engine RNG is not seedable, so reruns differ by a few matches; 100 matches gives a ±~10-point CI.
- Teacher = heuristic-v1 only; no outcome-based labels were tried.
- Official Kaggle Skill Ratings pending upload (team decision on active slots).

## 6. What we would try next

1. Outcome-based targets: label visited options with the game result (return-to-go) and
   compare against teacher imitation under the same protocol.
2. Hybrid agent: heuristic-v1 for safety-critical choices, tree for the rest.
3. Keep the single tree as the deployable model: equal strength, ~19× faster, easy to explain.
