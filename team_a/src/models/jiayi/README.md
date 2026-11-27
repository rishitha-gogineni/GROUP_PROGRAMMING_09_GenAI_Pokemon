# A-J1 and A-J2 policy training and evaluation

Jiayi trained two policies using Team A's Day 4 heuristic demonstrations. A-J1 is a linear weighted sum of candidate features. A-J2 uses a 32-unit tanh hidden layer and a scalar output for each candidate. Both score legal candidates in their original order using the shared `ar1-visible-v1` schema. Each raw candidate has 375 features, including state interactions, so the separate `state` argument is `[]`. The work was merged in [PR #8](https://github.com/rishitha-gogineni/GROUP_PROGRAMMING_09_GenAI_Pokemon/pull/8).

| Configuration | Model artifact | Validation cross entropy | Held-out cross entropy | Held-out exact-choice accuracy |
| --- | --- | ---: | ---: | ---: |
| A-J1 linear chosen-action BC | [aj1_bc.json](aj1_bc.json) | 0.681347 | 0.678851 | 97.18% |
| A-J2 32-unit tanh MLP chosen-action BC | [aj2_bc.json](aj2_bc.json) | 0.648977 | 0.652052 | 98.46% |
| A-J1 linear outcome-weighted BC, β = 0.5 | [aj1_weighted_bc.json](aj1_weighted_bc.json) | 0.681863 | 0.676058 | 96.92% |

The MLP had the highest held-out imitation accuracy. Outcome weighting lowered held-out cross entropy from 0.678851 to 0.676058, raised validation cross entropy from 0.681347 to 0.681863, and reduced held-out exact agreement by one decision (379/390 to 378/390). These metrics measure imitation on recorded decisions. The weighted variant was an offline comparison.

## Reproduce training

Open the [executed training notebook](../../../notebooks/AJ1_AJ2_Policy_Training.ipynb) to review the training outputs, comparison table, and validation learning curves. For a new run, use a Python kernel with NumPy, pandas, and Matplotlib, started from the repository root or `team_a/notebooks/`. For the recorded CPU setup, set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` before starting the kernel.

The notebook calls [train.py](train.py) and uses seed 42 with the existing 210/45/45 game splits. Scaling is fitted on training candidates, and checkpoints are selected by validation cross entropy before scoring the held-out split. The notebook writes the three JSON artifacts and [results.csv](results.csv) to `team_a/src/models/jiayi/` when executed. The CSV includes training, validation, and held-out metrics.

All three runs used Adam, batch size 64, and 60 epochs. The two linear runs used learning rate 0.03 and selected epochs 48 (unweighted) and 39 (weighted). The MLP used 0.003 and selected epoch 54. The unweighted models use chosen-action cross entropy. The weighted run scales that loss by the recorded terminal outcome for the acting player (`β = 0.5`).

## Load a trained policy

From the repository root, use the existing loaders directly. Inference uses the standard library and the repository's shared feature encoder.

```python
from pathlib import Path
from team_a.src.models.jiayi.model_1 import Model as LinearPolicy
from team_a.src.models.jiayi.model_2 import Model as MLPPolicy

models = Path("team_a/src/models/jiayi")
linear = LinearPolicy(models / "aj1_bc.json")
weighted_linear = LinearPolicy(models / "aj1_weighted_bc.json")
mlp = MLPPolicy(models / "aj2_bc.json")
```

Use `score_options([], options)` for logits and `policy_probabilities([], options)` for probabilities over supplied legal candidates. The `score_observation` and `probabilities_observation` helpers use the shared encoder, which supports MAIN decisions with exactly one required choice and at least two candidates. The callable agents use deterministic highest-score selection for supported decisions and Team A’s heuristic for unsupported decisions.

Run the inference and training tests from the repository root.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest team_a.src.models.jiayi.test_inference team_a.src.models.jiayi.test_training -v
```

## Rewards and self-play

[self_play.py](self_play.py) assigns terminal rewards from the acting player's seat (+1 for a win, −1 for a loss, and 0 for a draw). The shaping diagnostic adds the change in normalized visible prize lead scaled by 0.25, with zero terminal potential. The [executed self-play notebook](../../../notebooks/AJ1_AJ2_Policy_Self_Play.ipynb) uses seeds 61000–61003 and 10% uniform legal exploration mixed with policy sampling. It completed four battles per architecture with zero invalid responses or engine rejections. Both reward schemes were applied to the same trajectories, so the experiment measures how the reward signal changes without testing the effect of shaping on trained playing strength.

Run the self-play notebook from the repository in a Linux environment with `PTCG_SDK_DIR` pointing to the official SDK and sample deck.

## Evaluate callable agents in Colab

Open [AJ1_AJ2_Policy_Evaluation.ipynb](../../../notebooks/AJ1_AJ2_Policy_Evaluation.ipynb) in a Linux Colab runtime with the repository and official SDK available. Set `REPO_DIR` and `SDK_DIR` in its first code cell (or the `PTCG_REPO_DIR` and `PTCG_SDK_DIR` environment variables). The SDK directory must contain `cg/` and the official `deck.csv`. Run all cells and save the executed notebook. Native evaluation is intentionally blocked on macOS.

The notebook reuses Team A's evaluator for 40 individual battles per policy against the fixed heuristic, with seeds 60000–60039, alternating player seats, and the same deck and limits. A-J1 won 19 and lost 21 (47.5%). A-J2 won 20 and lost 20 (50.0%). Both completed all battles with zero draws, invalid responses, or engine rejections. The [battle CSV](evaluation_results.csv) records outcomes, errors, and timings. Win rates use valid completed battles. Python seeds provide a consistent schedule, but native-engine randomness was not verified as controllable.

[agents.py](agents.py) stages [aj1/main.py](aj1/main.py) or [aj2/main.py](aj2/main.py) with its existing fitted weights as `model.json` and the supplied `deck.csv`. The entry points support the evaluator's `__raw_path__` contract without `__file__` or a particular working directory.

For direct integration, call `stage_policy("A-J1", destination, deck_file)` (or `"A-J2"`) from `team_a.src.models.jiayi.agents`, then pass the staged directory to the existing evaluator. The destination must not already exist. Shared-runner bookkeeping remains temporary in the notebook; no new integrity bookkeeping is implemented here.

Run non-native integration tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest team_a.src.models.jiayi.test_agents team_a.src.models.jiayi.test_inference team_a.src.models.jiayi.test_training -v
```

## Build standalone agent archives

[package_submissions.py](package_submissions.py) builds separate A-J1 and A-J2 archives with the official sample deck. Each archive contains a top-level `main.py`, `deck.csv`, the selected fitted weights as `model.json`, and only the inference modules needed by the policy, frozen feature encoder, and heuristic fallback.

```bash
python -m team_a.src.models.jiayi.package_submissions \
  --deck-file /path/to/official/sample_submission/deck.csv \
  --output-dir /path/to/output
```

The outputs are `A-J1_Linear_Policy.tar.gz` and `A-J2_MLP_Policy.tar.gz`. They run without a repository checkout and satisfy the Common Agent Evaluation callback contract: `agent` is the final callable, `__file__` is not required, the fixed deck is returned initially, and gameplay responses are legal option indices. Run the focused packaging test locally without the native SDK:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest team_a.src.models.jiayi.test_packaging -v
```

## Common best-of-three evaluation

Both archives completed the same common benchmark in Colab under matching protocol and runtime settings. Each played 40 best-of-three matches per reference opponent.

| Policy | Sample random wins | First options wins | Last options wins | Overall W–L–D | Macro match score |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-J1 | 34/40 | 21/40 | 40/40 | 95–25–0 | 79.17% |
| A-J2 | 40/40 | 23/40 | 40/40 | 103–17–0 | 85.83% |

Both runs reported zero invalid responses and zero engine rejections. The original local Colab summaries have run IDs `20261124T035857Z_5f9c3b29` and `20261124T040209Z_b4e403e7`. Per-opponent results are in the [shared experiment log](../../../experiments/results.csv). The scores describe these reference opponents and do not establish a statistically significant improvement or predict hosted ratings. Both agents rely on the existing heuristic outside the supported MAIN decision scope.

To reproduce the benchmark, run [06_Common_Agent_Evaluation.ipynb](../../../notebooks/06_Common_Agent_Evaluation.ipynb) in a fresh Colab CPU session once per archive. Save each run's `comparison_summary.csv` and `comparison_summary.json` under distinct policy names.

## Kaggle submission status

Jiayi confirms that both A-J1 and A-J2 were officially uploaded. Submission IDs, timestamps, hosted validation status, Skill Ratings, and official evidence links were unavailable for independent verification and are blank in the [submission register](../../../submissions/register.csv). The Colab results above are local evaluation results.
