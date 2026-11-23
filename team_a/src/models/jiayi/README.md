# A-J1 and A-J2 policy training

Jiayi trained and evaluated two policies offline using Team A's Day 4 heuristic demonstrations. Both score legal candidates in their original order using the shared `ar1-visible-v1` schema. Each raw candidate has 375 features, including state interactions, so the separate `state` argument is `[]`.

| Configuration | Model artifact | Validation cross entropy | Held-out cross entropy | Held-out exact-choice accuracy |
| --- | --- | ---: | ---: | ---: |
| A-J1 linear chosen-action BC | [aj1_bc.json](aj1_bc.json) | 0.681347 | 0.678851 | 97.18% |
| A-J2 32-unit tanh MLP chosen-action BC | [aj2_bc.json](aj2_bc.json) | 0.648977 | 0.652052 | 98.46% |
| A-J1 linear outcome-weighted BC, β = 0.5 | [aj1_weighted_bc.json](aj1_weighted_bc.json) | 0.681863 | 0.676058 | 96.92% |

The MLP had the highest held-out imitation accuracy. Outcome weighting slightly reduced held-out cross entropy for the linear model, while validation cross entropy rose and held-out accuracy fell by one decision. These metrics measure imitation on recorded decisions. Gameplay evaluation is pending.

## Reproduce training

Open the [executed training notebook](../../../notebooks/AJ1_AJ2_Policy_Training.ipynb) to review the training outputs, comparison table, and validation learning curves. For a new run, use a Python kernel with NumPy, pandas, and Matplotlib, started from the repository root or `team_a/notebooks/`. For the recorded CPU setup, set `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` before starting the kernel.

The notebook calls [train.py](train.py) and uses seed 42 with the existing 210/45/45 game splits. Scaling is fitted on training candidates, and checkpoints are selected by validation cross entropy before scoring the held-out split. The notebook writes the three JSON artifacts and [results.csv](results.csv) to `team_a/src/models/jiayi/` when executed. The CSV includes training, validation, and held-out metrics.

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

Use `score_options([], options)` for logits and `policy_probabilities([], options)` for probabilities over supplied legal candidates. The `score_observation` and `probabilities_observation` helpers use the shared encoder, which supports MAIN decisions with exactly one required choice and at least two candidates. Standalone agent integration and fallback handling are pending.

Run the inference and training tests from the repository root.

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest team_a.src.models.jiayi.test_inference team_a.src.models.jiayi.test_training -v
```
