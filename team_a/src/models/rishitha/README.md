# Rishitha Gogineni

A-R1 is trained, evaluated and submitted. A-R2 remains planned. Simulator, baseline, package and Team A evidence responsibilities continue.

| Model | Status |
| --- | --- |
| A-R1 linear contextual scorer | Trained; Kaggle screenshot shows Complete at 828.8 |
| A-R2 small MLP scorer | Pending |

## Fitted A-R1

[ar1/main.py](ar1/main.py) is byte identical to the tested archive entry point. [ar1/model.json](ar1/model.json) contains its fitted weights and training only normalization. [ar1/features.py](ar1/features.py) preserves the exact ar1-visible-v1 feature definitions. The deployed policy learns MAIN decisions with one required option and at least two candidates; it uses the frozen heuristic elsewhere. The official deck stays local.

[model_1.py](model_1.py) provides a repository loader and scorer. Model().score_observation(observation) handles a supported observation. Model().score_options([], candidates) accepts complete 375 element vectors from ar1.features.candidate_features(observation). The state argument is empty because state interactions are already included in every candidate vector. It rejects a nonempty state to avoid silently dropping it. The wrapper is statically checked only; use the exact ar1/main.py entry point or executed notebook for the Colab tested deployed policy.

This implementation is self contained and does not depend on another member's review or the general feature stub. It does not handle unsupported selections through the scorer interface; the deployed agent handles those through its heuristic fallback.

## Evidence

The [executed notebook](../../../notebooks/04_AR1_Linear_Scorer.ipynb), [source and results](../../../experiments/day4/), [feature contract](../../../experiments/day4/training/feature_contract.json), [dataset manifest](../../../experiments/day4/training/dataset_manifest.json) and [Kaggle screenshot](../../../submissions/A-R1/) preserve the run.

Fresh collection used 300 games and saved 3,346 decisions. After removing 256 forced choices, game groups were split 210/45/45 and decision counts 2161/539/390. Targets are uniform over highest scoring teacher actions. Three regularization settings were tried; validation selected L2 0.0 and epoch 14. Held out teacher top set agreement was 98.72 percent. New battle results were 40/40 against random and 19/40 against heuristic. All four archive tests passed with zero invalid responses or engine rejections.

## Next model

Build A-R2 using the same feature schema and saved game splits. Keep the target, loss, preprocessing and evaluation controls matched. Record its actual results and any ineffective approach. Preserve the A-R1 comparison rather than claiming that teacher imitation established stronger play. The planned day numbers in plan.json are the original schedule, not completion timestamps.
