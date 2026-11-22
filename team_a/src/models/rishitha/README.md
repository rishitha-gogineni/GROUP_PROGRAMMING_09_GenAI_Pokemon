# Rishitha Gogineni

Both assigned architectures are trained and locally evaluated. A-R1 has Kaggle screenshot evidence. A-R2's manual upload is user-reported; official status and rating remain unverified. Assigned report and presentation work remains.

| Model | Architecture | Status |
| --- | --- | --- |
| A-R1 | Linear contextual scorer | Trained; supplied Kaggle image shows Complete at 828.8 |
| A-R2 | MLP with 32 tanh hidden units | Trained; upload reported, hosted evidence pending |

## Fitted files and interfaces

[ar1/](ar1/) and [ar2/](ar2/) contain main.py, model.json and features.py. Each main.py and model.json matches its Colab-tested package exactly. Both use ar1-visible-v1 and the same saved normalization. The official deck stays local.

[model_1.py](model_1.py) and [model_2.py](model_2.py) provide loaders and scorers. Model().score_observation(observation) accepts a supported MAIN observation. Model().score_options([], candidates) accepts complete 375-element vectors from the corresponding features.candidate_features(observation). The state argument must be empty because every candidate vector already contains its state interactions. Scores remain in the current legal option order.

Both wrappers passed the shared feature-interface checks on three saved observations using standard-library inference. The exact deployed main.py files were tested in Colab. The scorer interface rejects unsupported decisions; the full deployed agents use the frozen heuristic for those selections. Neither model depends on another member's files, review or the general feature stub.

## Reproducible comparison

A-R1 collection provided 300 games and 3,346 decisions. Removing 256 forced choices left 3,090 decisions grouped into 210/45/45 train, validation and test games with 2161/539/390 decisions. A-R2 reused these exact groups, features, normalization and targets. The objective distributes target probability equally among teacher maximum score legal options. Checkpoints are selected using validation loss.

A-R1 selected L2 0.0 and epoch 14. A-R2 selected L2 0.0001 and epoch 60 within a 60-epoch budget. Test teacher top-set agreement improved from 98.72 to 99.49 percent, while exact choice agreement fell from 96.15 to 95.13 percent. A-R2 won 36/40 against random, 15/40 against heuristic and 17/40 against A-R1 in the new evaluation. It did not establish a gameplay improvement. Both agents include the same heuristic fallback.

The same test groups were used for the planned architecture comparison and are not a new unseen benchmark. Models have different learning rates, as recorded. Fixed seeds do not establish controlled native engine randomness. Keep the original day4 and new day5 battle results distinct.

## Evidence and remaining tasks

[A-R1 notebook](../../../notebooks/04_AR1_Linear_Scorer.ipynb), [A-R2 notebook](../../../notebooks/05_AR2_MLP_Scorer.ipynb), [A-R1 records](../../../experiments/day4/), [A-R2 records](../../../experiments/day5/), [shared manifest](../../../experiments/day5/training/dataset_manifest.json), [A-R1 Kaggle evidence](../../../submissions/A-R1/) and [A-R2 upload record](../../../submissions/A-R2/) preserve the completed work.

Retain the weaker MLP comparisons as a lesson that improved imitation did not establish stronger play. Add official A-R2 acceptance and rating evidence when available. Complete the assigned report sections and slides, present the work and help with final group checks. Planned day numbers in plan.json are the original schedule, not completion timestamps.
