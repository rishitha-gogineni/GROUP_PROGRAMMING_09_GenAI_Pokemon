# A-R1 and A-R2: detailed comparison

Owner: Rishitha Gogineni. These are the recorded training and individual-battle results. They use different evaluation units from the later common benchmark.

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

All 240 evaluation battles and six archive tests passed without invalid responses or engine rejections. Gradient checks and NumPy versus exported policy checks also passed. The archive tests are execution checks and are not added to the evaluation win rates. The earlier A-R1 run recorded 40/40 against random and 19/40 against heuristic; those original results remain separate under the A-R1 experiment.

The MLP reduced test cross entropy and improved teacher top-set agreement, but did not establish stronger gameplay in these samples. This is an ineffective improvement attempt to discuss honestly in the report. It is not evidence that the training or package failed. The small battle samples and approximate intervals do not establish a conclusive ranking. Native randomness is not verified as seeded, and matching Python seeds does not produce controlled identical native games.

Both deployed agents use the learned scorer only for MAIN selections requiring one option and offering at least two. Other selections use the frozen heuristic. Battle results therefore describe the combined deployed policies. Usage counters include callback contract checks; no exact gameplay fallback fraction is inferred.

## Common evaluation and energy pilot

The later [common evaluation](common_evaluation/) scored A-R1 at 83.33% and A-R2 at 85.00%, over 120 matches each. All 240 matches passed. The two-match difference against simple controls does not establish an overall winner. The [energy attachment pilot](energy_attachment/) did not establish a reliable improvement and was not adopted.

## Reproduce and inspect

Use the [executed notebook index](../../notebooks/rishitha/) for exact Colab inputs. A-R2 needs the original official ZIP and A-R1 output ZIP. The recorded Colab runtime was Python 3.13.18, NumPy 2.4.6 and Matplotlib 3.10.0 on Linux x86_64. The deployed agents use standard-library inference.

[A-R1 source and weights](../../src/models/rishitha/ar1/), [A-R2 source and weights](../../src/models/rishitha/ar2/), [canonical demonstration data](ar1_linear/training/), [A-R1 original records](ar1_linear/), [A-R2 original records](ar2_mlp/) and [Kaggle evidence](../../submissions/rishitha/).

All original source, fitted parameters, data splits, collection outcomes and battle results remain available. The duplicate parent copy and empty logs removed during organization are listed in [cleanup_manifest.json](cleanup_manifest.json). Original exported run flags describe their preupload state.
