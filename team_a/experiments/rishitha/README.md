# Rishitha Gogineni: experiment records

Start here for the completed work. Detailed logs are supporting evidence; the README in each experiment explains its purpose and limitations.

| Experiment | What to read first | Executed notebook |
| --- | --- | --- |
| Simulator setup | [Status and observation handoff](simulator/README.md) | [Simulator](../../notebooks/rishitha/01_simulator_Day1.ipynb) |
| Random baseline | [Baseline checks](random_baseline/) | [Random agent](../../notebooks/rishitha/02_baseline_Day2.ipynb) |
| Heuristic baseline | [Evaluation and demonstration data](heuristic/) | [Heuristic](../../notebooks/rishitha/03_heuristic_baseline.ipynb) |
| A-R1 linear scorer | [Training, dataset and original battle results](ar1_linear/) | [A-R1](../../notebooks/rishitha/04_AR1_Linear_Scorer.ipynb) |
| A-R2 MLP scorer | [Matched model comparison](ar2_mlp/) | [A-R2](../../notebooks/rishitha/05_AR2_MLP_Scorer.ipynb) |
| Common evaluation | [83.33% versus 85.00%, with limitations](common_evaluation/) | [A-R1](../../notebooks/rishitha/06_Common_Evaluation_A-R1.ipynb), [A-R2](../../notebooks/rishitha/06_Common_Evaluation_A-R2.ipynb) |
| Energy attachment pilot | [Ineffective rule experiment](energy_attachment/) | [Energy rule](../../notebooks/rishitha/07_Energy_Attachment_Experiment.ipynb) |

## Shared inputs used by teammates

[ar1_linear/training/](ar1_linear/training/) is the canonical A-R1 dataset used by the A-R1/A-R2 comparison and by teammates' existing training code. It contains train, validation and held-out JSONL files, game groups and the feature contract. [ar1_linear/collection/](ar1_linear/collection/) retains all original game records and outcomes needed by Jiayi's audit. Do not remove these files as clutter.

The duplicate `reference_ar1` copy formerly under the A-R2 run was removed only after every file matched its canonical A-R1 counterpart byte for byte. The original export remains in the local backup; the [cleanup manifest](cleanup_manifest.json) lists retained replacements. A-R2's notebook still accepts the original A-R1 output ZIP in Colab.

## Methods and evidence

[Model methods and deployed weights](../../src/models/rishitha/) explain features, targets, training splits and fallback behavior. [Kaggle evidence](../../submissions/rishitha/) records official snapshots separately from local benchmarks. [Detailed review notes](REVIEW_NOTES.md) preserve the previous Team A writeup for this work.

Reports, measured outcome files and source snapshots preserve their original recorded contents and runtime paths. The cleanup removes empty logs, duplicate parent files and superseded exported README text; it does not alter model weights, raw decisions, battle outcomes or notebook calculations. No simulator or training run is performed by this reorganization.
