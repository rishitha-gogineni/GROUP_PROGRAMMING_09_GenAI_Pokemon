# Rishitha's executed notebooks

Open each notebook in a fresh Colab CPU session to reproduce it. Download official inputs from Kaggle; they are not included in GitHub.

| Notebook | Input files | Completed output |
| --- | --- | --- |
| [01 Simulator](01_simulator_Day1.ipynb) | Official competition ZIP | Environment checks, four battles and observation examples |
| [02 Random baseline](02_baseline_Day2.ipynb) | Official competition ZIP | Random agent, local checks and package |
| [03 Heuristic](03_heuristic_baseline.ipynb) | Official competition ZIP | Heuristic, demonstrations, evaluation and package |
| [04 A-R1 linear scorer](04_AR1_Linear_Scorer.ipynb) | Official competition ZIP | Fresh demonstrations, training, model checks and package |
| [05 A-R2 MLP scorer](05_AR2_MLP_Scorer.ipynb) | Official ZIP and `AR1_20261122T002608Z_bd91a79d_outputs.zip` | Matched training, comparisons and package |
| [06 A-R1 common evaluation](06_Common_Evaluation_A-R1.ipynb) | Official ZIP and original A-R1 submission `.tar.gz` | 120 completed matches; 83.33% match score |
| [06 A-R2 common evaluation](06_Common_Evaluation_A-R2.ipynb) | Official ZIP and original A-R2 submission `.tar.gz` | 120 completed matches; 85.00% match score |
| [07 Energy attachment pilot](07_Energy_Attachment_Experiment.ipynb) | Official competition ZIP; original reference policies are embedded | 13 checks and 240 individual games; no established improvement |

Use the [shared common template](../06_Common_Agent_Evaluation.ipynb) for new candidates rather than editing the completed A-R1/A-R2 records. [Experiment index](../../experiments/rishitha/) and [model code](../../src/models/rishitha/).

Notebook code, execution counts and computational outputs are preserved. The heuristic notebook's explanatory repository path was updated after the folder move. No computation was rerun during organization.
