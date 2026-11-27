# Team A: GROUP_PROGRAMMING_09_a

Team A is part of Group 9's shared DATA 266 project. This page links to the code, executed notebooks, measured results and Kaggle evidence. Both Kaggle teams contribute to one final report and presentation.

## Members and models

| Member | Models | Code and model notes | Executed notebooks |
| --- | --- | --- | --- |
| Rishitha Gogineni | A-R1 linear scorer; A-R2 MLP scorer | [Models](src/models/rishitha/) | [Notebook index](notebooks/rishitha/) |
| Kavya Ayyappan | A-K1 concatenated MLP; A-K2 separate encoders | [Models](src/models/kavya/) | See member documentation |
| Shashira Guntuka | A-S1 and A-S2 value-style candidates | [Work folder](src/models/shashira/) | Add completed notebooks when available |
| Jiayi Liang | A-J1 linear policy; A-J2 MLP policy | [Models](src/models/jiayi/) | [Training](notebooks/jiayi/AJ1_AJ2_Policy_Training.ipynb), [heuristic evaluation](notebooks/jiayi/AJ1_AJ2_Policy_Evaluation.ipynb), [self-play](notebooks/jiayi/AJ1_AJ2_Policy_Self_Play.ipynb) |

## Common evaluation

These are completed Colab benchmark scores, not Kaggle skill ratings. Each model plays 40 externally controlled matches against each of three simple reference policies using the official sample deck. Wins score 1, draws 0.5 and losses 0. A match stops at two game wins or after three games; more wins at the cap takes the match, otherwise it is a draw.

| Model | Random wins | First-option wins | Last-option wins | Overall match score | Evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| A-R1 | 39/40 | 21/40 | 40/40 | 83.33% | [Executed notebook](notebooks/rishitha/06_Common_Evaluation_A-R1.ipynb) |
| A-R2 | 40/40 | 22/40 | 40/40 | 85.00% | [Executed notebook](notebooks/rishitha/06_Common_Evaluation_A-R2.ipynb) |
| A-K1 and A-K2 | | | | Summary not uploaded | Add completed common evaluation evidence |
| A-J1 | 34/40 | 21/40 | 40/40 | 79.17% | [Executed notebook](notebooks/jiayi/06_Common_Evaluation_A_J1.ipynb), [CSV](experiments/jiayi/A-J1/A_J1_comparison_summary.csv), [JSON](experiments/jiayi/A-J1/A_J1_comparison_summary.json) |
| A-J2 | 40/40 | 23/40 | 40/40 | 85.83% | [Executed notebook](notebooks/jiayi/06_Common_Evaluation_A_J2.ipynb), [CSV](experiments/jiayi/A-J2/A_J2_comparison_summary.csv), [JSON](experiments/jiayi/A-J2/A_J2_comparison_summary.json) |
| A-S1 and A-S2 | | | | Not verified | Add completed evidence when available |

The two A-R1 and A-R2 runs used matching protocol and runtime fingerprints. All 240 matches and 536 games completed with zero invalid actions or engine rejections. A-R2 won two more matches; this small difference against simple controls does not establish an overall winner. Native engine randomness is not fully controlled. [Detailed comparison and run records](experiments/rishitha/common_evaluation/).

To evaluate another candidate:

1. Download [06_Common_Agent_Evaluation.ipynb](notebooks/06_Common_Agent_Evaluation.ipynb) and open it in a fresh Colab CPU session.
2. Run all cells. Upload the original official competition ZIP first, then one model's standalone submission `.tar.gz`.
3. Keep the settings and official sample deck unchanged. Use a new session for the next model.
4. Save the executed notebook, comparison CSV and JSON, and output ZIP. Add a member folder under `experiments/` for the results.
5. Compare completed runs with matching protocol fingerprints and review runtime differences. The notebook makes no Kaggle submissions.

## Experiment evidence

* [Experiment index](experiments/): find results by member and purpose.
* [Rishitha's experiments](experiments/rishitha/): simulator, random and heuristic baselines, A-R1, A-R2, common evaluation and energy-rule pilot.
* [Kavya's saved checkpoints](experiments/kavya/) and [model explanation](src/models/kavya/).
* [Jiayi's common evaluation records](experiments/jiayi/) and [model methods](src/models/jiayi/).

The energy attachment pilot passed its execution checks but did not establish a reliable improvement and was not adopted. The earlier A-R2 experiment improved teacher imitation without establishing stronger gameplay. Both remain documented ineffective approaches; keep individual-game results separate from common match scores.

## Kaggle evidence

[Submission index and register](submissions/) separate official evidence from local evaluation. The supplied screenshot received on 27 November shows Rishitha's A-R1 Complete at 268.3, A-R2 at 242.6, heuristic at 251.9 and random at 11.5. Earlier observations, including A-R1 at 828.8, remain in the dated history. These are snapshots, not current ratings. Submission IDs and exact observation times are unavailable in the image.

The same supplied screenshot shows Complete entries for Kavya and Jiayi. Jiayi's evidence shows A-J1 at 271.7 and A-J2 at 313.5. [Jiayi's Kaggle screenshot](submissions/jiayi/A-J1-J2/Kaggle_AJ1_AJ2.png). A screenshot does not verify uploaded archive bytes. [Preserved Team A screenshot](submissions/rishitha/evidence/team_a_submissions_received_20261127.png).

## Reproduce the work

Use the [notebook index](notebooks/) for inputs and completed outputs. [Shared command instructions](scripts/) explain feature encoding, individual-battle evaluation and packaging. The reusable common notebook is the comparison protocol above; `scripts/evaluate.py` uses individual battles and produces different metrics.

[Shared features](src/features.py) expose the frozen `ar1-visible-v1` encoder: 375 candidate features for MAIN selections requiring exactly one choice and offering at least two options. Other contexts need a valid fallback. Each member retains the encoder used to train their own models. [Feature contract](configs/feature_schema.json), [runtime](configs/environment.json), [individual-battle settings](configs/evaluation.json) and [training dependencies](requirements.txt).

Official input ZIPs, SDK binaries, decks and generated submission archives stay local. Original run reports preserve their recorded runtime paths and filenames. [Cleanup record](experiments/rishitha/cleanup_manifest.json) maps relocated paths and removed duplicate files to retained evidence.

## Remaining group work

Complete pending models and common comparisons, reconcile member registers with evidence, select finalists, capture course-cutoff results, and finish the combined report, presentation and individual peer reviews. [Actual contributions](../docs/contributions.csv), [report outline](../docs/report/) and [slide requirements](../docs/slides/).
