# A-R2 small MLP experiment

Run: 20261122T052210Z_0c2ec591. Parent: 20261122T002608Z_bd91a79d. Owner: Rishitha Gogineni. Local project date: 21 November 2026, America/Los_Angeles.

This run reuses the exact A-R1 feature module, normalization, teacher targets and game groups. Training uses 210 games and 2,161 decisions, validation 45 games and 539 decisions, and test 45 games and 390 decisions. No new collection or shuffle was performed. reference_ar1/ preserves the required parent files. These test games were already used for the planned A-R1 comparison, so they are not a fresh unseen benchmark.

The MLP has 375 inputs, 32 tanh hidden units and a scalar candidate score, with 12,064 parameters. Of the 375 features, 140 vary in training; other input rows stay zero. Training uses Adam, learning rate 0.003, seed 42, batch size 64, at most 60 epochs and patience 10. The three L2 settings match A-R1. Validation selected L2 0.0001 and epoch 60. A-R1 used learning rate 0.03, so the comparison preserves common data and objective without claiming all optimizer settings are identical.

Test cross entropy was 0.67886 versus A-R1 0.71579. Teacher top-set agreement was 99.49 versus 98.72 percent; exact choice agreement was 95.13 versus 96.15 percent. Teacher ties permit different highest scoring actions. The MLP did not establish stronger play: it won 36/40 against random, 15/40 against heuristic and 17/40 against A-R1. New controls were heuristic versus random 37/40, A-R1 versus random 37/40 and A-R1 versus heuristic 25/40. Keep these samples separate from the earlier day4 results.

All 240 evaluation battles and six extracted archive tests passed. No invalid responses or engine rejections were recorded. Finite difference gradient checks and 300 exported score comparisons passed. The exact source is in source/, fitted parameters and checkpoints in training/, charts in figures/, and per battle reports in each comparison folder. The model and deployed source are also copied without changes to ../../src/models/rishitha/ar2/.

The tested package hash is c420603d62a97462f135f1b5579ed91dcfb573138960bdad138dd5af3b108c64. Its model JSON hash is e84cb09280a385f5de84030186bb92700f98d63c56fbb4e2db48114f61c34d62. The separately saved upload archive matches these tested bytes. The archive and official deck remain local.

The user reports a later manual Kaggle upload. No official status, rating, ID or screenshot was provided. ../../submissions/A-R2/ separates that report from the local test results. Original output flags and all exported files retain their preupload values. integration_validation.json records static checks; no training, model inference or native simulator code was run locally during integration.

Reproduce with ../../notebooks/05_AR2_MLP_Scorer.ipynb in a fresh Colab CPU session, uploading the same official competition ZIP and the exact A-R1 output ZIP. Package readiness depends on execution checks, not winning. Native RNG control is not verified. Usage counters include callback tests and are not exact gameplay fallback fractions.
