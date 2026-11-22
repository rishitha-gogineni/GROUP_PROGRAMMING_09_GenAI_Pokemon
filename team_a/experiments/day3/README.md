# Day 3 heuristic experiment

Run: 20261121T232326Z_2121ee87. Owner: Rishitha Gogineni. Policy: A-HEUR-01.

The executed Colab run completed 30 evaluation battles against the frozen Day 2 random policy: 24 wins, six losses, no draws or errors. The local win rate is 80 percent with an approximate Wilson 95 percent interval of 62.7 to 90.5 percent. This is a single opponent comparison with the official sample deck, not a hosted Kaggle match benchmark.

All 20 separate demonstration games passed and saved 176 accepted single choice MAIN decisions. dataset_manifest.json defines 14 training games with 127 rows, three validation games with 28 rows and three held out games with 21 rows. Groups are disjoint and exclude every evaluation game. These are heuristic labels, not expert actions or complete RL transitions.

Both tests from the freshly extracted archive passed. No invalid responses or engine rejections were recorded in any phase. package_manifest.json records a ready 12,504 byte archive and its hashes. The archive itself remains local.

source/ contains the exact exported scripts. heuristic.py and heuristic_runner.py are also copied without changes to team_a/src/. random_baseline_frozen.py matches the previously accepted Day 2 entry point. The adapter and original runner were not replaced. Reproduce the workflow using ../../notebooks/03_heuristic_baseline.ipynb in a fresh Colab CPU session with the official competition ZIP.

The original reports, logs, metadata, manifest and experiment_log.csv retain their Colab values. In particular, not_submitted was recorded before the manual Kaggle upload. Later evidence in ../../submissions/A-HEUR-01/ shows Complete with a displayed rating of 489.7, alongside random at 103.0. No submission ID or absolute rating observation time is visible.

integration_validation.json records source identity, dataset checks and notebook preservation checks. Local integration used static Python syntax checks and comparisons with the saved outputs. No simulator, training code or notebook cell was executed locally.
