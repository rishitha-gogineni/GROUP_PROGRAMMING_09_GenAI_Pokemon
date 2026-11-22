HEURISTIC EXPERIMENT

source/heuristic.py is a separate rule-based candidate. Do not overwrite the accepted random baseline.
evaluation/ contains head-to-head results; demonstrations/ contains separately generated teacher examples.
dataset_manifest.json defines game-level train, validation and held-out groups.
Use only passed eligible games in the manifest. Partial failed-game rows are diagnostics.
source/heuristic_runner.py adds an optional observation hook to the Day 2 battle logic.
If package_ready is true, submission.tar.gz passed a fresh-extraction process test.
Upload only that inner archive to Kaggle, not the output ZIP or notebook.
No trained linear or MLP model is produced by this notebook.
