# A-BASE random baseline

Three supplied Kaggle screenshots mark the same named random submission Complete. They show 600.0 at initialization, then 490.5, then 103.0 in the image supplied after the Day 3 heuristic upload. The last image also shows the heuristic at 489.7. These are successive rating snapshots for the same random policy, not three model versions.

[Submission record](submission_record.json), [rating history](rating_observations.csv), [initial 600.0 image](kaggle_submission_complete.png), [490.5 image](kaggle_rating_490_5.png), [latest comparison image](../A-HEUR-01/kaggle_heuristic_489_7_random_103_0.png) and [local Day 2 checks](../../experiments/day2/).

The screenshots do not show submission IDs, rating uncertainty, absolute observation times or the exact team name. GROUP_PROGRAMMING_09_a is the intended team. The verified local package hash remains unchanged; a remote hash cannot be read from the images.

The archive and deck remain local. agent_package/main.py matches the tested archive entry point. Original Colab status files retain their preupload not_submitted flags; the later screenshots document the subsequent Kaggle result.
