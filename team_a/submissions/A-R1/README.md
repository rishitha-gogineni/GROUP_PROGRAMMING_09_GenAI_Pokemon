# A-R1 submission

The supplied Kaggle screenshot shows Group9_TeamA_AR1_Linear_Action_Scorer_submission.tar.gz as Complete at 828.8. The same image shows heuristic Complete at 257.6 and random Complete at 11.5. These are rating snapshots with no visible absolute observation time.

[Screenshot](kaggle_ar1_828_8_heuristic_257_6_random_11_5.png), [submission record](submission_record.json), [rating observation](rating_observations.csv), [executed notebook](../../notebooks/04_AR1_Linear_Scorer.ipynb) and [local run evidence](../../experiments/day4/).

The verified local archive is 20,803 bytes with SHA256 97971f8d74f76180fea2632b168425c7259908497573ea2622ebbae8097cb6d3. It contains main.py, model.json and deck.csv. The separately downloaded submission archive matches the package that passed the local checks. The deployed entry point and fitted model are integrated under ../../src/models/rishitha/ar1/. The deck and archive remain local.

Local results were 40 wins in 40 battles against random and 19 wins in 40 battles against the heuristic. Both tests against each opponent after archive extraction passed. No invalid responses or engine rejections were recorded. The deployed agent uses heuristic responses for unsupported decisions.

Submission ID, exact submission time, absolute rating observation time, uncertainty and actual Kaggle team name are not visible in the image. These fields remain blank. GROUP_PROGRAMMING_09_a is the intended team. The remote archive hash cannot be verified from a screenshot. Original Colab files retain not_submitted because manual upload followed the saved run.
