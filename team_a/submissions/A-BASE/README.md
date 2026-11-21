# A-BASE - random baseline

The user supplied two Kaggle screenshots of the same named submission, both marked **Complete**. The first shows **600.0** at initialization; the later one shows **490.5**. The latter is the latest observed rating in this evidence set. These are snapshots of a changing ladder rating, not two model versions.

- [Submission record](submission_record.json)
- [Rating history](rating_observations.csv)
- [Initial Complete / 600.0 screenshot](kaggle_submission_complete.png)
- [Later Complete / 490.5 screenshot](kaggle_rating_490_5.png)
- [Local package and callback test evidence](../../experiments/day2/)

Neither image shows a submission ID, rating uncertainty, absolute observation time or exact Kaggle team name. Those fields remain unknown; GROUP_PROGRAMMING_09_a is the intended team. The local artifact hash is recorded, but a remote artifact hash cannot be read from a screenshot.

The archive itself and deck file remain local. The standalone entry point in agent_package/main.py matches the tested local archive exactly. Original pre-upload notebook status files retain their not_submitted flags; these later screenshots provide the subsequent Kaggle evidence.
