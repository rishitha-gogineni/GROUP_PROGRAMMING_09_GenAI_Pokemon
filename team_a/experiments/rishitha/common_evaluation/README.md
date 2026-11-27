# Common evaluation of A-R1 and A-R2

Both frozen agents completed the same `group9_common_bo3_v1` protocol in Colab on 26 November 2026. Protocol and recorded runtime fingerprints match. File hashes identify the original tested A-R1 and A-R2 submission archives, with no energy-rule changes.

| Opponent | A-R1 match wins | A-R2 match wins |
| --- | ---: | ---: |
| Sample random | 39/40 | 40/40 |
| First options | 21/40 | 22/40 |
| Last options | 40/40 | 40/40 |
| Overall match score | 83.33% | 85.00% |

Each agent completed 120 externally controlled matches. A-R1 played 271 individual games; A-R2 played 265. All 240 matches and 536 games completed with zero invalid responses, engine rejections or failed workers. Every match was checked against its recorded game winners, stop rule, seed, seat and reverse schedule. Both executed notebooks match the prepared code and contain no execution errors.

The aggregate score is the unweighted mean of the three opponent match score rates. Wins earn one point and draws half a point. These runs had no draws. A match stops at two wins or after three games; at the cap, more wins takes the match and equal wins is a draw. Each game is a new SDK battle; prior match results are not injected into the observation.

A-R2 won two more matches, a 1.67 percentage point lead. Forty matches per reference, simple controls and uncontrolled native randomness do not establish a reliable overall winner. The recorded mean callback times were 0.296 ms for A-R1 and 1.676 ms for A-R2, including per-game deck callbacks; these are diagnostic Colab timings, not matched hardware latency guarantees. Keep the earlier head to head and heuristic evaluations separate. Neither this benchmark nor `selection_eligible` establishes hosted Kaggle acceptance, a new rating or final model selection.

## Files

* [A-R1 executed notebook](../../../notebooks/rishitha/06_Common_Evaluation_A-R1.ipynb) and [run records](A-R1/20261126T223241Z_b7f632ac/).
* [A-R2 executed notebook](../../../notebooks/rishitha/06_Common_Evaluation_A-R2.ipynb) and [run records](A-R2/20261126T223506Z_daef86ed/).
* [Results by opponent](comparison_by_opponent.csv), [summary by agent](summary_by_agent.csv) and [combined record](comparison.json).
* [Reusable common notebook](../../../notebooks/06_Common_Agent_Evaluation.ipynb) stays in its original location for both teams' existing links.

Each run retains its original reports, game CSVs, protocol, source, per-opponent summaries, process logs, candidate/package hashes and integration manifest. Original downloaded ZIPs and notebook files remain in the local project archive. Official input archives, native libraries and agent submission archives stay local.

To reproduce, use a fresh Colab CPU session for each model. Run the reusable notebook, upload the original official competition ZIP first and the model's own submission `.tar.gz` second. Keep the settings unchanged and the same sample deck. The notebook runs local games and makes no Kaggle submissions. Share permitted summary evidence; keep detailed private logs and model files within the appropriate team.
