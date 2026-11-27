# Energy attachment pilot

Run `20261126T211344Z_3118d91a` was executed in Colab on 26 November 2026. The experiment changes only the selected Energy card's legal attachment target in a frozen heuristic. It leaves the original heuristic outside that rule, shared encoder, A-R1 and A-R2 unchanged. It is an additional handcrafted experiment, not another trained architecture.

See the [executed notebook](../../../notebooks/rishitha/07_Energy_Attachment_Experiment.ipynb), [original evaluation summary](20261126T211344Z_3118d91a/evaluation_summary.json), [batch results](20261126T211344Z_3118d91a/batch_results.csv), [comparison CSV](20261126T211344Z_3118d91a/comparison_results.csv) and [figure](20261126T211344Z_3118d91a/battle_comparison.png).

| Comparison | Wins | Losses | Individual game win rate |
| --- | ---: | ---: | ---: |
| Original heuristic vs A-R1 | 40 | 40 | 50.00% |
| Energy rule vs original heuristic | 43 | 37 | 53.75% |
| Energy rule vs A-R1 | 42 | 38 | 52.50% |

All 13 rule checks and 240 individual games passed. Every process finished with zero invalid responses and engine rejections. Against the original heuristic, the new rule won 17/40 in batch 1 and 26/40 in batch 2. Against A-R1, it won 22/40 and 20/40. This does not establish a reliable improvement; the pilot assessment is `no_consistent_gain_across_both_batches`.

The rule changed 12 attachment choices across 10 of the 160 games using it. It examined 542 attachment decisions; 37 of these were Tool attachments, correctly left unchanged. [Saved-state diagnostics](diagnostics.json) explain the guards and one missed Bench attack upgrade. An offline candidate would change one additional saved choice, but was not promoted, packaged or tested in new games. Do not count it as a successful experiment or a new trained model.

## Evidence and reproduction

The run folder preserves exact source, protocol, checks, six game CSVs and reports, decision logs, process records and terminal observations. In each comparison folder, join `decisions.jsonl` to `games.csv` using `game_id`; attachment choices include the acting player's visible observation. Keep these detailed records within the authorized project audience.

The six full `energy_metadata.json` files are original competition reference material and remain local under `results/20261126T211344Z_3118d91a/` and in the archived output ZIP. The [integration manifest](20261126T211344Z_3118d91a/integration_manifest.json) records hashes and every omitted path. The supplied official SDK can regenerate this metadata in Colab using the exact source preserved here. SDK binaries, official downloads and transfer ZIPs stay local.

To reproduce, open the executed notebook in a fresh Colab CPU session and upload the original official competition ZIP when prompted. Python seeds do not control native shuffles and coin flips. These individual game results are separate from the common evaluation's external best of three matches. No Kaggle submission or model fitting occurred in this run.
