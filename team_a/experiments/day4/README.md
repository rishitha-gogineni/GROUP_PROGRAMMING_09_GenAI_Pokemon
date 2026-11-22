# A-R1 linear scorer experiment

Run: 20261122T002608Z_bd91a79d. Owner: Rishitha Gogineni. Local project date: 21 November 2026, America/Los_Angeles.

The completed Colab run collected 300 fresh heuristic versus random games with 3,346 accepted decisions. It excluded 256 forced choices and used 3,090 decisions: 2,161 from 210 training games, 539 from 45 validation games, and 390 from 45 held out games. Game groups are disjoint. Both wins and losses were eligible. The earlier Day 3 games were not reused.

The model uses 375 ar1-visible-v1 candidate features. These include visible state interactions with action type. Teacher scores and selected choices are labels, not features. Legal candidate cross entropy distributes the target equally among teacher maximum score actions. Normalization uses training candidate rows only. Three L2 settings were tried with the same seed. Validation selected L2 0.0 and epoch 14. Held out teacher top set agreement is 98.72 percent; exact teacher choice agreement is 96.15 percent.

The deployed policy uses the learned scorer only for supported MAIN decisions and the frozen heuristic otherwise. New local evaluation recorded heuristic versus random 35/40, A-R1 versus random 40/40, and A-R1 versus heuristic 19/40. No improvement over the heuristic is established by the last comparison. All four freshly extracted archive tests passed. No invalid responses or engine rejections were recorded across collection, evaluation and archive phases.

source/ contains the exact exported scripts. training/ contains game splits, schema, fitted JSON, checkpoints, metrics and training history. collection/ contains the original 300 game record files. figures/ contains the two saved plots. The fitted main.py and model.json are also integrated in ../../src/models/rishitha/ar1/.

Use ../../notebooks/04_AR1_Linear_Scorer.ipynb in a fresh Colab CPU session with the same official ZIP to reproduce the workflow. The original output reports, CSVs, source bytes and preupload not_submitted flags are preserved. Later Kaggle evidence in ../../submissions/A-R1/ shows Complete at 828.8, heuristic 257.6 and random 11.5. Absolute observation times and IDs are not visible.

The local submission archive hash is 97971f8d74f76180fea2632b168425c7259908497573ea2622ebbae8097cb6d3. The fitted model JSON hash is ff58cc8edfc730d4f3529568a880c1d49cd05b1ea4588875b14a0a57a11ebd4c. The separately downloaded package matches the tested archive. The archive and official deck remain local. integration_validation.json records static integration checks; no native SDK or training code was executed locally.

Policy usage counters include synthetic callback checks. They should not be reported as an exact fraction of gameplay decisions without separating those checks. The 40 battle samples and approximate Wilson intervals do not establish performance across other decks or opponents. Native RNG control is not verified.
