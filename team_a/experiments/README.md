# Experiment records

results.csv records measured runs. Day 3 preserves the initial heuristic comparison. Day 4 preserves A-R1 fitting and three local comparisons. Day 5 preserves A-R2 fitting and six new comparisons, including frozen A-R1 controls. Do not overwrite the older results when a fresh sample differs. Historical Day 3 through Day 5 rows report individual battles and leave matches blank. The energy pilot also uses individual games. The common_evaluation rows use externally controlled best of three matches and populate matches; these are separate from the hosted Kaggle harness.

model_registry.csv records A-R1 as trained and submitted with screenshot evidence, and A-R2 as trained with a user-reported upload awaiting hosted evidence. Other members' statuses remain unchanged. The handcrafted heuristic does not count as a learned architecture.

Both learned models use the same features, normalization, targets and game groups. A-R2 improved teacher top-set agreement but did not establish stronger play in its battle samples. Keep this ineffective improvement attempt and the weaker comparisons in the report. Training and packaging succeeded; distinguish those execution checks from playing strength and Kaggle acceptance.

Preserve data and schema versions, source hashes, controls, uncertainty and the separation of fitting, validation, test and battle evaluation. Reused test groups support the planned comparison, not a fresh unseen benchmark. Policy usage counters include callback contract checks. Missing measurements remain blank.

## Completed evaluation additions

* [Energy rule pilot](energy_rule/): 240 individual games, 13 checks and saved-state diagnostics. No reliable improvement was established; reference agents and shared encoder remain unchanged.
* [Common evaluation](common_evaluation/): A-R1 and A-R2 each completed 120 matches with the same protocol/runtime. Overall scores were 83.33% and 85.00%; the small difference does not establish a reliable winner.

Original run records are immutable. Integration manifests record every source file and the local-only official metadata. No Kaggle submission status is inferred from these Colab results. Statuses belonging to other members were not revised by this integration.
