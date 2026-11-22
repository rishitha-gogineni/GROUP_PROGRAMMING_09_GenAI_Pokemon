# Experiment records

results.csv records measured runs. Day 3 preserves the initial heuristic comparison. Day 4 preserves A-R1 fitting and three local comparisons. Day 5 preserves A-R2 fitting and six new comparisons, including frozen A-R1 controls. Do not overwrite the older results when a fresh sample differs. Each row reports individual battles; matches stays blank because the full hosted match harness was not run.

model_registry.csv records A-R1 as trained and submitted with screenshot evidence, and A-R2 as trained with a user-reported upload awaiting hosted evidence. Other members' statuses remain unchanged. The handcrafted heuristic does not count as a learned architecture.

Both learned models use the same features, normalization, targets and game groups. A-R2 improved teacher top-set agreement but did not establish stronger play in its battle samples. Keep this ineffective improvement attempt and the weaker comparisons in the report. Training and packaging succeeded; distinguish those execution checks from playing strength and Kaggle acceptance.

Preserve data and schema versions, source hashes, controls, uncertainty and the separation of fitting, validation, test and battle evaluation. Reused test groups support the planned comparison, not a fresh unseen benchmark. Policy usage counters include callback contract checks. Missing measurements remain blank.
