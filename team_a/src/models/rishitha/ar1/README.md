# Fitted A-R1 policy

main.py and model.json match the locally tested submission archive exactly. features.py matches the exported feature module. The model has 375 parameters for candidate features, plus fitted mean and scale vectors. No shared additive bias is needed because it would cancel within a candidate set.

The standard library agent reads model.json and the official deck.csv from the configured package path. deck.csv is not committed. Keep agent as the final callable when packaging. Use the executed A-R1 notebook to create and verify a package with main.py, model.json and deck.csv at the archive root.

The model scores supported single choice MAIN observations. Other selections use the frozen heuristic embedded in main.py. Source metadata and limits are in ../../../../experiments/rishitha/ar1_linear/training/feature_contract.json. Fitting and hosted Kaggle results are separate evidence; retain the provided screenshots and original run reports.
