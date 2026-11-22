# Fitted A-R2 MLP policy

main.py and model.json are byte identical to the verified Colab package. features.py is the same ar1-visible-v1 module used by A-R1. The network has 375 candidate inputs, 32 tanh units and a scalar output. The model file stores the fitted mean, scales, hidden weights, hidden biases and output weights.

The standard library entry point folds normalization into a cached network representation. It scores supported MAIN decisions with one required selection and at least two candidates. The frozen heuristic handles other decisions. Keep agent as the last callable when packaging.

The official deck is not committed. Use the executed A-R2 notebook to create a package with main.py, model.json and deck.csv at the archive root and test a fresh extraction. Exact input lineage and fitting checks are in ../../../../experiments/day5/. The separate repository Model wrapper passed shared feature-interface checks on three saved observations; the deployed entry point is the version actually exercised in Colab.
