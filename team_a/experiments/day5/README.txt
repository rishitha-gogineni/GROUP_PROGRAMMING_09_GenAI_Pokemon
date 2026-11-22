A-R2 RUN

Small tanh MLP using the frozen A-R1 features, game groups and normalization. Teacher targets and fallback policy are unchanged. reference_ar1/ preserves the required prior evidence. training/ contains model weights, checks, checkpoints and the A-R1 versus A-R2 imitation comparison. Test groups are reused for the planned comparison, not a new unseen benchmark. Six new battle comparisons use the sample deck and individual battles. Native RNG is not verified as seeded. Policy usage includes callback checks. No Kaggle upload is made by this notebook. Keep weaker results and failures.
