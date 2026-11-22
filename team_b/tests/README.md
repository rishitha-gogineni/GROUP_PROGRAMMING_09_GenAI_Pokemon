# Acceptance checks to implement

Use genuine SDK observations and actual runtime behavior.

- Complete baseline game and clean-process loading.
- Initial deck phase and missing current/select fields.
- Single, multiple and optional selections with context-specific constraints.
- Variable options, index mapping, NaN scores and full valid fallback.
- Acting-player perspective and hidden-information exclusion.
- Game/match-grouped splits and training-only preprocessing.
- Candidate score shapes and feature/checkpoint schema compatibility.
- Official package layout and validated runtime paths.

Implemented in `test_baseline_pipeline.py`; run `python3 -m unittest discover -s team_b/tests -t . -v` from the repository root. Engine tests skip when `team_b/data/raw/cg` is missing.
