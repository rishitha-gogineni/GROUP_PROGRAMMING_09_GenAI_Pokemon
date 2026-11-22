# Shared Team A commands

These commands support standalone agents using the direct SDK individual battle API. The engine stays in a separate process. New output directories are required so old evidence is not overwritten. No command installs packages or uploads to Kaggle.

## Inputs

Extract the official competition ZIP locally. The SDK path below must contain cg/ and deck.csv from the innermost sample_submission folder. The recorded engine workflow used Python 3.13.18 on Linux x86_64. The shared commands themselves use the Python standard library. Training and plotting versions are recorded in ../requirements.txt.

Configured agents are random, heuristic, ar1 and ar2. Their entry points and model files resolve relative to the repository, regardless of the working directory. The deck defaults to SDK_DIR/deck.csv when present. Use --deck-file for an explicit local copy. The sample benchmark checks the recorded deck hash and requires the same deck for both policies. SDK Python source hashes are checked and all available SDK file hashes are recorded. The competition ZIP, native binaries and deck are not copied into Git.

## Evaluate a pair

From the repository root:

```bash
python team_a/scripts/evaluate.py \
  --candidate ar2 --opponent ar1 \
  --sdk-dir /absolute/path/to/official_sample \
  --output-dir /absolute/path/to/new_evaluation
```

This uses the 40 Python seeds in ../configs/evaluation.json, alternates candidate seats, and toggles the SDK reverse-player flag every two games. It reports wins, losses, draws, errors, invalid actions and approximate Wilson intervals. The half-draw score rate is separate from the binary win interval. Native engine randomness is not verified as seeded; equal Python seeds are not identical native matches. There is no best-of-three controller.

For a custom standalone folder:

```bash
python team_a/scripts/evaluate.py \
  --agent-dir /absolute/path/to/my_agent --opponent heuristic \
  --asset weights.bin \
  --sdk-dir /absolute/path/to/official_sample \
  --games 10 --seed-base 80000 \
  --output-dir /absolute/path/to/new_screening
```

The folder needs main.py and its resources. model.json is included automatically when present. Additional files or directories need repeated --asset arguments, relative to the entry point's folder. Custom opponent folders use --opponent-dir and --opponent-asset. Each entry point must expose agent as the final callable, support the initial deck callback and locate resources through configuration['__raw_path__']. Its working directory is intentionally unrelated to the package.

Main files should be self contained or use explicitly included, uniquely named Python asset packages. Conflicting Python asset namespaces between agents are rejected. Imports, framework installations and arbitrary repository dependencies are not made portable automatically. The four configured policies are self contained and use standard-library inference. New members' agents must implement the same full response contract, including a valid policy for decisions their learned model does not handle.

Use --config to supply a separate benchmark with a different declared deck or agent registry. Keep its actual unit and limitations accurate. --games selects a prefix of configured seeds; provide --seed-base for a larger or different schedule. These values are run budgets, not completed results.

## Build and validate a package

```bash
python team_a/scripts/package_agent.py \
  --candidate ar2 \
  --sdk-dir /absolute/path/to/official_sample \
  --output-dir /absolute/path/to/new_package
```

The command stages main.py, deck.csv, model.json when present, and explicitly selected assets. It checks archive membership and byte hashes, extracts into a new directory, then runs two battles against each of random, heuristic and ar1 in separate workers. Repeat --opponent to choose a subset of configured opponents. --games sets archive battles per opponent. Validation uses distinct seeds from the configuration's package_validation section.

Only when every check passes does the output contain submission.tar.gz with package_ready true. Winning is not required. Failed checks preserve diagnostics and do not publish a ready archive. Local readiness is not hosted Kaggle acceptance. The package and SDK must still satisfy the competition's actual hosted requirements.

To inspect a package without executing an agent or SDK:

```bash
python team_a/scripts/package_agent.py \
  --candidate ar2 --deck-file /absolute/path/to/deck.csv \
  --build-only --output-dir /absolute/path/to/new_inspection
```

Build-only writes candidate_unvalidated.tar.gz and always leaves package_ready false. It does not produce submission.tar.gz. Do not treat it as a tested upload. Other data files use --asset just as evaluation does. Symlinks and paths escaping the selected source folder are rejected.

## Read the outputs

Evaluation saves job.json, process.json, completion.json, report.json, summary.json, battle_results.csv, stdout.txt and stderr.txt. JSON and CSV checkpoints are written after every game. A timeout or abrupt worker exit preserves completed records and marks the parent completion failed. The parent process does not load the native SDK. max_decision_ms measures either policy, not the candidate alone. Optional policy counters exclude the command's synthetic deck check but include deck callbacks during games; no generic fallback fraction is inferred.

Packaging adds package_manifest.json and one archive_check folder per opponent. A nonzero exit means a required check failed. Error rates and partial results must remain in the record. Test fixtures are not real Pokemon performance evidence.

## Shared feature interface

team_a.src.features exposes ar1-visible-v1, the frozen encoder used by A-R1 and A-R2. It supports MAIN selections requiring one choice and containing at least two legal options. Each 375-element candidate vector already includes visible state interactions. Fitted normalization belongs to the model.

```python
from team_a.src.features import encode_observation
from team_a.src.models.rishitha.model_2 import Model

encoded = encode_observation(observation)
scores = Model().score_options(encoded['state'], encoded['options'])
chosen_row = max(range(len(scores)), key=scores.__getitem__)
chosen_option_index = encoded['option_indices'][chosen_row]
```

encoded['state'] is intentionally empty, matching the fitted scorer interfaces. encode_options returns complete candidate vectors in the original legal order. describe_state provides 21 inspectable state values, not a separate model input. The encoder raises ValueError for unsupported decisions; full agents must supply their own valid fallback. Extend coverage under a new schema rather than changing the frozen trained model format. See ../configs/feature_schema.json.

## Verification scope

The shared commands passed 11 standard-library tests covering saved observations, both fitted scorer interfaces, temporary fake agents, archive selection, failure gates, changed package assets, worker exits, configuration and seat schedules. The synthetic SDK is pure Python and loads no native library. Native SDK execution, model training and package installation were not run during this shared-tool verification. Original executed Colab notebooks remain the native gameplay evidence.

check_baseline.py remains available for the original Day 2 checks. train.py remains a generic placeholder; use the model-specific executed notebooks for actual fitting. These common commands do not implement reinforcement learning, replay collection or the full hosted Kaggle harness.
