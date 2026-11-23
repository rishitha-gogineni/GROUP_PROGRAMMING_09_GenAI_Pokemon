## Kavya
## Implementation and Results
### Feature Engineering

Implemented in `team_a/src/models/kavya/features.py`.

- `encode_state()`: 21 state features, reusing Rishitha's state encoder.
- `encode_options()`: 30 features per legal option.
- Combined model input: 51 dimensions.
- Supported selection types: 0, 1, 4, 8, 9.

Action feature breakdown:
- Selection type one-hot: 5
- Action type one-hot (including unknown): 16
- Normalized numerical fields: 9

Numerical fields: index, count, number, area,
inPlayArea, inPlayIndex, playerIndex,
energyIndex, attackId.

### Encoded Examples

| Example | Selection Type | State Dimensions | Option Dimensions |
|---|---|---|---|
| 1 | 9 | 21 | 30 |
| 2 | 1 | 21 | 30 |
| 3 | 1 | 21 | 30 |

Example 1:
- State prefix: [0.0, 0.0333, 0.0, 0.0, 0.0]
- Option prefix: [0.0, 0.0, 0.0, 0.0, 1.0]

Example 2:
- State prefix: [0.0, 0.0667, 0.0, 0.0, 0.0]
- Option prefix: [0.0, 1.0, 0.0, 0.0, 0.0]

Example 3:
- State prefix: [0.0, 0.1, 0.0, 0.0, 0.0]
- Option prefix: [0.0, 1.0, 0.0, 0.0, 0.0]

### Model Architectures

A-K1: Concatenated state-option MLP
- Input: 51
- Hidden layers: 128, 64
- Output: One score per legal option
- Parameters: 14,977

A-K2: Separate State and Option Encoders
- State encoder: 21 -> 64 -> 32
- Option encoder: 30 -> 64 -> 32
- Joint scoring head: 64 -> 64 -> 1
- Parameters: 11,777

### Training

- Dataset: Heuristic-teacher imitation data
- Training records: 2,161
- Validation records: 539
- Held-out records: 390
- Split grouping: Game ID
- Epochs: 20
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Soft-target cross-entropy
- Hardware: MacBook Air with PyTorch MPS

### Evaluation Results

| Metric | A-K1 | A-K2 |
|---|---|---|
| Held-out accuracy | 86.41% | 93.59% |
| Held-out loss | 0.8010 | 0.7924 |
| Best validation loss | 0.7725 | 0.8345 |

A-K2 achieved the highest held-out agreement with the
heuristic teacher.

Agent legality tests: 79/79 passed.

These results measure imitation accuracy, not battle win rate.

### Reproducibility

Train Model 1:
`python -m team_a.src.models.kavya.train --model 1 --epochs 20`

Train Model 2:
`python -m team_a.src.models.kavya.train --model 2 --epochs 20`

Saved checkpoints:
- `team_a/experiments/kavya/model_1.pt`
- `team_a/experiments/kavya/model_2.pt`

### Integration

Both models implement `score_options()`.

An independent test agent is available in
`team_a/src/models/kavya/kavya_agent.py`.

Shared Team A files have not been modified.

### Remaining Team Integration

- Confirm whether the shared feature interface should use
  Kavya's encoders.
- Evaluate deployed battle performance.
- Coordinate official submission and model registry updates.
