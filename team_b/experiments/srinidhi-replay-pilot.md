# Team B progress for B-SR1 and B-SR2

**November 22, 2026 · In progress**

## Completed

- Verified replay extraction and separation of training, validation, and test data by replay.
- Implemented the B-SR1 linear policy and the B-SR2 multilayer perceptron. Ran a local pilot for both models.
- Verified that the exported checkpoints run through the policy interface.
- Passed all 26 local simulator, adapter, and model tests.

## Local simulator results

Each model played two 200-match best-of-three series against Chelsi’s frozen `heuristic-v1` baseline (`c3b5bb4`). Both sides used the official sample deck, with alternating starting seats. [Team B simulator PR](https://github.com/rishitha-gogineni/GROUP_PROGRAMMING_09_GenAI_Pokemon/pull/2)

| Model | First run (W–L–D) | Repeat run (W–L–D) | Combined win rate |
| --- | --- | --- | ---: |
| B-SR1 linear | 109–91–0 | 99–101–0 | 208/400 (52.0%) |
| B-SR2 MLP | 124–76–0 | 110–90–0 | 234/400 (58.5%) |

Both policies completed all matches with zero invalid moves and zero aborted games. B-SR2 won more matches in both runs.

## Next

- Coordinate with Sadaf to log one row per model and run in the Team B experiment log.
- Finish the expanded replay training run and repeat the simulator comparison.
- Prepare the selected model for submission and record the official result.
