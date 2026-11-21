# Runtime scripts

## check_baseline.py

Portable version of the tested Day 2 checker. It loads a standalone agent with the inspected last-callable loader pattern, checks deck/option response contracts and runs individual battles through team_a.src.simulator. Import paths were adjusted for this repository; those changes were statically checked. The original Colab reports document 10 local and 2 archive battles.

After agreeing on execution with the responsible user, run from the repository root in the verified Linux runtime:

```bash
python team_a/scripts/check_baseline.py \
  --agent-dir /absolute/path/to/unpacked/submission \
  --sdk-dir /absolute/path/to/official/sample/package \
  --output-dir /absolute/path/to/new/check/results \
  --games 10 --seed-base 100
```

The agent directory must contain main.py and deck.csv. The SDK directory must contain cg/. Keep both inputs local. The executed Day 2 notebook provides the complete extraction, testing and packaging workflow without installing extra packages.

## Other scripts

train.py, evaluate.py and package_agent.py remain generic placeholders. Training, controlled model evaluation and the general packaging CLI are not implemented by this baseline integration.
