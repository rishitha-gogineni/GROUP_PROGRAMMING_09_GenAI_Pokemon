"""Build a Kaggle submission for Sadaf's model, reusing Chelsi's packager.

Run from the repository root AFTER training (train.py saves the model):
    python3 -m team_b.src.models.sadaf.package --model tree
    python3 -m team_b.src.models.sadaf.package --model forest

Output: team_b/outputs/B-SA1.tar.gz or B-SA2.tar.gz (git-ignored).
The archive is smoke-tested as a fresh process before it is written.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from team_b.scripts import package_agent as pa
from team_b.src.models.sadaf.portable import export_json

HERE = Path(__file__).resolve().parent
ARTIFACTS = pa.TEAM_B / "artifacts" / "sadaf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["tree", "forest"], required=True)
    ap.add_argument("--games", type=int, default=10, help="smoke-test games against random")
    args = ap.parse_args()

    if args.model == "tree":
        from team_b.src.models.sadaf.model_1 import MODEL_ID, Model
    else:
        from team_b.src.models.sadaf.model_2 import MODEL_ID, Model
    trained = ARTIFACTS / f"{MODEL_ID}.joblib"
    if not trained.exists():
        raise SystemExit(f"{trained} not found: run train.py --model {args.model} first")

    # 1. Chelsi's staging: deck.csv, cg/ engine, teamb/ (copy of team_b/src).
    pa.stage(pa.TEAM_B / "data" / "raw" / "deck_sample.csv")
    # 2. Swap in Sadaf's entry point and the exported (sklearn-free) model.
    shutil.copy2(HERE / "kaggle_main.py", pa.STAGING / "main.py")
    export_json(Model.load(trained), pa.STAGING / "sadaf_model.json")
    # 3. Make sure the package really uses the model (not the baseline fallback).
    check = ("import os,sys; os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1]); ns={}; "
             "exec(open('main.py').read(), ns); "
             "from teamb.baseline import baseline_response; "
             "sys.exit('model NOT loaded: package would play the baseline' "
             "if ns['_inner'] is baseline_response else 0)")
    subprocess.run([sys.executable, "-I", "-c", check, str(pa.STAGING)], check=True)
    print("model check: sadaf_model.json loads inside the package")
    # 4. Smoke test the staged package, then build the archive.
    pa.smoke_test(args.games)
    pa.ARCHIVE = pa.OUT / f"{MODEL_ID}.tar.gz"
    pa.build_archive()


if __name__ == "__main__":
    main()
