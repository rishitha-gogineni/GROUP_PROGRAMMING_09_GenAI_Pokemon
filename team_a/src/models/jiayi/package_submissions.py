import argparse
from pathlib import Path
import shutil
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = Path(__file__).resolve().parent
COMMON_FILES = (
    "team_a/__init__.py",
    "team_a/src/__init__.py",
    "team_a/src/contracts.py",
    "team_a/src/features.py",
    "team_a/src/heuristic.py",
    "team_a/src/models/__init__.py",
    "team_a/src/models/rishitha/__init__.py",
    "team_a/src/models/rishitha/ar1/__init__.py",
    "team_a/src/models/rishitha/ar1/features.py",
    "team_a/src/models/jiayi/__init__.py",
    "team_a/src/models/jiayi/agents.py",
    "team_a/src/models/jiayi/policy.py",
)
POLICIES = {
    "A-J1": {
        "entrypoint": "aj1/main.py",
        "model": "aj1_bc.json",
        "loader": "team_a/src/models/jiayi/model_1.py",
        "archive": "A-J1_Linear_Policy.tar.gz",
    },
    "A-J2": {
        "entrypoint": "aj2/main.py",
        "model": "aj2_bc.json",
        "loader": "team_a/src/models/jiayi/model_2.py",
        "archive": "A-J2_MLP_Policy.tar.gz",
    },
}


def read_deck(path):
    values = [int(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
              if line.strip()]
    if len(values) != 60:
        raise ValueError("deck.csv must contain 60 integer card IDs")
    return values


def build_submission(model_id, deck_file, output_dir):
    spec = POLICIES[model_id]
    deck_file = Path(deck_file).resolve()
    read_deck(deck_file)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / spec["archive"]
    with tempfile.TemporaryDirectory(prefix="jiayi_submission_") as temporary:
        staging = Path(temporary)
        files = {"main.py": MODEL_DIR / spec["entrypoint"],
                 "model.json": MODEL_DIR / spec["model"],
                 "deck.csv": deck_file,
                 spec["loader"]: ROOT / spec["loader"]}
        files.update({name: ROOT / name for name in COMMON_FILES})
        for name, source in files.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        with tarfile.open(target, "w:gz") as archive:
            for name in sorted(files):
                archive.add(staging / name, arcname=name, recursive=False)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build standalone A-J1 and A-J2 agent archives")
    parser.add_argument("--deck-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    for model_id in POLICIES:
        print(build_submission(model_id, args.deck_file, args.output_dir))


if __name__ == "__main__":
    main()
