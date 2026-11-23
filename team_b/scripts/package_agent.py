"""Build and smoke-test the Team B Kaggle submission archive.

Run from the repository root:

    python3 -m team_b.scripts.package_agent                 # baseline, sample deck
    python3 -m team_b.scripts.package_agent --deck path/to/deck.csv --games 10

Layout of team_b/outputs/submission.tar.gz (archive root, as Kaggle requires):
    main.py        <- team_b/agent_package/main.py
    deck.csv       <- --deck (validated: 60 known card IDs, basic legality checks)
    teamb/         <- team_b/src/*.py (+ models/ if present)
    cg/            <- official engine wrapper + Linux libraries from team_b/data/raw/cg

The smoke test runs the STAGED package in a fresh Python process (so nothing is
imported from the repository) and plays complete games against a random agent.
Archives are git-ignored; record uploads in team_b/submissions/register.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TEAM_B = Path(__file__).resolve().parents[1]
OUT = TEAM_B / "outputs"
STAGING = OUT / "package_staging"
ARCHIVE = OUT / "submission.tar.gz"
CG_SRC = TEAM_B / "data" / "raw" / "cg"
CG_FILES = ["__init__.py", "api.py", "game.py", "sim.py", "utils.py", "libcg.so", "libcg-arm64.so"]
SIZE_LIMIT_MB = 197.7

SMOKE_TEST = r"""
import os, random, sys, time
os.chdir(sys.argv[1]); sys.path.insert(0, sys.argv[1])
ns = {}
exec(open("main.py").read(), ns)               # mimic the simulator loading main.py
agent = ns["agent"]
from cg.game import battle_start, battle_select, battle_finish
deck = agent({"select": None, "logs": [], "current": None})
assert isinstance(deck, list) and len(deck) == 60, "deck phase must return 60 card IDs"
games, wins, invalid, worst_ms = int(sys.argv[2]), 0, 0, 0.0
for g in range(games):
    me = g % 2
    obs, _ = battle_start(deck, deck)
    while obs["current"]["result"] == -1:
        s = obs["select"]
        if obs["current"]["yourIndex"] == me:
            t = time.perf_counter(); resp = agent(obs); worst_ms = max(worst_ms, (time.perf_counter() - t) * 1000)
            ok = (len(set(resp)) == len(resp) and s["minCount"] <= len(resp) <= s["maxCount"]
                  and all(0 <= i < len(s["option"]) for i in resp))
            invalid += not ok
        else:
            resp = random.sample(range(len(s["option"])), s["maxCount"])
        obs = battle_select(resp)
    wins += obs["current"]["result"] == me
    battle_finish()
print(f"smoke test: {games} games, {wins} wins vs random, {invalid} invalid responses, worst decision {worst_ms:.2f} ms")
sys.exit(1 if invalid else 0)
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage(deck_path: Path) -> None:
    sys.path.insert(0, str(TEAM_B.parent))
    from team_b.src.adapter import read_deck
    from team_b.src.card_features import deck_problems

    deck = read_deck(deck_path)
    problems = deck_problems(deck)
    if problems:
        raise SystemExit(f"deck {deck_path} failed checks: {problems}")

    if STAGING.exists():
        shutil.rmtree(STAGING)
    (STAGING / "teamb").mkdir(parents=True)
    (STAGING / "cg").mkdir()
    shutil.copy2(TEAM_B / "agent_package" / "main.py", STAGING / "main.py")
    (STAGING / "deck.csv").write_text("\n".join(map(str, deck)) + "\n")
    for py in (TEAM_B / "src").glob("*.py"):
        shutil.copy2(py, STAGING / "teamb" / py.name)
    shutil.copytree(TEAM_B / "src" / "models", STAGING / "teamb" / "models",
                    ignore=shutil.ignore_patterns("__pycache__", "*.md", "*.json"))
    for name in CG_FILES:
        src = CG_SRC / name
        if not src.exists():
            raise SystemExit(f"missing {src}: copy the official sample_submission/cg folder there")
        shutil.copy2(src, STAGING / "cg" / name)


def smoke_test(games: int) -> None:
    # On macOS the staged package has no libcg.dylib (Kaggle runs Linux), so the
    # smoke test borrows the local build via a temporary copy.
    dylib = CG_SRC / "libcg.dylib"
    temp = STAGING / "cg" / "libcg.dylib"
    if sys.platform == "darwin":
        if not dylib.exists():
            raise SystemExit("macOS smoke test needs team_b/data/raw/cg/libcg.dylib (see team_b/README.md)")
        shutil.copy2(dylib, temp)
    try:
        subprocess.run([sys.executable, "-I", "-c", SMOKE_TEST, str(STAGING), str(games)], check=True)
    finally:
        temp.unlink(missing_ok=True)
        for cache in STAGING.rglob("__pycache__"):
            shutil.rmtree(cache)


def build_archive() -> None:
    with tarfile.open(ARCHIVE, "w:gz") as tar:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(STAGING)))
    with tarfile.open(ARCHIVE) as tar:
        names = tar.getnames()
    assert "main.py" in names and "deck.csv" in names, "main.py/deck.csv must be at the archive root"
    size_mb = ARCHIVE.stat().st_size / (1024 * 1024)
    if size_mb > SIZE_LIMIT_MB:
        raise SystemExit(f"archive is {size_mb:.1f} MiB > {SIZE_LIMIT_MB} MiB limit")
    print(f"archive: {ARCHIVE.relative_to(TEAM_B.parent)}  ({size_mb:.2f} MiB, {len(names)} files)")
    print(f"sha256:  {sha256(ARCHIVE)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", default=str(TEAM_B / "data" / "raw" / "deck_sample.csv"))
    ap.add_argument("--games", type=int, default=6, help="smoke-test games against random")
    args = ap.parse_args()
    stage(Path(args.deck))
    smoke_test(args.games)
    build_archive()


if __name__ == "__main__":
    main()
