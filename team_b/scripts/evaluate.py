"""Evaluate one Team B policy against another in local best-of-three matches.

Run from the repository root:

    python3 -m team_b.scripts.evaluate --agent heuristic --opponent random --matches 200 --seed 7
    python3 -m team_b.scripts.evaluate --agent heuristic --opponent first --matches 200 --seed 7 --record

--record appends one row to team_b/experiments/results.csv (only measured values;
the git commit is taken from HEAD and flagged if the working tree has changes).
A JSON summary with per-game details is always written to team_b/outputs/ (ignored).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

from team_b.src.adapter import read_deck
from team_b.src.baseline import BASELINE_VERSION, POLICIES
from team_b.src.simulator import run_series

TEAM_B = Path(__file__).resolve().parents[1]
DEFAULT_DECK = TEAM_B / "data" / "raw" / "deck_sample.csv"
RESULTS_CSV = TEAM_B / "experiments" / "results.csv"

MODEL_IDS = {"heuristic": ("B-BASE", f"Rule-based heuristic scorer ({BASELINE_VERSION})"),
             "random": ("B-RANDOM", "Uniform random legal selection"),
             "first": ("B-FIRST", "First legal option(s)")}


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=TEAM_B, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=TEAM_B, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def deck_hash(deck: list[int]) -> str:
    return hashlib.sha256("\n".join(map(str, deck)).encode()).hexdigest()[:12]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent", default="heuristic", choices=sorted(POLICIES))
    ap.add_argument("--opponent", default="random", choices=sorted(POLICIES))
    ap.add_argument("--deck", default=str(DEFAULT_DECK), help="deck file for the agent")
    ap.add_argument("--opponent-deck", default=None, help="deck file for the opponent (default: same)")
    ap.add_argument("--matches", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", action="store_true", help="append a row to experiments/results.csv")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    deck_a = read_deck(args.deck)
    deck_b = read_deck(args.opponent_deck or args.deck)
    summary = run_series(POLICIES[args.agent], POLICIES[args.opponent], deck_a, deck_b,
                         matches=args.matches, seed=args.seed)

    now = dt.datetime.now(dt.timezone.utc)
    exp_id = f"B-{now:%Y%m%d-%H%M%S}-{args.agent}-vs-{args.opponent}"
    config = {"agent": args.agent, "opponent": args.opponent, "deck_hash": deck_hash(deck_a),
              "opponent_deck_hash": deck_hash(deck_b), "matches": args.matches, "seed": args.seed,
              "unit": "best_of_three_match", "seat_protocol": "alternating", "engine_seeded": False}
    out_dir = TEAM_B / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{exp_id}.json").write_text(json.dumps({"config": config, **summary}, indent=1))

    brief = {k: v for k, v in summary.items() if k != "match_results"}
    print(json.dumps({"experiment_id": exp_id, "config": config, **brief}, indent=2))

    if args.record:
        model_id, arch = MODEL_IDS[args.agent]
        decisions = sum(g["decisions"][g["a_seat"]] for r in summary["match_results"] for g in r["games"])
        row = {
            "date": now.date().isoformat(), "team": "B", "member": "Chelsi Shulamit Elthuri",
            "experiment_id": exp_id, "model_id": model_id, "architecture": arch,
            "commit": git_commit(), "artifact_hash": "", "schema_version": "", "data_version": "",
            "target": "", "objective": "", "training_seed": "", "training_rows_or_episodes": "",
            "training_hours": "",
            "evaluation_config": json.dumps(config, separators=(",", ":")),
            "matches": summary["matches"], "wins": summary["wins"], "losses": summary["losses"],
            "draws": summary["draws"], "win_rate": round(summary["win_rate"], 4),
            "interval_method": "wilson95", "interval_lower": summary["wilson95"][0],
            "interval_upper": summary["wilson95"][1],
            "invalid_responses": summary["invalid_responses_a"], "crashes": summary["aborted_games"],
            "mean_decision_ms": round(summary["mean_decision_ms_a"], 4),
            "fallback_fraction": round(summary["invalid_responses_a"] / decisions, 6) if decisions else "",
            "kaggle_submission_id": "", "status": "measured_local",
            "notes": args.notes or f"vs {args.opponent}; local cabt engine; engine RNG not seedable",
        }
        with open(RESULTS_CSV, newline="") as f:
            header = next(csv.reader(f))
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow(row)
        print(f"recorded {exp_id} in {RESULTS_CSV.relative_to(TEAM_B.parent)}")


if __name__ == "__main__":
    main()
