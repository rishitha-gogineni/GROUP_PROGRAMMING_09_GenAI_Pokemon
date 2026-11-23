"""Team B evaluation tools (owner: Sadaf).

- write_experiment(record): add one row to experiments/results.csv safely.
- summarize_matches(outcomes): W/L/D/error counts, win rate and Wilson 95% CI.
- outcomes_from_series(summary): turn simulator.run_series() output into outcomes.
"""

import csv
import datetime as _dt
import math
from pathlib import Path
from typing import Mapping

TEAM_B_DIR = Path(__file__).resolve().parents[1]
RESULTS_CSV = TEAM_B_DIR / "experiments" / "results.csv"

REQUIRED = ("member", "experiment_id", "model_id", "status")

# simulator.run_match() result codes -> outcome words ("E" = aborted series).
RESULT_CODES = {"W": "win", "L": "loss", "D": "draw", "E": "error"}


def read_columns(path: Path = RESULTS_CSV) -> list[str]:
    """Return the column names from the CSV header row."""
    with open(path, newline="") as f:
        return next(csv.reader(f))


def write_experiment(record: Mapping[str, object], path: Path = RESULTS_CSV) -> dict:
    """Append one experiment result to the Team B results log.

    - Unknown column names raise an error (catches typos like "winrate").
    - Missing columns are left blank.
    - `date` and `team` are filled in automatically if not given.
    - `win_rate` is calculated from wins/matches if not given.
    Returns the row that was written.
    """
    columns = read_columns(path)

    unknown = set(record) - set(columns)
    if unknown:
        raise ValueError(f"Unknown column(s): {sorted(unknown)}. Allowed: {columns}")
    missing = [c for c in REQUIRED if not record.get(c)]
    if missing:
        raise ValueError(f"Missing required field(s): {missing}")

    row = {c: "" for c in columns}
    row.update({k: "" if v is None else v for k, v in record.items()})
    row["date"] = row["date"] or _dt.date.today().isoformat()
    row["team"] = row["team"] or "B"

    # Works whether numbers arrive as int or text ("0", "10"); never divides by 0.
    if row["win_rate"] == "" and row["wins"] != "" and int(row["matches"] or 0) > 0:
        row["win_rate"] = round(int(row["wins"]) / int(row["matches"]), 4)

    for c in ("win_rate", "mean_decision_ms"):
        if isinstance(row[c], float) and not math.isfinite(row[c]):
            raise ValueError(f"{c} must be a finite number, got {row[c]}")

    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=columns, lineterminator="\n").writerow(row)
    return row



def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence range for the true win rate (handbook 9.1)."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def outcomes_from_series(summary: dict) -> list[str]:
    """simulator.run_series(...) summary -> ["win", "loss", "draw", "error", ...]."""
    return [RESULT_CODES[m["result_for_a"]] for m in summary["match_results"]]


def summarize_matches(outcomes: list[str]) -> dict:
    """Add up a list of BO3 match results: "win", "loss", "draw" or "error".
    Errors (aborted series) are counted but excluded from the win rate.

    The returned keys match the results.csv columns, so the output can go
    straight into write_experiment().
    """
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    draws = outcomes.count("draw")
    errors = outcomes.count("error")
    completed = wins + losses + draws
    low, high = wilson_interval(wins, completed)
    return {
        "matches": completed,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "crashes": errors,
        "win_rate": round(wins / completed, 4) if completed else 0.0,
        "interval_method": "wilson95",
        "interval_lower": low,
        "interval_upper": high,
    }
