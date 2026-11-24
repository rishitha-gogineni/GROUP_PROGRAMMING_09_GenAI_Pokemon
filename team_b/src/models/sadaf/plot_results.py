"""Comparison chart + table for Sadaf's models (report / slides).

Run from the repository root:
    python3 -m team_b.src.models.sadaf.plot_results

Reads the newest evidence file per model in team_b/experiments/sadaf/ and writes
    team_b/experiments/sadaf/figures/sadaf_model_comparison.png
    team_b/experiments/sadaf/figures/sadaf_model_comparison.csv
Win rates are BO3 match win rates with Wilson 95% intervals (handbook 9.1).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVIDENCE = Path(__file__).resolve().parents[3] / "experiments" / "sadaf"
FIGURES = EVIDENCE / "figures"
LABELS = {"B-SA1": "Decision Tree (B-SA1)", "B-SA2": "Random Forest (B-SA2)"}
COLORS = {"baseline": "#2a6f97", "random": "#a9c5d9"}


def latest_runs() -> dict[str, dict]:
    runs = {}
    for f in sorted(EVIDENCE.glob("B-*.json")):          # names sort by timestamp
        data = json.loads(f.read_text())
        runs[data["model_id"]] = data | {"file": f.name}
    return runs


def main() -> None:
    runs = latest_runs()
    if not runs:
        raise SystemExit(f"no evidence files in {EVIDENCE}; run train.py first")
    models = sorted(runs)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Table (CSV) for the report.
    table = FIGURES / "sadaf_model_comparison.csv"
    with open(table, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["model", "opponent", "wins", "losses", "draws", "errors", "win_rate",
                    "ci_low", "ci_high", "ms_per_decision", "fit_seconds", "test_top1", "evidence"])
        for m in models:
            r = runs[m]
            for opp, res in r["evaluation"]["results"].items():
                w.writerow([m, opp, res["win"], res["loss"], res["draw"], res["error"], res["win_rate"],
                            res["wilson95"][0], res["wilson95"][1], res["mean_decision_ms"],
                            r["fit_seconds"], r["offline"]["test_top1"], r["file"]])

    # Chart: grouped bars with 95% interval whiskers and a 50% reference line.
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    width = 0.36
    for j, opp in enumerate(("baseline", "random")):
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(models):
            res = runs[m]["evaluation"]["results"][opp]
            xs.append(i + (j - 0.5) * width)
            ys.append(res["win_rate"] * 100)
            lo.append((res["win_rate"] - res["wilson95"][0]) * 100)
            hi.append((res["wilson95"][1] - res["win_rate"]) * 100)
        label = "vs heuristic-v1 baseline" if opp == "baseline" else "vs random"
        bars = ax.bar(xs, ys, width, yerr=[lo, hi], capsize=4, color=COLORS[opp],
                      label=label, error_kw={"elinewidth": 1, "ecolor": "#333333"})
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, 2, f"{y:.0f}%", ha="center",
                    va="bottom", fontsize=9, color="white" if opp == "baseline" else "#1b1b1b")
    ax.axhline(50, color="#888888", linestyle="--", linewidth=1, label="50% = even with opponent")
    n = runs[models[0]]["args"]["matches"]
    ax.set_xticks(range(len(models)), [LABELS.get(m, m) for m in models])
    ax.set_ylim(0, 118)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Best-of-three match win rate (%)")
    ax.set_title(f"Sadaf's models: local win rate ({n} BO3 matches each, 95% CI)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5)
    fig.tight_layout()
    out = FIGURES / "sadaf_model_comparison.png"
    fig.savefig(out)
    print(f"[saved] {out.relative_to(EVIDENCE.parents[2])}")
    print(f"[saved] {table.relative_to(EVIDENCE.parents[2])}")


if __name__ == "__main__":
    main()
