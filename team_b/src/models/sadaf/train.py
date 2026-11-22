"""Train + evaluate Sadaf's models (B-SA1 Decision Tree, B-SA2 Random Forest).

Run from the repository root:
    python3 -m team_b.src.models.sadaf.train --model tree
    python3 -m team_b.src.models.sadaf.train --model forest
Add --record to append the measured results to team_b/experiments/results.csv.

Steps: collect games -> split by game -> fit -> offline metrics ->
full BO3 matches vs heuristic-v1 baseline and vs random -> save evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

from team_b.src import baseline as B
from team_b.src import simulator
from team_b.src.adapter import read_deck, safe_agent
from team_b.src.models.sadaf import pipeline as P

TEAM_B = Path(__file__).resolve().parents[3]
DECK = TEAM_B / "data" / "raw" / "deck_sample.csv"
ARTIFACTS = TEAM_B / "artifacts" / "sadaf"         # git-ignored (handbook 13.8)
EVIDENCE = TEAM_B / "experiments" / "sadaf"        # small JSON summaries, committed
OUTCOME = {"W": "win", "L": "loss", "D": "draw"}


def load_model_class(name: str):
    if name == "tree":
        from team_b.src.models.sadaf.model_1 import ARCHITECTURE, MODEL_ID, Model
    else:
        from team_b.src.models.sadaf.model_2 import ARCHITECTURE, MODEL_ID, Model
    return Model, MODEL_ID, ARCHITECTURE


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def match_outcomes(summary: dict) -> list[str]:
    """BO3 results for our agent. A match with an aborted game counts as 'error'
    and is left out of the win rate (Revanth's review point on incomplete series)."""
    out = []
    for m in summary["match_results"]:
        aborted = any(g["outcome_for_a"] == "X" for g in m["games"])
        out.append("error" if aborted else OUTCOME[m["result_for_a"]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["tree", "forest"], required=True)
    ap.add_argument("--games", type=int, default=300, help="training games to record")
    ap.add_argument("--matches", type=int, default=30, help="BO3 matches per opponent")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record", action="store_true", help="append rows to results.csv")
    args = ap.parse_args()

    Model, model_id, architecture = load_model_class(args.model)
    deck = read_deck(DECK)

    # 1. Data: every decision from heuristic-vs-random games, split by whole game.
    t0 = time.time()
    data = P.collect(args.games, deck, seed=args.seed)
    tr, va, te = P.split_by_game(data["games"], seed=args.seed)
    rows, y = data["rows"], data["targets"]
    print(f"[data] {args.games} games, {len(rows)} option rows "
          f"(train {len(tr)}, val {len(va)}, test {len(te)})  {time.time() - t0:.1f}s")

    # 2. Fit on the training games only.
    t0 = time.time()
    model = Model(seed=args.seed).fit([rows[i] for i in tr], [y[i] for i in tr])
    fit_seconds = time.time() - t0

    # 3. Offline metrics (diagnostic only, handbook 9.3).
    def offline(idx):
        pred = model.score_rows([rows[i] for i in idx])
        mse = sum((p - y[i]) ** 2 for p, i in zip(pred, idx)) / len(idx)
        return round(mse, 3), round(P.top1_agreement(pred, data, idx), 4)
    val_mse, val_top1 = offline(va)
    test_mse, test_top1 = offline(te)
    print(f"[fit] {fit_seconds:.1f}s  val MSE {val_mse}  test MSE {test_mse}  "
          f"test top-1 agreement with teacher {test_top1:.1%}")

    # 4. Save the trained model (git-ignored) and its hash.
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    artifact = ARTIFACTS / f"{model_id}.joblib"
    model.save(artifact)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()[:12]

    # 5. Full-game evaluation: same deck both sides, alternating seats.
    results = {}
    for name, opponent in (("baseline", B.baseline_response), ("random", B.random_response)):
        agent = safe_agent(P.make_policy(model), deck)
        t0 = time.time()
        summary = simulator.run_series(agent, opponent, deck, deck, matches=args.matches, seed=args.seed)
        outcomes = match_outcomes(summary)
        counts = {k: outcomes.count(k) for k in ("win", "loss", "draw", "error")}
        done = counts["win"] + counts["loss"] + counts["draw"]
        lo, hi = simulator.wilson_interval(counts["win"], done)
        results[name] = {
            **counts, "completed": done,
            "win_rate": round(counts["win"] / done, 4) if done else 0.0,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "mean_decision_ms": round(summary["mean_decision_ms_a"], 4),
            "fallbacks": agent.stats["fallbacks"],
            "decisions": agent.stats["decisions"],
            "seconds": round(time.time() - t0, 1),
            "outcomes": outcomes,
        }
        r = results[name]
        print(f"[vs {name}] W-L-D {r['win']}-{r['loss']}-{r['draw']} (errors {r['error']})  "
              f"win rate {r['win_rate']:.1%}  95% CI {r['wilson95']}  "
              f"{r['mean_decision_ms']} ms/decision  fallbacks {r['fallbacks']}")

    # 6. Evidence file (committed) with everything needed to reproduce.
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"B-{stamp}-{model_id}"
    evidence = {
        "experiment_id": experiment_id, "model_id": model_id, "architecture": architecture,
        "commit": git_commit(), "artifact_hash": artifact_hash, "schema_version": P.SCHEMA_VERSION,
        "target": "teacher_option_scores (heuristic-v1)", "args": vars(args),
        "data": {"games": args.games, "rows": len(rows), "split": "by game 70/15/15"},
        "model": model.describe(), "fit_seconds": round(fit_seconds, 2),
        "offline": {"val_mse": val_mse, "test_mse": test_mse, "val_top1": val_top1, "test_top1": test_top1},
        "evaluation": {"unit": "best_of_three_match", "seat_protocol": "alternating",
                       "deck": "deck_sample.csv both sides", "engine_seeded": False, "results": results},
    }
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE / f"{experiment_id}.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"[saved] {out.relative_to(TEAM_B.parent)}")

    # 7. Optional: one results.csv row per opponent (uses Sadaf's write_experiment).
    if args.record:
        from team_b.src.evaluation import summarize_matches, write_experiment
        for name, r in results.items():
            write_experiment({
                "member": "Sadaf Fatima Syeda", "experiment_id": f"{experiment_id}-vs-{name}",
                "model_id": model_id, "architecture": architecture, "commit": evidence["commit"],
                "artifact_hash": artifact_hash, "schema_version": P.SCHEMA_VERSION,
                "target": "teacher_option_scores", "objective": "regression (MSE)",
                "training_seed": args.seed, "training_rows_or_episodes": f"{args.games} games",
                "evaluation_config": json.dumps({"opponent": name, "matches": args.matches,
                                                 "unit": "best_of_three_match", "seats": "alternating",
                                                 "deck": "deck_sample.csv", "engine_seeded": False}),
                **summarize_matches(r["outcomes"]),
                "invalid_responses": r["fallbacks"],
                "mean_decision_ms": r["mean_decision_ms"],
                "fallback_fraction": round(r["fallbacks"] / r["decisions"], 4) if r["decisions"] else 0.0,
                "status": "measured_local",
                "notes": f"vs {name}; evidence experiments/sadaf/{out.name}",
            })
        print("[recorded] rows appended to team_b/experiments/results.csv")


if __name__ == "__main__":
    main()
