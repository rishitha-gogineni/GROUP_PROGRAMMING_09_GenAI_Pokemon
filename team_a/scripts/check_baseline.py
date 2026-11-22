"""Run callback checks and SDK battles in an isolated Python process."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import traceback


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_callback(path):
    # Match the inspected Kaggle loader's exec-and-last-callable behavior.
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    functions = [value for value in namespace.values() if callable(value)]
    if not functions or functions[-1].__name__ != "agent":
        raise ValueError("The last callable in main.py must be agent.")
    callback = functions[-1]
    configuration = {"__raw_path__": str(path)}
    return lambda observation: callback(observation, configuration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-dir", required=True)
    parser.add_argument("--sdk-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=100)
    args = parser.parse_args()
    agent_dir = Path(args.agent_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(args.sdk_dir).resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from team_a.src.simulator import check_action, run_battle

    report = {
        "status": "running", "all_passed": False, "unit": "individual_battle",
        "loader_check": "exec without __file__; last callable; configured agent path",
        "contract_checks": [], "battles": [], "error": None,
        "scope": "Direct SDK and local loader checks; not the full Kaggle harness or a leaderboard evaluation.",
    }
    save_json(out / "report.json", report)
    try:
        agents = [load_callback(agent_dir / "main.py") for _ in range(2)]
        initial = {"select": None, "logs": [], "current": None, "search_begin_input": None}
        returned_deck = agents[0](initial)
        expected_deck = [int(line) for line in (agent_dir / "deck.csv").read_text().splitlines() if line.strip()]
        assert returned_deck == expected_deck and len(returned_deck) == 60
        report["contract_checks"].append({"case": "initial_deck_from_unrelated_working_directory", "passed": True})
        cases = [
            ("single", {"minCount": 1, "maxCount": 1, "option": [{}, {}]}),
            ("multiple", {"minCount": 2, "maxCount": 3, "option": [{}, {}, {}, {}, {}]}),
            ("zero_options", {"minCount": 0, "maxCount": 0, "option": []}),
            ("optional", {"minCount": 0, "maxCount": 2, "option": [{}, {}, {}]}),
        ]
        for name, selection in cases:
            check_action(agents[0]({"select": selection}), selection)
            report["contract_checks"].append({"case": name, "passed": True})
        save_json(out / "report.json", report)
        for index in range(args.games):
            record = run_battle(agents, seed=args.seed_base + index, reverse_player=bool(index % 2))
            record["battle_id"] = index + 1
            report["battles"].append(record)
            save_json(out / "report.json", report)
            print({key: record[key] for key in ("battle_id", "status", "steps", "seconds", "invalid_responses", "engine_rejections")}, flush=True)
        report["all_passed"] = (
            len(report["battles"]) == args.games
            and all(row["finished"] and row["status"] == "passed" for row in report["battles"])
            and all(row["passed"] for row in report["contract_checks"])
        )
        report["status"] = "passed" if report["all_passed"] else "failed"
    except Exception as error:
        report.update(status="error", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
    finally:
        save_json(out / "report.json", report)
        fields = ["battle_id", "python_seed", "status", "finished", "result_code", "steps", "seconds", "max_decision_ms", "reverse_player", "invalid_responses", "engine_rejections", "error_type", "error_message", "cleanup_error"]
        with (out / "battle_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(report["battles"])
    print("Local checks:", report["status"], flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
