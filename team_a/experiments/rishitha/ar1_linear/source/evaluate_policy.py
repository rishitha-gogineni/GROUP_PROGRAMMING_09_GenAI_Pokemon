"""Individual-battle comparison, incremental records and separate teacher demonstrations."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import traceback


def save_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def load_agent(directory):
    path=(directory/"main.py").resolve()
    namespace={}
    exec(compile(path.read_text(encoding="utf-8"),str(path),"exec"),namespace)
    functions=[value for value in namespace.values() if callable(value)]
    if not functions or functions[-1].__name__!="agent":
        raise ValueError("Expected agent as the last callable.")
    function=functions[-1]
    config={"__raw_path__":str(path)}
    return lambda obs:function(obs,config),namespace


def wilson(wins,total):
    if not total:return None
    z=1.959963984540054
    p=wins/total
    denominator=1+z*z/total
    center=(p+z*z/(2*total))/denominator
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0.0,center-half),min(1.0,center+half)]


def main():
    parser=argparse.ArgumentParser()
    for flag in ("sdk-dir","random-dir","heuristic-dir","output-dir","run-id","phase"):
        parser.add_argument("--"+flag,required=True)
    parser.add_argument("--games",type=int,default=30)
    parser.add_argument("--seed-start",type=int,default=7000)
    parser.add_argument("--collect",action="store_true")
    parser.add_argument("--max-demo-records-per-game",type=int,default=500)
    args=parser.parse_args()
    if args.games<=0 or args.max_demo_records_per_game<=0:
        parser.error("Games and record cap must be positive.")
    root=Path(args.output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(args.sdk_dir).resolve()))
    from heuristic_runner import run_battle,check_action
    rows=[]
    contracts=[]
    report={"status":"running","all_passed":False,"requested_battles":args.games,
            "run_id":args.run_id,"phase":args.phase,"unit":"individual_battle",
            "contract_checks":contracts,"battles":rows,"current_game_id":None,
            "demonstrations_enabled":args.collect,"error":None}
    fields=["game_id","battle_id","python_seed","candidate_seat","reverse_player","outcome",
            "status","finished","result_code","steps","seconds","max_decision_ms",
            "invalid_responses","engine_rejections","demo_records","supported_decisions_seen",
            "demo_cap_reached","demo_file","error_type","error_message","cleanup_error"]

    def checkpoint():
        save_json(root/"report.json",report)
        temporary=root/"battle_results.csv.tmp"
        with temporary.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore")
            writer.writeheader();writer.writerows(rows)
        temporary.replace(root/"battle_results.csv")

    checkpoint()
    try:
        heuristic,heuristic_namespace=load_agent(Path(args.heuristic_dir))
        random_agent,_=load_agent(Path(args.random_dir))
        initial={"select":None,"logs":[],"current":None,"search_begin_input":None}
        expected=[int(line) for line in (Path(args.heuristic_dir)/"deck.csv").read_text().splitlines() if line.strip()]
        assert len(expected)==60 and heuristic(initial)==expected
        contracts.append({"case":"deck_callback_without___file__","passed":True})
        cases=[
            ("single",{"minCount":1,"maxCount":1,"option":[{"type":14},{"type":13,"attackId":1}],"type":0,"context":0}),
            ("multiple",{"minCount":2,"maxCount":2,"option":[{"type":3}]*3,"type":1,"context":2}),
            ("zero",{"minCount":0,"maxCount":0,"option":[],"type":4,"context":30}),
            ("optional",{"minCount":0,"maxCount":1,"option":[{"type":1},{"type":2}],"type":9,"context":43}),
        ]
        for name,selection in cases:
            check_action(heuristic({"select":selection,"current":None}),selection)
            contracts.append({"case":name,"passed":True})
        checkpoint()
        for index in range(args.games):
            game_id=f"{args.run_id}:{args.phase}:{index+1:04d}"
            report["current_game_id"]=game_id
            checkpoint()
            print("Starting",game_id,flush=True)
            seat=index%2
            agents=[heuristic,random_agent] if seat==0 else [random_agent,heuristic]
            counts={"saved":0,"seen":0}
            relative=f"records/game_{index+1:04d}.jsonl" if args.collect else ""
            demo_handle=None
            if args.collect:
                (root/"records").mkdir(exist_ok=True)
                demo_handle=(root/relative).open("w",encoding="utf-8")

            def observe(observation,action,step,player):
                selection=observation.get("select") or {}
                if player!=seat or selection.get("type")!=0 or selection.get("minCount")!=1 or selection.get("maxCount")!=1:
                    return
                counts["seen"]+=1
                if counts["saved"]>=args.max_demo_records_per_game:return
                options=selection["option"]
                scores=[float(heuristic_namespace["_option_score"](option,observation)) for option in options]
                assert len(action)==1 and all(math.isfinite(score) for score in scores)
                assert scores[action[0]]==max(scores)
                example={"schema_version":"heuristic-main-v1","run_id":args.run_id,"game_id":game_id,
                         "step":step,"turn":observation["current"].get("turn"),"acting_player":player,
                         "teacher_id":"A-HEUR-01","teacher_kind":"handcrafted_rules_not_expert",
                         "selection_type":selection["type"],"selection_context":selection.get("context"),
                         "legal_option_indices":list(range(len(options))),"chosen_option_index":action[0],
                         "chosen_option_indices":action,"teacher_scores":scores,"selection_accepted":True,
                         "observation":observation}
                demo_handle.write(json.dumps(example)+"\n")
                demo_handle.flush()
                counts["saved"]+=1

            try:
                result=run_battle(agents,seed=args.seed_start+index,reverse_player=bool((index//2)%2),
                                  decision_observer=observe if args.collect else None)
            finally:
                if demo_handle is not None:demo_handle.close()
            win_code=result["result_code"]
            if result["status"]!="passed" or not result["finished"]:outcome="error"
            elif win_code==2:outcome="draw"
            elif win_code==seat:outcome="win"
            elif win_code in (0,1):outcome="loss"
            else:outcome="unknown_result"
            result["policy_usage"] = dict(heuristic_namespace.get("POLICY_USAGE", {}))
            result.update(game_id=game_id,battle_id=index+1,candidate_seat=seat,outcome=outcome,
                          demo_records=counts["saved"],supported_decisions_seen=counts["seen"],
                          demo_cap_reached=counts["seen"]>counts["saved"],demo_file=relative)
            rows.append(result)
            checkpoint()
            print(f"Finished {index+1}/{args.games}: {outcome}; status={result['status']}; samples={counts['saved']}",flush=True)
        report["all_passed"]=(len(rows)==args.games and all(row["status"]=="passed" and row["finished"] and row["outcome"] in ("win","loss","draw") for row in rows))
        report["status"]="passed" if report["all_passed"] else "failed"
    except Exception as error:
        report.update(status="error",error={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()})
    finally:
        checkpoint()
        counts=Counter(row["outcome"] for row in rows)
        valid=counts["win"]+counts["loss"]+counts["draw"]
        summary={"phase":args.phase,"requested_battles":args.games,"recorded_battles":len(rows),
                 "valid_battles":valid,"win":counts["win"],"loss":counts["loss"],"draw":counts["draw"],
                 "error":counts["error"],"unknown_result":counts["unknown_result"],
                 "win_rate":counts["win"]/valid if valid else None,
                 "score_rate":(counts["win"]+.5*counts["draw"])/valid if valid else None,
                 "win_rate_wilson_95":wilson(counts["win"],valid),
                 "local_checks_passed":report["all_passed"],"unit":"individual_battle",
                 "demonstration_records":sum(row["demo_records"] for row in rows),
                 "note":"Rates use completed valid battles. Do not claim improvement if any game failed. The interval is an approximate binomial interval; it does not account for opponent/deck dependence or native RNG control."}
        summary["policy_usage"] = rows[-1].get("policy_usage", {}) if rows else {}
        save_json(root/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)
    return 0 if report["all_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
