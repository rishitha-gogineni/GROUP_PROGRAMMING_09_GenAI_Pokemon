"""Isolated direct SDK evaluation for the energy target experiment."""
import csv
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time
import traceback

from common_benchmark_core import deck_ids, digest_json, package_context, save_json, validate_action


def load_agent(directory):
    directory = Path(directory).resolve()
    main = directory / 'main.py'
    namespace = {}
    with package_context(directory):
        exec(compile(main.read_text(encoding='utf-8-sig'), str(main), 'exec'), namespace)
    function = namespace['agent']
    if [value for value in namespace.values() if callable(value)][-1] is not function:
        raise ValueError('agent must be the final callable.')
    config = {'__raw_path__': str(main)}

    def callback(observation, seed):
        config['seed'] = seed
        with package_context(directory):
            return function(observation, config)

    return callback, namespace


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    output = Path(job['output_dir'])
    report = {'status': 'running', 'contract_passed': False, 'games': [], 'error': None,
              'diagnostic_rows': 0, 'diagnostics_truncated': False, 'metadata_sha256': None}
    fields = ['game_id', 'seed', 'candidate_seat', 'reverse_player', 'outcome', 'status', 'steps',
              'seconds', 'invalid_responses', 'engine_rejections', 'rule_eligible', 'rule_changes',
              'candidate_decisions', 'candidate_total_decision_ms', 'candidate_max_decision_ms',
              'error', 'cleanup_error']

    def checkpoint():
        save_json(output / 'report.json', report)
        temporary = output / 'games.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            writer.writerows(report['games'])
        temporary.replace(output / 'games.csv')

    checkpoint()
    try:
        sys.path.insert(0, job['sdk_dir'])
        from cg.api import all_card_data, all_attack
        from cg.game import battle_start, battle_select, battle_finish
        from cg.sim import Battle
        metadata = {'cards': {str(item.cardId): asdict(item) for item in all_card_data()},
                    'attacks': {str(item.attackId): asdict(item) for item in all_attack()}}
        # JSON removes IntEnum values before rule type checks.
        metadata = json.loads(json.dumps(metadata))
        report['metadata_sha256'] = digest_json(metadata)
        save_json(output / 'energy_metadata.json', metadata)
        save_json(Path(job['agent_dirs']['energy']) / 'energy_metadata.json', metadata)
        random.seed(job['seed_start'])
        first, first_namespace = load_agent(job['agent_dirs'][job['candidate']])
        second, second_namespace = load_agent(job['agent_dirs'][job['opponent']])
        expected = deck_ids(Path(job['sdk_dir']) / 'deck.csv')
        initial = {'select': None, 'current': None, 'logs': [], 'search_begin_input': None}
        for callback in (first, second):
            returned = callback(dict(initial), job['seed_start'])
            if not isinstance(returned, list) or any(type(card) is not int for card in returned) or returned != expected:
                raise ValueError('Agent deck callback differs from the fixed sample deck.')
        report['contract_passed'] = True
        checkpoint()
        with (output / 'decisions.jsonl').open('w', encoding='utf-8') as log:
            for index in range(job['games']):
                seat, seed = index % 2, job['seed_start']+index
                reverse = bool((index//2) % 2)
                agents = [(first, first_namespace, job['candidate']), (second, second_namespace, job['opponent'])]
                if seat == 1:
                    agents.reverse()
                random.seed(seed)
                began = time.perf_counter()
                row = dict(game_id=index+1, seed=seed, candidate_seat=seat, reverse_player=reverse,
                           outcome='error', status='running', steps=0, seconds=None, invalid_responses=0,
                           engine_rejections=0, rule_eligible=0, rule_changes=0, candidate_decisions=0,
                           candidate_total_decision_ms=0.0, candidate_max_decision_ms=0.0,
                           error='', cleanup_error='')
                report['games'].append(row)
                checkpoint()
                try:
                    decks = [callback(dict(initial), seed) for callback, _, _ in agents]
                    if any(not isinstance(deck, list) or any(type(card) is not int for card in deck) or deck != expected for deck in decks):
                        row['invalid_responses'] += 1
                        raise ValueError('Deck changed during evaluation.')
                    observation, start_info = battle_start(decks[0], decks[1], reverse)
                    if observation is None:
                        raise RuntimeError('Battle start failed: '+str(start_info))
                    for step in range(job['max_steps']+1):
                        current = observation['current']
                        result = current['result']
                        if result != -1:
                            if result not in (0, 1, 2):
                                raise ValueError('Unexpected terminal result.')
                            row.update(status='passed', outcome='draw' if result == 2 else 'win' if result == seat else 'loss',
                                       final_visible_current=current)
                            break
                        if step == job['max_steps']:
                            raise RuntimeError('Selection limit reached.')
                        player, selection = current['yourIndex'], observation.get('select')
                        if player not in (0, 1) or selection is None:
                            raise ValueError('Invalid acting state.')
                        callback, namespace, policy = agents[player]
                        before = time.perf_counter()
                        action = callback(observation, seed)
                        elapsed = (time.perf_counter()-before)*1000
                        if player == seat:
                            row['candidate_decisions'] += 1
                            row['candidate_total_decision_ms'] += elapsed
                            row['candidate_max_decision_ms'] = max(row['candidate_max_decision_ms'], elapsed)
                        try:
                            validate_action(action, selection)
                        except Exception:
                            row['invalid_responses'] += 1
                            raise
                        detail = namespace.get('ENERGY_LAST_DECISION') if policy == 'energy' else None
                        if detail and player == seat:
                            row['rule_eligible'] += bool(detail['eligible'])
                            row['rule_changes'] += bool(detail['changed'])
                        record = {'game_id': index+1, 'step': step, 'player': player, 'policy': policy,
                                  'selection_type': selection.get('type'), 'context': selection.get('context'),
                                  'action': action, 'chosen_options': [selection['option'][i] for i in action],
                                  'energy_rule': detail}
                        if any(option.get('type') == 8 for option in record['chosen_options']):
                            record['visible_observation'] = {'current': current, 'select': selection}
                        if report['diagnostic_rows'] < job['max_diagnostic_rows']:
                            log.write(json.dumps(record, allow_nan=False)+'\n')
                            log.flush()
                            report['diagnostic_rows'] += 1
                        else:
                            report['diagnostics_truncated'] = True
                        try:
                            observation = battle_select(action)
                            if observation is None:
                                raise RuntimeError('Simulator rejected the response.')
                        except Exception:
                            row['engine_rejections'] += 1
                            raise
                        row['steps'] += 1
                except Exception as error:
                    row.update(status='error', outcome='error', error=type(error).__name__+': '+str(error),
                               traceback=traceback.format_exc())
                finally:
                    try:
                        if Battle.battle_ptr:
                            battle_finish()
                    except Exception as error:
                        row.update(status='cleanup_error', outcome='error', cleanup_error=str(error))
                    finally:
                        Battle.battle_ptr = None
                    row['seconds'] = time.perf_counter()-began
                    checkpoint()
                print('Game', index+1, '/', job['games'], row['outcome'], '| rule changes:', row['rule_changes'], flush=True)
        report['status'] = 'passed' if all(row['status'] == 'passed' for row in report['games']) else 'failed'
    except Exception as error:
        report.update(status='error', error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        checkpoint()
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1]))
