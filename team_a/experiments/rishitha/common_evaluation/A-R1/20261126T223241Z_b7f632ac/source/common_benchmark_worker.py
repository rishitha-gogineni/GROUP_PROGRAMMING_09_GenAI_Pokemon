"""Run one public reference comparison in an isolated process."""
import csv
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

from common_benchmark_core import (deck_ids, load_candidate, match_outcome,
                                   reference_action, save_json, validate_action)


def run_game(candidate, opponent, deck, seed, seat, reverse, max_steps, api):
    start, select, finish, Battle = api
    random.seed(seed)
    rng = random.Random(seed)
    began = time.perf_counter()
    row = dict(python_seed=seed, candidate_seat=seat, reverse_player=reverse,
               status='running', outcome='error', result_code=None, steps=0,
               invalid_responses=0, engine_rejections=0, candidate_decisions=0,
               candidate_total_decision_ms=0.0, candidate_max_decision_ms=0.0,
               error='', traceback='', cleanup_error='')

    def candidate_call(observation):
        before = time.perf_counter()
        try:
            return candidate(observation, seed)
        finally:
            elapsed = (time.perf_counter() - before)*1000
            row['candidate_decisions'] += 1
            row['candidate_total_decision_ms'] += elapsed
            row['candidate_max_decision_ms'] = max(row['candidate_max_decision_ms'], elapsed)

    try:
        initial = {'select': None, 'current': None, 'logs': [], 'search_begin_input': None}
        returned = candidate_call(dict(initial))
        if not isinstance(returned, list) or any(type(card) is not int for card in returned) or returned != deck:
            row['invalid_responses'] += 1
            raise ValueError('Candidate must return the fixed 60-card sample deck in the recorded order.')
        observation, start_info = start(list(deck), list(deck), reverse)
        if observation is None:
            raise RuntimeError('Battle start failed: ' + str(start_info))
        for step in range(max_steps+1):
            current = observation.get('current')
            if current is None:
                raise ValueError('Simulator returned no current state.')
            result = current['result']
            if result != -1:
                if result not in (0, 1, 2):
                    raise ValueError('Unknown terminal result: ' + str(result))
                row.update(status='passed', result_code=result,
                           outcome='draw' if result == 2 else 'win' if result == seat else 'loss')
                break
            if step == max_steps:
                raise RuntimeError('Selection limit reached before the game ended.')
            player, selection = current['yourIndex'], observation.get('select')
            if player not in (0, 1) or selection is None:
                raise ValueError('Unexpected acting player or missing selection.')
            action = (candidate_call(observation) if player == seat else
                      reference_action(opponent, observation, deck, rng))
            try:
                validate_action(action, selection)
            except Exception:
                row['invalid_responses'] += 1
                raise
            try:
                observation = select(action)
                if observation is None:
                    raise RuntimeError('Simulator rejected the selection.')
            except Exception:
                row['engine_rejections'] += 1
                raise
            row['steps'] += 1
    except Exception as error:
        row.update(status='error', outcome='error', error=type(error).__name__ + ': ' + str(error),
                   traceback=traceback.format_exc())
    finally:
        try:
            if Battle.battle_ptr:
                finish()
        except Exception as error:
            row.update(status='cleanup_error', outcome='error', cleanup_error=str(error))
        finally:
            Battle.battle_ptr = None
        row['seconds'] = time.perf_counter() - began
    return row


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    output = Path(job['output_dir'])
    report = {'status': 'running', 'contract_passed': False, 'matches': [], 'error': None}

    def checkpoint():
        save_json(output / 'report.json', report)
        fields = ['match_id', 'game_id', 'outcome', 'status', 'python_seed', 'candidate_seat',
                  'reverse_player', 'result_code', 'steps', 'seconds', 'invalid_responses',
                  'engine_rejections', 'candidate_decisions', 'candidate_total_decision_ms',
                  'candidate_max_decision_ms', 'error', 'cleanup_error']
        temporary = output / 'games.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            for match in report['matches']:
                writer.writerows(match['games'])
        temporary.replace(output / 'games.csv')

    checkpoint()
    try:
        os.environ['PTCG_CG_DIR'] = str(Path(job['sdk_dir']) / 'cg')
        sys.path.insert(0, job['sdk_dir'])
        from cg.game import battle_start, battle_select, battle_finish
        from cg.sim import Battle
        api = battle_start, battle_select, battle_finish, Battle
        random.seed(job['seed_base'])
        candidate, mode = load_candidate(job['candidate_dir'])
        deck = deck_ids(Path(job['sdk_dir']) / 'deck.csv')
        if deck_ids(Path(job['candidate_dir']) / 'deck.csv') != deck:
            raise ValueError('Candidate deck.csv differs from the common sample deck.')
        initial = {'select': None, 'current': None, 'logs': [], 'search_begin_input': None}
        returned = candidate(initial, job['seed_base'])
        if not isinstance(returned, list) or any(type(card) is not int for card in returned) or returned != deck:
            raise ValueError('Candidate deck callback failed the fixed deck check.')
        report.update(contract_passed=True, callback_mode=mode)
        checkpoint()
        for match_index in range(job['matches_per_opponent']):
            match = {'match_id': match_index+1, 'outcome': None, 'games': []}
            report['matches'].append(match)
            checkpoint()
            for game_index in range(3):
                seed = job['seed_base'] + match_index*3 + game_index
                seat = (match_index + game_index) % 2
                reverse = bool((match_index//2 + game_index//2) % 2)
                game = run_game(candidate, job['opponent'], deck, seed, seat, reverse, job['max_steps'], api)
                game.update(match_id=match_index+1, game_id=game_index+1)
                match['games'].append(game)
                match['outcome'] = match_outcome([row['outcome'] for row in match['games']])
                checkpoint()
                if match['outcome'] is not None:
                    break
            print('Match', match_index+1, '/', job['matches_per_opponent'], match['outcome'], flush=True)
        report['status'] = ('passed' if all(row['outcome'] in ('win', 'loss', 'draw')
                                           for row in report['matches']) else 'failed')
    except Exception as error:
        report.update(status='error', error={'type': type(error).__name__, 'message': str(error),
                                            'traceback': traceback.format_exc()})
    finally:
        checkpoint()
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(worker(sys.argv[1]))
