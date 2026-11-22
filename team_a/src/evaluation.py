"""Portable individual-battle evaluation with an isolated SDK worker."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback
import uuid

from .common_runtime import (DEFAULT_CONFIG, REPO_ROOT, check_import_collisions, load_agent,
                             load_config, new_output, resolve_agent, save_json, select_seeds,
                             stage_agent, verify_decks, verify_sdk)


def wilson(wins, total):
    if not total:
        return None
    z = 1.959963984540054
    p, denominator = wins / total, 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-half), min(1.0, center+half)]


def usage(namespace):
    values = namespace.get('POLICY_USAGE')
    return {key: value for key, value in values.items() if type(value) is int and value >= 0} if isinstance(values, dict) else {}


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    output = Path(job['output_dir'])
    sys.path.insert(0, job['sdk_dir'])
    from .simulator import run_battle

    rows, checks = [], []
    report = {'status': 'running', 'all_passed': False, 'unit': 'individual_battle',
              'run_id': job['run_id'], 'phase': job['phase'], 'requested_battles': len(job['seeds']),
              'contract_checks': checks, 'battles': rows, 'current_game_id': None, 'error': None,
              'scope': 'Direct SDK and standalone callback checks, not the full hosted Kaggle harness.'}
    fields = ['game_id', 'battle_id', 'python_seed', 'candidate_seat', 'reverse_player', 'outcome',
              'status', 'finished', 'result_code', 'steps', 'seconds', 'max_decision_ms',
              'invalid_responses', 'engine_rejections', 'error_type', 'error_message', 'cleanup_error']

    def checkpoint():
        save_json(output / 'report.json', report)
        temporary = output / 'battle_results.csv.tmp'
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / 'battle_results.csv')

    checkpoint()
    try:
        candidate, namespace = load_agent(job['candidate_dir'])
        opponent, _ = load_agent(job['opponent_dir'])
        initial = {'select': None, 'current': None, 'logs': [], 'search_begin_input': None}
        for label, callback, directory in [('candidate', candidate, job['candidate_dir']), ('opponent', opponent, job['opponent_dir'])]:
            expected = [int(line) for line in (Path(directory) / 'deck.csv').read_text().splitlines() if line.strip()]
            deck = callback(dict(initial))
            if not isinstance(deck, list) or len(deck) != 60 or not all(type(card) is int for card in deck) or deck != expected:
                raise ValueError(label + ' did not return its configured 60-card deck.')
            checks.append({'case': label + '_deck_without___file__', 'passed': True})
        report['contract_policy_usage'] = usage(namespace)
        checkpoint()
        for index, seed in enumerate(job['seeds']):
            game_id = f"{job['run_id']}:{job['phase']}:{index+1:04d}"
            report['current_game_id'] = game_id
            checkpoint()
            print('Starting', game_id, flush=True)
            before = usage(namespace)
            seat = index % 2
            agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
            result = run_battle(agents, seed=seed, max_steps=job['config']['max_steps'], reverse_player=bool((index//2)%2))
            after = usage(namespace)
            if result['status'] != 'passed' or not result['finished']:
                outcome = 'error'
            elif result['result_code'] == 2:
                outcome = 'draw'
            elif result['result_code'] in (0, 1):
                outcome = 'win' if result['result_code'] == seat else 'loss'
            else:
                outcome = 'unknown_result'
            result.update(game_id=game_id, battle_id=index+1, candidate_seat=seat, outcome=outcome,
                          policy_usage={key: after[key]-before.get(key, 0) for key in after if after[key] >= before.get(key, 0)})
            rows.append(result)
            checkpoint()
            print('Finished', index+1, '/', len(job['seeds']), outcome, result['status'], flush=True)
        report['all_passed'] = (len(rows) == len(job['seeds']) and all(row['status'] == 'passed' and row['finished']
                                and row['outcome'] in ('win', 'loss', 'draw') for row in rows))
        report['status'] = 'passed' if report['all_passed'] else 'failed'
    except Exception as error:
        report.update(status='error', error={'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()})
    finally:
        checkpoint()
        counts = Counter(row['outcome'] for row in rows)
        valid = counts['win'] + counts['loss'] + counts['draw']
        counters = Counter()
        for row in rows:
            counters.update(row['policy_usage'])
        summary = {'unit': 'individual_battle', 'phase': job['phase'], 'requested_battles': len(job['seeds']),
                   'recorded_battles': len(rows), 'valid_battles': valid, 'win': counts['win'], 'loss': counts['loss'],
                   'draw': counts['draw'], 'error': counts['error'], 'unknown_result': counts['unknown_result'],
                   'win_rate': counts['win']/valid if valid else None,
                   'score_rate': (counts['win']+.5*counts['draw'])/valid if valid else None,
                   'win_rate_wilson_95': wilson(counts['win'], valid), 'local_checks_passed': report['all_passed'],
                   'invalid_responses': sum(row['invalid_responses'] for row in rows),
                   'engine_rejections': sum(row['engine_rejections'] for row in rows),
                   'game_policy_usage': dict(counters),
                   'notes': 'Rates use valid completed battles. Draws count as non-wins in the binary Wilson interval and half a point in score_rate. The interval does not adjust for game dependence. max_decision_ms measures either agent. Usage excludes synthetic contract calls but includes game deck callbacks. Native RNG control is unverified.'}
        save_json(output / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if report['all_passed'] else 1


def run_pair(candidate_dir, opponent_dir, sdk_dir, output_dir, config, seeds, run_id, phase, metadata):
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Evaluation requires distinct nonnegative Python seeds.')
    output = new_output(output_dir)
    check_import_collisions(candidate_dir, opponent_dir)
    sdk_hashes = verify_sdk(sdk_dir, config)
    job = {'run_id': run_id, 'phase': phase, 'candidate_dir': str(Path(candidate_dir).resolve()),
           'opponent_dir': str(Path(opponent_dir).resolve()), 'sdk_dir': str(Path(sdk_dir).resolve()),
           'output_dir': str(output), 'config': config, 'seeds': seeds, 'agents': metadata,
           'python': sys.version, 'platform': platform.platform(), 'sdk_hashes': sdk_hashes}
    save_json(output / 'job.json', job)
    command = [sys.executable, str(REPO_ROOT / 'team_a/scripts/evaluate.py'), '--worker', str(output / 'job.json')]
    process = {'returncode': None, 'timed_out': False, 'error': None, 'command': command}
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    with tempfile.TemporaryDirectory(prefix='ptcg_worker_') as cwd:
        try:
            with (output / 'stdout.txt').open('w') as stdout, (output / 'stderr.txt').open('w') as stderr:
                completed = subprocess.run(command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr,
                                           timeout=config['process_timeout_seconds'], check=False)
            process['returncode'] = completed.returncode
        except subprocess.TimeoutExpired:
            process['timed_out'] = True
        except Exception as error:
            process['error'] = str(error)
    save_json(output / 'process.json', process)
    report_path = output / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    passed = (process['returncode'] == 0 and not process['timed_out'] and report.get('all_passed') is True
              and len(report.get('battles', [])) == len(seeds))
    summary_path = output / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'valid_battles': 0}
    summary['local_checks_passed'] = passed
    save_json(summary_path, summary)
    save_json(output / 'completion.json', {'passed': passed, 'expected_battles': len(seeds), 'process': process})
    return passed, summary


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == '--worker':
        return worker(Path(argv[1]))
    parser = argparse.ArgumentParser(description='Compare standalone agents in isolated direct-SDK individual battles.')
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument('--candidate', help='Agent name in the configuration, such as ar2.')
    candidate.add_argument('--agent-dir', type=Path, help='Custom folder with standalone main.py and assets.')
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument('--opponent', help='Configured opponent name.')
    opponent.add_argument('--opponent-dir', type=Path)
    parser.add_argument('--asset', action='append', default=[], help='Additional candidate asset path relative to its entry point.')
    parser.add_argument('--opponent-asset', action='append', default=[])
    parser.add_argument('--sdk-dir', type=Path, required=True)
    parser.add_argument('--deck-file', type=Path, help='Use this deck for both policies; defaults to SDK deck.csv when present.')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--games', type=int)
    parser.add_argument('--seed-base', type=int)
    args = parser.parse_args(argv)
    output = None
    try:
        config = load_config(args.config)
        seeds = select_seeds(config, args.games, args.seed_base)
        candidate = resolve_agent(config, args.candidate, args.agent_dir, args.asset)
        opponent = resolve_agent(config, args.opponent, args.opponent_dir, args.opponent_asset)
        output = new_output(args.output_dir, [candidate['entrypoint'].parent, opponent['entrypoint'].parent])
        deck = args.deck_file or (args.sdk_dir / 'deck.csv' if (args.sdk_dir / 'deck.csv').is_file() else None)
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
        with tempfile.TemporaryDirectory(prefix='ptcg_agents_') as temporary:
            first, second = Path(temporary) / 'candidate', Path(temporary) / 'opponent'
            first_info = stage_agent(candidate, first, deck)
            second_info = stage_agent(opponent, second, deck)
            verify_decks(first_info, second_info, config)
            passed, summary = run_pair(first, second, args.sdk_dir, output, config, seeds, run_id,
                                       'evaluation', {'candidate': first_info, 'opponent': second_info})
        print(json.dumps(summary, indent=2))
        return 0 if passed else 1
    except Exception as error:
        if output is not None:
            save_json(output / 'failure.json', {'passed': False, 'type': type(error).__name__, 'message': str(error)})
        print(type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1
