"""Train one linear scorer using complete game groups and training only scaling."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import ar1_features as features


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def prepare(collection, output, split_seed):
    report = json.loads((collection / 'report.json').read_text())
    if report.get('all_passed') is not True:
        raise ValueError('Collection did not pass. Preserve logs and rerun collection before fitting.')
    groups, excluded = {}, []
    for game in report['battles']:
        if game['status'] != 'passed' or not game['finished'] or game.get('demo_cap_reached'):
            excluded.append(game['game_id'])
            continue
        path = collection / game['demo_file']
        raw_rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(raw_rows) != game['demo_records']:
            raise ValueError('Collection count mismatch.')
        examples = []
        for row in raw_rows:
            observation = row['observation']
            if row['game_id'] != game['game_id'] or not row['selection_accepted']:
                raise ValueError('Invalid record provenance.')
            if observation['current']['yourIndex'] != row['acting_player']:
                raise ValueError('Acting player mismatch.')
            if set(observation) - {'current', 'select', 'logs'}:
                raise ValueError('Unexpected observation payload.')
            if not features.supported_main(observation):
                continue
            matrix = np.asarray(features.candidate_features(observation), dtype=np.float64)
            scores = np.asarray(row['teacher_scores'], dtype=np.float64)
            chosen = row['chosen_option_index']
            if scores.shape != (len(matrix),) or not np.isfinite(scores).all():
                raise ValueError('Invalid teacher labels.')
            if type(chosen) is not int or chosen not in range(len(matrix)) or scores[chosen] != scores.max():
                raise ValueError('Chosen option is not a teacher maximum.')
            if row['legal_option_indices'] != list(range(len(matrix))):
                raise ValueError('Legal option mapping mismatch.')
            target = (scores == scores.max()).astype(np.float64)
            target /= target.sum()
            examples.append({'x': matrix, 'target': target, 'chosen': chosen, 'row': row})
        if examples:
            groups[game['game_id']] = examples
        else:
            excluded.append(game['game_id'])
    total = sum(map(len, groups.values()))
    if len(groups) < 100 or total < 500:
        raise ValueError(f'Only {len(groups)} eligible games and {total} decisions. Collect more games before fitting.')
    ids = sorted(groups)
    random.Random(split_seed).shuffle(ids)
    held_count = max(1, int(len(ids) * .15))
    validation_count = max(1, int(len(ids) * .15))
    partitions = {'held_out': ids[:held_count],
                  'validation': ids[held_count:held_count + validation_count],
                  'train': ids[held_count + validation_count:]}
    manifest = {'schema_version': features.SCHEMA_VERSION, 'grouping_key': 'game_id',
                'split_seed': split_seed, 'source': 'fresh_collection_only',
                'excluded_game_ids': excluded, 'teacher_id': 'A-HEUR-01',
                'target': 'Uniform probability over teacher maximum score options',
                'notes': 'No earlier Day 3 data or evaluation games are reused. Losses are eligible. Forced one option decisions are excluded. Exact option ordering is not a feature.'}
    datasets = {}
    for split, game_ids in partitions.items():
        datasets[split] = [example for game_id in game_ids for example in groups[game_id]]
        manifest[split] = {'game_ids': game_ids, 'games': len(game_ids), 'rows': len(datasets[split])}
        with (output / (split + '.jsonl')).open('w') as handle:
            for example in datasets[split]:
                handle.write(json.dumps(example['row']) + '\n')
    assert not (set(partitions['train']) & set(partitions['validation'])
                or set(partitions['train']) & set(partitions['held_out'])
                or set(partitions['validation']) & set(partitions['held_out']))
    save_json(output / 'dataset_manifest.json', manifest)
    return datasets, manifest


def pack(examples):
    sizes = np.array([len(example['x']) for example in examples], dtype=np.int64)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    return (np.concatenate([example['x'] for example in examples]),
            np.concatenate([example['target'] for example in examples]), sizes, starts)


def distribution(scores, sizes, starts):
    maximum = np.maximum.reduceat(scores, starts)
    shifted = scores - np.repeat(maximum, sizes)
    exponentials = np.exp(shifted)
    totals = np.add.reduceat(exponentials, starts)
    probabilities = exponentials / np.repeat(totals, sizes)
    log_probabilities = shifted - np.repeat(np.log(totals), sizes)
    return probabilities, log_probabilities


def metrics(examples, weights):
    x, target, sizes, starts = pack(examples)
    scores = x @ weights
    _, logp = distribution(scores, sizes, starts)
    hits, exact = [], []
    for index, (start, size) in enumerate(zip(starts, sizes)):
        predicted = int(np.argmax(scores[start:start+size]))
        hits.append(bool(examples[index]['target'][predicted] > 0))
        exact.append(predicted == examples[index]['chosen'])
    return {'decisions': len(examples), 'cross_entropy': float(-target @ logp / len(examples)),
            'teacher_top_set_accuracy': float(np.mean(hits)), 'exact_choice_accuracy': float(np.mean(exact)),
            'uniform_choice_top_set_accuracy': float(np.mean([np.count_nonzero(e['target']) / len(e['target']) for e in examples])),
            'uniform_cross_entropy': float(np.log(sizes).mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError('Epochs must be positive.')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    data, manifest = prepare(Path(args.collection), output, 2027)
    train_matrix = np.concatenate([example['x'] for example in data['train']])
    mean = train_matrix.mean(axis=0)
    scale = train_matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    for examples in data.values():
        for example in examples:
            example['x'] = (example['x'] - mean) / scale
    save_json(output / 'feature_contract.json', {
        'schema_version': features.SCHEMA_VERSION, 'feature_names': list(features.FEATURE_NAMES),
        'feature_dimension': len(features.FEATURE_NAMES), 'supported_context': 'MAIN, minCount=maxCount=1, at least two options',
        'model': 'One shared linear dot product per candidate; softmax over current legal candidates during fitting',
        'state_interactions': 'Action type multiplied by visible state values; state-only additive terms are omitted',
        'normalization': 'Mean and standard deviation fitted on training candidate rows only',
        'excluded_inputs': ['teacher_scores', 'chosen_option_index', 'game_id', 'step', 'result', 'logs', 'serial', 'search_begin_input', 'opponent hidden card identities', 'option list position'],
        'fallback': 'Frozen A-HEUR-01 for unsupported selections',
        'teacher_limit': 'Imitation of handcrafted preferences; no optimality or game reward objective',
    })
    histories, candidates = [], []
    for regularization in (0.0, 1e-4, 1e-3):
        rng = np.random.default_rng(args.seed)
        weights = np.zeros(len(mean), dtype=np.float64)
        first_moment, second_moment = np.zeros_like(weights), np.zeros_like(weights)
        updates, stale = 0, 0
        best_loss, best_weights, best_epoch = math.inf, None, None
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(len(data['train']))
            for offset in range(0, len(order), 64):
                examples = [data['train'][int(i)] for i in order[offset:offset+64]]
                x, target, sizes, starts = pack(examples)
                probability, _ = distribution(x @ weights, sizes, starts)
                gradient = x.T @ (probability - target) / len(examples) + regularization * weights
                updates += 1
                first_moment = .9 * first_moment + .1 * gradient
                second_moment = .999 * second_moment + .001 * gradient * gradient
                weights -= .03 * (first_moment / (1 - .9**updates)) / (np.sqrt(second_moment / (1 - .999**updates)) + 1e-8)
                if not np.isfinite(weights).all():
                    raise ValueError('Training produced nonfinite weights.')
            train_result = metrics(data['train'], weights)
            validation = metrics(data['validation'], weights)
            histories.append({'regularization': regularization, 'epoch': epoch,
                              'train_cross_entropy': train_result['cross_entropy'],
                              'validation_cross_entropy': validation['cross_entropy'],
                              'validation_top_set_accuracy': validation['teacher_top_set_accuracy']})
            if validation['cross_entropy'] < best_loss - 1e-6:
                best_loss, best_weights, best_epoch, stale = validation['cross_entropy'], weights.copy(), epoch, 0
            else:
                stale += 1
            if epoch == 1 or epoch % 10 == 0:
                print('L2', regularization, 'epoch', epoch, 'validation loss', round(validation['cross_entropy'], 4), flush=True)
            if stale >= 10:
                break
        candidate = {'l2': regularization, 'best_epoch': best_epoch, 'validation_cross_entropy': best_loss,
                     'weights': best_weights}
        candidates.append(candidate)
        save_json(output / ('checkpoint_' + str(regularization) + '.json'),
                  {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in candidate.items()})
        with (output / 'training_history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(histories[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(histories)
    selected = min(candidates, key=lambda item: item['validation_cross_entropy'])
    weights = selected['weights']
    model = {'model_id': 'A-R1', 'schema_version': features.SCHEMA_VERSION,
             'feature_names': list(features.FEATURE_NAMES), 'mean': mean.tolist(), 'scale': scale.tolist(),
             'weights': weights.tolist(), 'l2': selected['l2'], 'best_epoch': selected['best_epoch'],
             'training_seed': args.seed, 'training_decisions': len(data['train']),
             'target': 'Uniform over teacher maximum score options',
             'selection': 'Lowest validation cross entropy; held out games not used for selection'}
    save_json(output / 'model.json', model)
    results = {'model_id': 'A-R1', 'fitting_passed': True, 'selected_l2': selected['l2'],
               'selected_epoch': selected['best_epoch'], 'training_seconds': time.perf_counter() - started,
               'splits': {split: metrics(examples, weights) for split, examples in data.items()},
               'configurations': [{key: value for key, value in candidate.items() if key != 'weights'} for candidate in candidates],
               'model_sha256': hashlib.sha256((output / 'model.json').read_bytes()).hexdigest(),
               'notes': 'Held out metrics were calculated only after model selection. Decision accuracy is descriptive; decisions within games are dependent. Agreement with the teacher is not game winning performance.'}
    save_json(output / 'metrics.json', results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
