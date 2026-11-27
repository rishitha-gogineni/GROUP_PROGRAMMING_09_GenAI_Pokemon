"""Fit A-R2 on the exact A-R1 feature definitions and game groups."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import ar1_features as features
from candidate_training import save_json, pack, distribution, linear_metrics


def load_data(parent, output):
    manifest = json.loads((parent / 'training/dataset_manifest.json').read_text())
    baseline = json.loads((parent / 'training/model.json').read_text())
    assert baseline['model_id'] == 'A-R1' and baseline['schema_version'] == features.SCHEMA_VERSION
    assert baseline['feature_names'] == list(features.FEATURE_NAMES)
    partitions, id_sets, keys = {}, [], set()
    for split in ['train', 'validation', 'held_out']:
        entry = manifest[split]
        ids = set(entry['game_ids'])
        rows = [json.loads(line) for line in (parent / 'training' / (split + '.jsonl')).read_text().splitlines() if line.strip()]
        assert len(rows) == entry['rows'] and len(ids) == entry['games']
        assert {row['game_id'] for row in rows} == ids
        examples = []
        for row in rows:
            key = row['game_id'], row['step']
            assert key not in keys
            keys.add(key)
            observation = row['observation']
            assert features.supported_main(observation) and row['selection_accepted']
            assert observation['current']['yourIndex'] == row['acting_player']
            assert not (set(observation) - {'current', 'select', 'logs'})
            matrix = np.asarray(features.candidate_features(observation), dtype=np.float64)
            scores = np.asarray(row['teacher_scores'], dtype=np.float64)
            chosen = row['chosen_option_index']
            assert np.isfinite(matrix).all() and scores.shape == (len(matrix),) and np.isfinite(scores).all()
            assert type(chosen) is int and chosen in range(len(matrix)) and scores[chosen] == scores.max()
            assert row['legal_option_indices'] == list(range(len(matrix)))
            target = (scores == scores.max()).astype(np.float64)
            target /= target.sum()
            examples.append({'x': matrix, 'target': target, 'chosen': chosen, 'row': row})
        partitions[split] = examples
        id_sets.append(ids)
    assert not (id_sets[0]&id_sets[1] or id_sets[0]&id_sets[2] or id_sets[1]&id_sets[2])
    train_matrix = np.concatenate([example['x'] for example in partitions['train']])
    computed_mean, computed_scale = train_matrix.mean(axis=0), train_matrix.std(axis=0)
    active = computed_scale >= 1e-6
    computed_scale[~active] = 1.0
    mean, scale = np.asarray(baseline['mean']), np.asarray(baseline['scale'])
    assert np.allclose(computed_mean, mean, rtol=1e-12, atol=1e-12)
    assert np.allclose(computed_scale, scale, rtol=1e-12, atol=1e-12)
    assert active.any() and np.isfinite(mean).all() and np.isfinite(scale).all() and scale.min() > 0
    for examples in partitions.values():
        for example in examples:
            example['x'] = (example['x'] - mean) / scale
    shutil.copy2(parent / 'training/dataset_manifest.json', output / 'dataset_manifest.json')
    save_json(output / 'data_audit.json', {
        'passed': True, 'schema_version': features.SCHEMA_VERSION, 'game_groups_reused_without_changes': True,
        'same_ar1_normalization': True, 'game_groups_disjoint': True,
        'active_training_features': int(active.sum()),
        'splits': {split: {'games': manifest[split]['games'], 'rows': len(examples)} for split, examples in partitions.items()},
        'note': 'Existing A-R1 test groups are reused for the planned architecture comparison. This is not a new unseen test set.'})
    return partitions, baseline, mean, scale, active


def initialize(dimension, hidden, active, seed):
    rng = np.random.default_rng(seed)
    limit = np.sqrt(6.0 / (int(active.sum()) + hidden))
    weights = rng.uniform(-limit, limit, (dimension, hidden))
    weights[~active] = 0.0
    return {'W1': weights, 'b1': np.zeros(hidden), 'w2': rng.normal(0, 1 / np.sqrt(hidden), hidden)}


def forward(x, parameters):
    hidden = np.tanh(x @ parameters['W1'] + parameters['b1'])
    return hidden @ parameters['w2'], hidden


def loss_and_gradients(examples, parameters, regularization):
    x, target, sizes, starts = pack(examples)
    scores, hidden = forward(x, parameters)
    probability, logp = distribution(scores, sizes, starts)
    derivative = (probability - target) / len(examples)
    hidden_derivative = derivative[:, None] * parameters['w2'][None, :] * (1 - hidden * hidden)
    gradients = {
        'W1': x.T @ hidden_derivative + regularization * parameters['W1'],
        'b1': hidden_derivative.sum(axis=0),
        'w2': hidden.T @ derivative + regularization * parameters['w2'],
    }
    penalty = .5 * regularization * (np.square(parameters['W1']).sum() + np.square(parameters['w2']).sum())
    loss = float(-target @ logp / len(examples) + penalty)
    return loss, gradients


def evaluate(examples, parameters):
    x, target, sizes, starts = pack(examples)
    scores, _ = forward(x, parameters)
    _, logp = distribution(scores, sizes, starts)
    top, exact = [], []
    for index, (start, size) in enumerate(zip(starts, sizes)):
        chosen = int(np.argmax(scores[start:start + size]))
        top.append(bool(examples[index]['target'][chosen] > 0))
        exact.append(chosen == examples[index]['chosen'])
    return {'decisions': len(examples), 'cross_entropy': float(-target @ logp / len(examples)),
            'teacher_top_set_accuracy': float(np.mean(top)), 'exact_choice_accuracy': float(np.mean(exact)),
            'uniform_choice_top_set_accuracy': float(np.mean([np.count_nonzero(e['target']) / len(e['target']) for e in examples])),
            'uniform_cross_entropy': float(np.log(sizes).mean())}


def gradient_check(examples, active, output):
    parameters = initialize(len(features.FEATURE_NAMES), 4, active, 123)
    small = examples[:3]
    _, gradients = loss_and_gradients(small, parameters, 1e-3)
    checks = []
    epsilon = 1e-5
    for key in parameters:
        order = np.argsort(np.abs(gradients[key]).ravel())[-4:]
        for flat_index in order:
            index = np.unravel_index(int(flat_index), parameters[key].shape)
            original = parameters[key][index]
            parameters[key][index] = original + epsilon
            above, _ = loss_and_gradients(small, parameters, 1e-3)
            parameters[key][index] = original - epsilon
            below, _ = loss_and_gradients(small, parameters, 1e-3)
            parameters[key][index] = original
            numeric, analytic = (above - below) / (2 * epsilon), float(gradients[key][index])
            error = abs(numeric - analytic)
            assert error <= 1e-6 + 1e-4 * abs(analytic), (key, index, analytic, numeric)
            checks.append({'parameter': key, 'index': [int(i) for i in index], 'absolute_error': error})
    report = {'passed': True, 'checks': checks, 'source': 'First three training decisions only',
              'maximum_absolute_error': max(check['absolute_error'] for check in checks)}
    save_json(output / 'gradient_check.json', report)
    print('Gradient check passed.', flush=True)


def serial(parameters):
    return {key: value.tolist() for key, value in parameters.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert args.hidden > 0 and args.epochs > 0
    output, parent = Path(args.output).resolve(), Path(args.parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    started = time.perf_counter()
    data, baseline, mean, scale, active = load_data(parent, output)
    gradient_check(data['train'], active, output)
    learning_rate = .003
    histories, candidates = [], []
    for regularization in (0.0, 1e-4, 1e-3):
        parameters = initialize(len(mean), args.hidden, active, args.seed)
        first = {key: np.zeros_like(value) for key, value in parameters.items()}
        second = copy.deepcopy(first)
        rng = np.random.default_rng(args.seed)
        best_loss, best_parameters, best_epoch, stale, updates = float('inf'), None, None, 0, 0
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(len(data['train']))
            for offset in range(0, len(order), 64):
                batch = [data['train'][int(i)] for i in order[offset:offset + 64]]
                _, gradients = loss_and_gradients(batch, parameters, regularization)
                updates += 1
                for key in parameters:
                    first[key] = .9 * first[key] + .1 * gradients[key]
                    second[key] = .999 * second[key] + .001 * np.square(gradients[key])
                    parameters[key] -= learning_rate * (first[key] / (1 - .9**updates)) / (np.sqrt(second[key] / (1 - .999**updates)) + 1e-8)
                    assert np.isfinite(parameters[key]).all(), 'Nonfinite MLP parameters.'
                parameters['W1'][~active] = 0.0
            train_result, validation = evaluate(data['train'], parameters), evaluate(data['validation'], parameters)
            histories.append({'l2': regularization, 'epoch': epoch,
                              'train_cross_entropy': train_result['cross_entropy'],
                              'validation_cross_entropy': validation['cross_entropy'],
                              'validation_top_set_accuracy': validation['teacher_top_set_accuracy']})
            if validation['cross_entropy'] < best_loss - 1e-6:
                best_loss, best_parameters, best_epoch, stale = validation['cross_entropy'], copy.deepcopy(parameters), epoch, 0
                save_json(checkpoint_dir / ('l2_' + str(regularization) + '.json'),
                          {'l2': regularization, 'epoch': epoch, 'validation_cross_entropy': best_loss, **serial(best_parameters)})
            else:
                stale += 1
            with (output / 'training_history.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(histories[0]), lineterminator='\n')
                writer.writeheader()
                writer.writerows(histories)
            save_json(output / 'training_progress.json', {'l2': regularization, 'epoch': epoch,
                      'best_validation_cross_entropy': best_loss, 'best_epoch': best_epoch})
            if epoch == 1 or epoch % 10 == 0:
                print('L2', regularization, '| epoch', epoch, '| validation loss', round(validation['cross_entropy'], 5), flush=True)
            if stale >= 10:
                break
        candidates.append({'l2': regularization, 'best_epoch': best_epoch,
                           'validation_cross_entropy': best_loss, 'parameters': best_parameters})
    selected = min(candidates, key=lambda trial: trial['validation_cross_entropy'])
    parameters = selected['parameters']
    model = {'model_id': 'A-R2', 'schema_version': features.SCHEMA_VERSION, 'feature_names': list(features.FEATURE_NAMES),
             'activation': 'tanh', 'hidden_units': args.hidden, 'mean': mean.tolist(), 'scale': scale.tolist(),
             **serial(parameters), 'l2': selected['l2'], 'best_epoch': selected['best_epoch'],
             'training_seed': args.seed, 'learning_rate': learning_rate, 'training_decisions': len(data['train']),
             'active_training_features': int(active.sum()), 'parent_ar1_model_sha256': hashlib.sha256((parent / 'training/model.json').read_bytes()).hexdigest(),
             'target': 'Uniform over teacher maximum score options',
             'selection': 'Lowest validation cross entropy; no test result used for selection'}
    save_json(output / 'model.json', model)
    test_metrics = {split: evaluate(examples, parameters) for split, examples in data.items()}
    linear = {split: linear_metrics(examples, np.asarray(baseline['weights'])) for split, examples in data.items()}
    old_metrics = json.loads((parent / 'training/metrics.json').read_text())
    for split in linear:
        for key in ['cross_entropy', 'teacher_top_set_accuracy', 'exact_choice_accuracy']:
            assert np.isclose(linear[split][key], old_metrics['splits'][split][key], rtol=1e-10, atol=1e-10)
    result = {'model_id': 'A-R2', 'fitting_passed': True, 'selected_l2': selected['l2'], 'selected_epoch': selected['best_epoch'],
              'hidden_units': args.hidden, 'activation': 'tanh', 'learning_rate': learning_rate,
              'parameter_count': int(sum(value.size for value in parameters.values())),
              'training_seconds': time.perf_counter() - started, 'splits': test_metrics, 'ar1_same_splits': linear,
              'configurations': [{key: value for key, value in trial.items() if key != 'parameters'} for trial in candidates],
              'model_sha256': hashlib.sha256((output / 'model.json').read_bytes()).hexdigest(),
              'notes': 'Checkpoint and regularization selection used validation only. Shared A-R1 test groups are a planned architecture comparison, not a new unseen test set. Teacher agreement does not establish stronger gameplay. A-R1 used learning rate 0.03; A-R2 uses 0.003, with the same Adam family, batch size, seed, maximum epochs, patience and regularization grid.'}
    save_json(output / 'metrics.json', result)
    contract = json.loads((parent / 'training/feature_contract.json').read_text())
    contract.update(model=f'{len(mean)} input features, one {args.hidden}-unit tanh hidden layer, scalar score per legal candidate',
                    normalization='Exact fitted A-R1 training normalization; independently reconciled with the same training rows',
                    architecture='Small MLP; no output bias because it would cancel within each candidate set',
                    target='Uniform probability over teacher maximum score options',
                    constant_features='Input weight rows with no training variance are fixed at zero')
    save_json(output / 'feature_contract.json', contract)
    with (output / 'imitation_comparison.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model_id', 'split', 'decisions', 'cross_entropy', 'teacher_top_set_accuracy', 'exact_choice_accuracy'], lineterminator='\n')
        writer.writeheader()
        for model_id, results in [('A-R1', linear), ('A-R2', test_metrics)]:
            for split, values in results.items():
                writer.writerow({'model_id': model_id, 'split': split, **{key: values[key] for key in ['decisions', 'cross_entropy', 'teacher_top_set_accuracy', 'exact_choice_accuracy']}})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
