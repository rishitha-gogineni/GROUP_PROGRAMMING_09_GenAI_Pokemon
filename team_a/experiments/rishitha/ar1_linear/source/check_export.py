"""Check the exported standard library policy against the fitted linear model."""
import argparse
import json
from pathlib import Path

import numpy as np
from evaluate_policy import load_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    directory = Path(args.agent).resolve()
    training = Path(args.training).resolve()
    callback, namespace = load_agent(directory)
    model = json.loads((directory / 'model.json').read_text())
    mean, scale, weights = [np.asarray(model[key], dtype=np.float64) for key in ('mean', 'scale', 'weights')]
    initial = {'select': None, 'current': None, 'logs': [], 'search_begin_input': None}
    expected = [int(line) for line in (directory / 'deck.csv').read_text().splitlines() if line.strip()]
    assert callback(initial) == expected and len(expected) == 60
    checks = {}
    largest_error = 0.0
    for split in ['train', 'validation', 'held_out']:
        count = 0
        with (training / (split + '.jsonl')).open() as handle:
            for line in handle:
                if count >= 100:
                    break
                row = json.loads(line)
                observation = row['observation']
                matrix = np.asarray(namespace['candidate_features'](observation), dtype=np.float64)
                expected_scores = ((matrix - mean) / scale) @ weights
                actual = np.asarray(namespace['linear_scores'](observation, {'__raw_path__': str(directory / 'main.py')}))
                assert np.allclose(expected_scores, actual, rtol=1e-8, atol=1e-8)
                choice = callback(observation)
                assert len(choice) == 1 and type(choice[0]) is int and choice[0] in range(len(matrix))
                assert np.isclose(expected_scores[choice[0]], expected_scores.max(), rtol=1e-8, atol=1e-8)
                largest_error = max(largest_error, float(np.max(np.abs(expected_scores - actual))))
                count += 1
        assert count > 0
        checks[split] = count
    assert namespace['POLICY_USAGE']['learned'] > 0
    report = {'passed': True, 'tested_decisions': checks, 'maximum_score_difference': largest_error,
              'deck_callback_without_file': True, 'policy_usage': namespace['POLICY_USAGE'],
              'note': 'Parity checks use saved observations after fitting. They do not select or refit the model.'}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
